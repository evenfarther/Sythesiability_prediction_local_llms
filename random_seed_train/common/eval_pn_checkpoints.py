#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate P/N checkpoints on /DATA/npj_compt.mat_project_github/data/valid_llm_pn.jsonl.

This is a P/N analogue of evaluation/eval_pu_checkpoints_fixed.py:
- Uses log-sum-exp difference between P and N token logits on last token
- Supports LoRA adapter checkpoints under a single directory
- Writes per-checkpoint metrics CSV for model selection
"""

import os
import re
import json
import math
import csv
import gc
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import Counter

import torch
from torch.utils.data import DataLoader

try:
    from unsloth import FastLanguageModel
    from peft import PeftModel
except Exception as e:
    raise RuntimeError("Missing dependencies: install 'unsloth' and 'peft' to run this evaluator.") from e


SYSTEM_MSG = {
    "role": "system",
    "content": "You are a materials science assistant. Given a chemical composition, answer only with 'P' (synthesizable/positive) or 'N' (non-synthesizable/negative).",
}


def cuda_mem_summary() -> str:
    if not torch.cuda.is_available():
        return "cuda: not available"
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = free_b / (1024**3)
        total_gb = total_b / (1024**3)
        alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        return f"cuda_mem free={free_gb:.2f}GiB total={total_gb:.2f}GiB allocated={alloc_gb:.2f}GiB reserved={reserved_gb:.2f}GiB"
    except Exception as e:
        return f"cuda_mem: unavailable ({e})"


def is_cublas_init_error(e: BaseException) -> bool:
    msg = str(e)
    return (
        "CUBLAS_STATUS_NOT_INITIALIZED" in msg
        or "cublasSgemm" in msg
        or "cublasCreate" in msg
        or "CUBLAS_STATUS_ALLOC_FAILED" in msg
    )


def warmup_cublas(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        # Trigger cuBLAS init early to fail-fast on broken GPU/CUDA setups.
        a = torch.randn((256, 256), device=device, dtype=torch.float32)
        b = torch.randn((256, 256), device=device, dtype=torch.float32)
        _ = a @ b
        torch.cuda.synchronize()
        print(f"[Preflight] cuBLAS GEMM ok ({cuda_mem_summary()})")
    except Exception as e:
        print(f"[FATAL] cuBLAS GEMM preflight failed: {e} ({cuda_mem_summary()})")
        raise


def last_token_ids(tokenizer, c: str) -> List[int]:
    ids = set()
    for s in (c, " " + c, "\n" + c):
        enc = tokenizer.encode(s, add_special_tokens=False)
        if enc:
            ids.add(enc[-1])
    return sorted(ids)


def verify_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    p_ids = last_token_ids(tokenizer, "P")
    n_ids = last_token_ids(tokenizer, "N")

    print(f"[Verify] P token IDs: {p_ids}")
    print(f"[Verify] N token IDs: {n_ids}")

    if not p_ids or not n_ids:
        raise RuntimeError("Failed to obtain token ids for 'P'/'N'. Check tokenizer.")

    return p_ids, n_ids


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    lengths: torch.Tensor


def collate_pad(batch, pad_id: int) -> Tuple[EncodedBatch, List[str]]:
    max_len = max(len(x) for x, _ in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    lengths = torch.zeros((batch_size,), dtype=torch.long)
    labels: List[str] = []

    for i, (ids, y) in enumerate(batch):
        seq_len = len(ids)
        input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = 1
        lengths[i] = seq_len
        labels.append(y)

    return EncodedBatch(input_ids=input_ids, attention_mask=attention_mask, lengths=lengths), labels


def read_jsonl_msgs(path: str) -> List[Dict]:
    data: List[Dict] = []
    print(f"[Data] Reading {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Validation file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "messages" not in obj:
                    print(f"[WARN] Line {line_num}: missing 'messages' field")
                    continue
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON decode error: {e}")
                continue

    print(f"[Data] Loaded {len(data)} valid samples")
    return data


def prepare_messages_for_eval(msgs: List[Dict]) -> List[Dict]:
    if not msgs:
        return [SYSTEM_MSG]

    if msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]

    if not msgs or msgs[0].get("role") != "system":
        msgs = [SYSTEM_MSG] + msgs

    return msgs


def tokenize_dataset(tokenizer, data: List[Dict], max_seq_len: int, cache_path: str = None):
    if cache_path and os.path.exists(cache_path):
        print(f"[Cache] Loading from {cache_path}")
        ckpt = torch.load(cache_path, map_location="cpu")
        return ckpt["encoded_inputs"], ckpt["labels"]

    encoded_inputs = []
    labels: List[str] = []

    print(f"[Tokenize] Processing {len(data)} samples...")
    for i, obj in enumerate(data):
        if i % 1000 == 0:
            print(f"  {i}/{len(data)} processed")

        msgs = obj["messages"]
        if not msgs or msgs[-1].get("role") != "assistant":
            print(f"[WARN] Sample {i}: Invalid format, skipping")
            continue

        gold = msgs[-1]["content"].strip()
        if gold not in ["P", "N"]:
            print(f"[WARN] Sample {i}: Invalid label '{gold}', skipping")
            continue

        labels.append(gold)
        eval_msgs = prepare_messages_for_eval(msgs)

        # Some chat templates (e.g., Qwen/DeepSeek families) support `enable_thinking`.
        # For P/N classification we explicitly disable it when available to avoid extra tokens.
        try:
            ids = tokenizer.apply_chat_template(
                eval_msgs,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors=None,
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                eval_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )

        if len(ids) > max_seq_len:
            ids = ids[-max_seq_len:]

        encoded_inputs.append(torch.tensor(ids, dtype=torch.long))

    print(f"[Tokenize] Encoded {len(encoded_inputs)} valid samples")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"encoded_inputs": encoded_inputs, "labels": labels}, cache_path)
        print(f"[Cache] Saved to {cache_path}")

    return encoded_inputs, labels


def confusion_and_metrics(golds: List[str], preds: List[str], scores: List[float] = None) -> Dict[str, float]:
    cm = Counter()
    for g, p in zip(golds, preds):
        cm[(g, p)] += 1

    tp = cm[("P", "P")]
    tn = cm[("N", "N")]
    fp = cm[("N", "P")]
    fn = cm[("P", "N")]
    total = tp + tn + fp + fn

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    acc = safe_div(tp + tn, total)
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    fpr = 1.0 - tnr
    fnr = 1.0 - tpr
    prec_p = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)

    f1_p = safe_div(2 * tp, 2 * tp + fp + fn)
    f1_n = safe_div(2 * tn, 2 * tn + fp + fn)

    bal_acc = 0.5 * (tpr + tnr)
    err_rate = 1.0 - acc

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    auroc = None
    if scores is not None and len(set(golds)) == 2:
        labels = [1 if g == "P" else 0 for g in golds]
        paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

        pos_count = sum(labels)
        neg_count = len(labels) - pos_count

        if pos_count > 0 and neg_count > 0:
            tp_c = fp_c = 0
            prev_score = None
            roc_points = [(0.0, 0.0)]

            for s, y in paired:
                if prev_score is not None and s != prev_score:
                    roc_points.append((fp_c / neg_count, tp_c / pos_count))
                if y == 1:
                    tp_c += 1
                else:
                    fp_c += 1
                prev_score = s

            roc_points.append((1.0, 1.0))

            auc = 0.0
            for i in range(1, len(roc_points)):
                x1, y1 = roc_points[i - 1]
                x2, y2 = roc_points[i]
                auc += (x2 - x1) * (y1 + y2) / 2.0

            auroc = max(0.0, min(1.0, auc))

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "total": total,
        "accuracy": acc,
        "TPR_recall_P": tpr,
        "TNR_specificity": tnr,
        "FPR": fpr,
        "FNR": fnr,
        "precision_P": prec_p,
        "NPV": npv,
        "F1_P": f1_p,
        "F1_N": f1_n,
        "MCC": mcc,
        "balanced_accuracy": bal_acc,
        "error_rate": err_rate,
        "AUROC": auroc,
    }


def is_peft_adapter_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    return "adapter_config.json" in files or any(fn.startswith("adapter") for fn in files)


def resolve_peft_adapter_dir(path: str) -> Optional[str]:
    """
    Returns a directory path that can be passed to `PeftModel.from_pretrained`.

    Most checkpoints in this project save adapter files directly under the checkpoint directory.
    However, some TRL/PEFT flows (e.g., named adapters) may save adapter files under a subdirectory
    like `checkpoint-*/resume/adapter_config.json`.
    """
    if is_peft_adapter_dir(path):
        return path

    if not os.path.isdir(path):
        return None

    try:
        candidates: List[str] = []
        for name in sorted(os.listdir(path)):
            sub = os.path.join(path, name)
            if is_peft_adapter_dir(sub):
                candidates.append(sub)

        if not candidates:
            return None

        # Prefer common adapter directory names used in this repo.
        for preferred in ("resume", "default", "adapter"):
            for c in candidates:
                if os.path.basename(c) == preferred:
                    return c

        return candidates[0]
    except Exception as e:
        print(f"[WARN] Failed to resolve adapter dir under {path}: {e}")
        return None


INT_FIELDS = {"TP", "TN", "FP", "FN", "total"}


def _coerce_existing_row_types(row: Dict[str, str]) -> Dict:
    out: Dict = dict(row)
    for k, v in row.items():
        if k in ("checkpoint", "path"):
            continue
        if v is None:
            out[k] = None
            continue

        s = str(v).strip()
        if s == "" or s.lower() == "none":
            out[k] = None
            continue

        if k in INT_FIELDS:
            try:
                out[k] = int(float(s))
            except ValueError:
                out[k] = None
            continue

        try:
            out[k] = float(s)
        except ValueError:
            out[k] = None

    return out


def load_existing_results_csv(path: str) -> Tuple[List[Dict], List[str], set]:
    if not os.path.exists(path):
        return [], [], set()

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [_coerce_existing_row_types(r) for r in reader]

    existing_paths = {r.get("path") for r in rows if r.get("path")}
    return rows, fieldnames, existing_paths


def write_results_csv(path: str, results: List[Dict], fieldnames_hint: List[str] = None) -> None:
    if not results:
        raise RuntimeError("Refusing to write empty results CSV.")

    fieldnames = list(fieldnames_hint or [])
    for r in results:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def evaluate_single_checkpoint(
    model,
    loader,
    p_ids: torch.Tensor,
    n_ids: torch.Tensor,
    device: str,
    checkpoint_name: str,
    threshold: float = 0.0,
) -> Tuple[List[str], List[float]]:
    print(f"[Eval] Evaluating {checkpoint_name}...")

    model.eval()
    preds: List[str] = []
    scores: List[float] = []

    with torch.inference_mode():
        for batch_idx, (enc, labels) in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}")

            enc = EncodedBatch(
                input_ids=enc.input_ids.to(device),
                attention_mask=enc.attention_mask.to(device),
                lengths=enc.lengths.to(device),
            )

            try:
                # OOM mitigation: disable KV cache; use `logits_to_keep` when supported.
                forward_kwargs = {
                    "input_ids": enc.input_ids,
                    "attention_mask": enc.attention_mask,
                    "use_cache": False,
                }

                seq_len = enc.input_ids.shape[1]
                pos = enc.lengths - 1  # [B]
                min_pos = int(pos.min().item())
                required_keep = int(seq_len - min_pos)
                required_keep = max(1, min(seq_len, required_keep))

                logits_to_keep = 0
                try:
                    forward_kwargs["logits_to_keep"] = required_keep
                    outputs = model(**forward_kwargs)
                    logits_to_keep = required_keep
                except TypeError:
                    forward_kwargs.pop("logits_to_keep", None)
                    try:
                        outputs = model(**forward_kwargs)
                    except TypeError:
                        forward_kwargs.pop("use_cache", None)
                        outputs = model(**forward_kwargs)
            except torch.cuda.OutOfMemoryError:
                print(f"[ERROR] CUDA OOM at batch {batch_idx}. Consider reducing batch_size.")
                torch.cuda.empty_cache()
                raise

            logits = outputs.logits
            batch_size = logits.shape[0]
            ar = torch.arange(batch_size, device=device)
            # Map per-sample true last positions into the kept logits window if applicable.
            if logits_to_keep > 0 and logits.shape[1] == logits_to_keep:
                start_pos = seq_len - logits_to_keep
                idx = pos - start_pos
                last_logits = logits[ar, idx, :]
            else:
                last_logits = logits[ar, pos, :]

            lp = torch.logsumexp(last_logits.index_select(1, p_ids), dim=1)
            ln = torch.logsumexp(last_logits.index_select(1, n_ids), dim=1)
            score = (lp - ln).cpu()

            batch_preds = ["P" if s.item() > threshold else "N" for s in score]
            preds.extend(batch_preds)
            scores.extend([s.item() for s in score])

    return preds, scores


def evaluate_checkpoints(
    base_model: str,
    checkpoints_dir: str,
    val_jsonl: str,
    max_seq_len: int = 180,
    batch_size: int = 32,
    cache_dir: str = None,
    device: str = None,
    output_csv: str = None,
    threshold: float = 0.0,
    ckpt_regex: str = None,
    include_baseline: bool = True,
    resume: bool = False,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")
    warmup_cublas(device)

    print(f"[Init] Loading base model: {base_model}")
    # Keep the base model in 4-bit even when evaluating random-init FP16 checkpoints.
    try:
        base_model_obj, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=True,
            full_finetuning=False,
            attn_implementation="eager",
        )
    except TypeError:
        base_model_obj, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_len,
            dtype=None,
            load_in_4bit=True,
            attn_implementation="eager",
        )

    if hasattr(base_model_obj, "config") and getattr(base_model_obj.config, "use_cache", None) is not None:
        base_model_obj.config.use_cache = False

    # Match common eval pipelines in this project: switch to inference-optimized mode when available.
    try:
        FastLanguageModel.for_inference(base_model_obj)
    except Exception:
        pass

    if hasattr(base_model_obj, "config") and getattr(base_model_obj.config, "use_cache", None) is not None:
        base_model_obj.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    p_ids, n_ids = verify_token_ids(tokenizer)
    p_ids_t = torch.tensor(p_ids, dtype=torch.long, device=device)
    n_ids_t = torch.tensor(n_ids, dtype=torch.long, device=device)

    data = read_jsonl_msgs(val_jsonl)

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Tokenization depends on tokenizer/chat_template; namespace caches by base model.
        model_tag = hashlib.sha256(base_model.encode("utf-8")).hexdigest()[:12]
        cache_name = f"cache_pn_{model_tag}_{os.path.basename(checkpoints_dir)}_seq{max_seq_len}.pt"
        cache_path = os.path.join(cache_dir, cache_name)

    encoded_inputs, golds = tokenize_dataset(tokenizer, data, max_seq_len, cache_path)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate_fn(batch):
        return collate_pad(batch, pad_id)

    dataset = list(zip(encoded_inputs, golds))

    def make_loader(bs: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    effective_batch_size = batch_size
    loader = make_loader(effective_batch_size)

    out_csv = output_csv or os.path.join(os.path.dirname(checkpoints_dir), "validation_metrics_pn.csv")
    results: List[Dict] = []
    existing_fieldnames: List[str] = []
    existing_paths = set()
    if resume and os.path.exists(out_csv):
        print(f"[Resume] Loading existing results: {out_csv}")
        results, existing_fieldnames, existing_paths = load_existing_results_csv(out_csv)
        print(f"[Resume] Loaded {len(results)} rows; {len(existing_paths)} unique paths")
    loaded_paths = set(existing_paths)

    attempted_ckpts: List[Tuple[str, str]] = []
    succeeded_ckpts: List[Tuple[str, str]] = []
    failed_ckpts: List[Tuple[str, str, str]] = []
    skipped_non_adapter: List[str] = []
    skipped_already_evaluated: List[str] = []

    # --------------------------------------
    # Baseline: base model without adapters
    # --------------------------------------
    if include_baseline:
        if resume and any(r.get("checkpoint") == "base_model" for r in results):
            print("[Resume] baseline already present -> skipping base_model evaluation")
        else:
            print("\n[BASELINE] Evaluating base model (no fine-tuning, P/N)...")
            while True:
                try:
                    preds_base, scores_base = evaluate_single_checkpoint(
                        base_model_obj,
                        loader,
                        p_ids_t,
                        n_ids_t,
                        device,
                        "base_model",
                        threshold,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    if effective_batch_size <= 1:
                        raise
                    effective_batch_size = max(1, effective_batch_size // 2)
                    print(f"[OOM] Reducing batch_size to {effective_batch_size} and retrying base_model...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    loader = make_loader(effective_batch_size)
                except RuntimeError as e:
                    if not is_cublas_init_error(e):
                        raise
                    if effective_batch_size <= 1:
                        print(f"[FATAL] cuBLAS init error at batch_size=1: {e} ({cuda_mem_summary()})")
                        raise
                    effective_batch_size = max(1, effective_batch_size // 2)
                    print(
                        f"[CUDA] {e} -> reducing batch_size to {effective_batch_size} and retrying base_model... ({cuda_mem_summary()})"
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    loader = make_loader(effective_batch_size)
            metrics_base = confusion_and_metrics(golds, preds_base, scores_base)
            base_result = {
                "checkpoint": "base_model",
                "path": "base_model (no adapter)",
            }
            base_result.update(metrics_base)
            results.append(base_result)
            if resume:
                write_results_csv(out_csv, results, existing_fieldnames)

            print(f"\n[Results] base_model:")
            print(f"  Accuracy: {metrics_base['accuracy']:.4f}")
            print(f"  Balanced Acc: {metrics_base['balanced_accuracy']:.4f}")
            print(f"  MCC: {metrics_base['MCC']:.4f}")
            print(f"  TPR (P recall): {metrics_base['TPR_recall_P']:.4f}")
            print(f"  TNR (N recall): {metrics_base['TNR_specificity']:.4f}")
            print(f"  F1_P: {metrics_base['F1_P']:.4f}")
            print(f"  F1_N: {metrics_base['F1_N']:.4f}")
            if metrics_base["AUROC"] is not None:
                print(f"  AUROC: {metrics_base['AUROC']:.4f}")

    if not os.path.isdir(checkpoints_dir):
        raise NotADirectoryError(f"Checkpoints directory not found: {checkpoints_dir}")

    entries = sorted(os.listdir(checkpoints_dir))
    if ckpt_regex:
        pattern = re.compile(ckpt_regex)
        entries = [e for e in entries if pattern.search(e)]

    if not entries:
        # Baseline-only mode:
        # Some callers want *just* the base model metrics (no adapters). Historically this script
        # always required at least 1 matching checkpoint entry, which made baseline-only eval
        # impossible. If we already computed baseline metrics, save them and exit cleanly.
        if include_baseline and results:
            print(f"[WARN] No checkpoints found in {checkpoints_dir} (ckpt_regex={ckpt_regex!r}) -> baseline-only results.")
            write_results_csv(out_csv, results, existing_fieldnames)
            print(f"\n[Done] Saved metrics to: {out_csv}")
            if resume:
                newly_added_rows = [r for r in results if r.get("path") and r.get("path") not in loaded_paths]
                print(
                    f"[Summary] resume=1 loaded_rows={len(results) - len(newly_added_rows)} newly_added_rows={len(newly_added_rows)} "
                    f"attempted_ckpts=0 succeeded_ckpts=0 failed_ckpts=0 skipped_already=0 skipped_non_adapter=0"
                )
            else:
                print(
                    "[Summary] resume=0 attempted_ckpts=0 succeeded_ckpts=0 failed_ckpts=0 skipped_non_adapter=0"
                )
            return

        raise RuntimeError(f"No checkpoints found in {checkpoints_dir}")

    # Avoid repeated `PeftModel.from_pretrained(base_model_obj, ...)` on the same base model.
    # PEFT mutates the model in-place (adapter injection), which can stack adapters and leak VRAM.
    peft_model = None
    active_adapter_name = None

    for entry in entries:
        ckpt_path = os.path.join(checkpoints_dir, entry)
        ckpt_name = entry

        if not os.path.isdir(ckpt_path):
            continue
        adapter_dir = resolve_peft_adapter_dir(ckpt_path)
        if adapter_dir is None:
            print(f"[Skip] {ckpt_path} is not a PEFT adapter dir")
            skipped_non_adapter.append(ckpt_path)
            continue
        if resume and adapter_dir in existing_paths:
            print(f"[Resume] Already evaluated -> skipping: {adapter_dir}")
            skipped_already_evaluated.append(adapter_dir)
            continue

        try:
            attempted_ckpts.append((ckpt_name, adapter_dir))
            if peft_model is None:
                print(f"\n[Load] Loading adapter from {adapter_dir}")
                peft_model = PeftModel.from_pretrained(
                    base_model_obj,
                    adapter_dir,
                    adapter_name=ckpt_name,
                    is_trainable=False,
                ).to(device)
                active_adapter_name = ckpt_name
            else:
                # Keep VRAM low: remove the previous adapter before loading the next one.
                if active_adapter_name is not None:
                    try:
                        peft_model.delete_adapter(active_adapter_name)
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as e:
                        print(f"[WARN] Failed to delete previous adapter '{active_adapter_name}': {e}")

                print(f"\n[Load] Loading adapter from {adapter_dir}")
                peft_model.load_adapter(
                    adapter_dir,
                    adapter_name=ckpt_name,
                    is_trainable=False,
                )
                peft_model.set_adapter(ckpt_name)
                active_adapter_name = ckpt_name

            peft_model.eval()

            while True:
                try:
                    preds, scores = evaluate_single_checkpoint(
                        peft_model,
                        loader,
                        p_ids_t,
                        n_ids_t,
                        device,
                        ckpt_name,
                        threshold,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    if effective_batch_size <= 1:
                        raise
                    effective_batch_size = max(1, effective_batch_size // 2)
                    print(f"[OOM] Reducing batch_size to {effective_batch_size} and retrying {ckpt_name}...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    loader = make_loader(effective_batch_size)
                except RuntimeError as e:
                    if not is_cublas_init_error(e):
                        raise
                    if effective_batch_size <= 1:
                        print(f"[FATAL] cuBLAS init error at batch_size=1 for {ckpt_name}: {e} ({cuda_mem_summary()})")
                        raise
                    effective_batch_size = max(1, effective_batch_size // 2)
                    print(
                        f"[CUDA] {e} -> reducing batch_size to {effective_batch_size} and retrying {ckpt_name}... ({cuda_mem_summary()})"
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    loader = make_loader(effective_batch_size)

            metrics = confusion_and_metrics(golds, preds, scores)
            result = {"checkpoint": ckpt_name, "path": adapter_dir}
            result.update(metrics)
            results.append(result)
            succeeded_ckpts.append((ckpt_name, adapter_dir))
            if resume:
                existing_paths.add(adapter_dir)
                write_results_csv(out_csv, results, existing_fieldnames)

            print(f"\n[Results] {ckpt_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
            print(f"  MCC: {metrics['MCC']:.4f}")
            print(f"  TPR (P recall): {metrics['TPR_recall_P']:.4f}")
            print(f"  TNR (N recall): {metrics['TNR_specificity']:.4f}")
            print(f"  F1_P: {metrics['F1_P']:.4f}")
            print(f"  F1_N: {metrics['F1_N']:.4f}")
            if metrics["AUROC"] is not None:
                print(f"  AUROC: {metrics['AUROC']:.4f}")

        except Exception as e:
            print(f"[ERROR] Evaluation failed for {ckpt_name}: {e}")
            import traceback

            traceback.print_exc()
            failed_ckpts.append((ckpt_name, adapter_dir, str(e)))
            continue

    if not results:
        print("[ERROR] No checkpoints were successfully evaluated!")
        return

    write_results_csv(out_csv, results, existing_fieldnames)

    print(f"\n[Done] Saved metrics to: {out_csv}")

    if resume:
        newly_added_rows = [r for r in results if r.get("path") and r.get("path") not in loaded_paths]
        print(
            f"[Summary] resume=1 loaded_rows={len(results) - len(newly_added_rows)} newly_added_rows={len(newly_added_rows)} "
            f"attempted_ckpts={len(attempted_ckpts)} succeeded_ckpts={len(succeeded_ckpts)} failed_ckpts={len(failed_ckpts)} "
            f"skipped_already={len(skipped_already_evaluated)} skipped_non_adapter={len(skipped_non_adapter)}"
        )
    else:
        print(
            f"[Summary] resume=0 attempted_ckpts={len(attempted_ckpts)} succeeded_ckpts={len(succeeded_ckpts)} "
            f"failed_ckpts={len(failed_ckpts)} skipped_non_adapter={len(skipped_non_adapter)}"
        )

    if failed_ckpts:
        print("[Summary] Failed checkpoints:")
        for ckpt_name, ckpt_path, err in failed_ckpts[:20]:
            print(f"  - {ckpt_name}: {ckpt_path} :: {err}")
        if len(failed_ckpts) > 20:
            print(f"  ... and {len(failed_ckpts) - 20} more")

    print("\n" + "=" * 70)
    print("TOP CHECKPOINTS (by Balanced Accuracy → MCC)")
    print("=" * 70)

    sorted_results = sorted(results, key=lambda r: (r["balanced_accuracy"], r["MCC"]), reverse=True)

    for i, r in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. {r['checkpoint']}")
        print(
            f"   Balanced Acc: {r['balanced_accuracy']:.4f} | MCC: {r['MCC']:.4f} | Accuracy: {r['accuracy']:.4f}"
        )
        print(
            f"   TPR (P recall): {r['TPR_recall_P']:.4f} | TNR (N recall): {r['TNR_specificity']:.4f}"
        )
        print(
            f"   Precision (P): {r['precision_P']:.4f} | F1_P: {r['F1_P']:.4f} | F1_N: {r['F1_N']:.4f}"
        )
        if r["AUROC"] is not None:
            print(f"   AUROC: {r['AUROC']:.4f}")

    best = sorted_results[0]
    print("\n" + "=" * 70)
    print(f"BEST CHECKPOINT: {best['checkpoint']}")
    print(f"Path: {best['path']}")
    print("=" * 70)

    # Optional summary: baseline vs best adapter
    baseline = next((r for r in results if r["checkpoint"] == "base_model"), None)
    adapters_only = [r for r in results if r["checkpoint"] != "base_model"]

    if baseline and adapters_only:
        best_adapter = sorted(adapters_only, key=lambda r: (r["balanced_accuracy"], r["MCC"]), reverse=True)[0]
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY (Baseline vs Best Adapter)")
        print("=" * 70)
        print("\n[BASELINE] base_model:")
        print(f"  Balanced Acc: {baseline['balanced_accuracy']:.4f}")
        print(f"  MCC: {baseline['MCC']:.4f}")
        print(f"  TPR: {baseline['TPR_recall_P']:.4f} | TNR: {baseline['TNR_specificity']:.4f}")
        print("\n[BEST ADAPTER] " + best_adapter["checkpoint"] + ":")
        print(f"  Balanced Acc: {best_adapter['balanced_accuracy']:.4f}")
        print(f"  MCC: {best_adapter['MCC']:.4f}")
        print(f"  TPR: {best_adapter['TPR_recall_P']:.4f} | TNR: {best_adapter['TNR_specificity']:.4f}")
        bal_acc_improve = (best_adapter["balanced_accuracy"] - baseline["balanced_accuracy"]) * 100
        mcc_improve = best_adapter["MCC"] - baseline["MCC"]
        print("\n[IMPROVEMENT]")
        print(f"  Balanced Acc: +{bal_acc_improve:.2f}%")
        print(f"  MCC: +{mcc_improve:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate P/N classification checkpoints")
    parser.add_argument("--base_model", type=str, default="unsloth/gpt-oss-20b-unsloth-bnb-4bit")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/DATA/gpt-oss/20b/gpt-oss-20b-focal-pn-v1/checkpoints",
        help="Directory containing checkpoints (epoch_1,...,epoch_5)",
    )
    parser.add_argument(
        "--val_jsonl",
        type=str,
        default="/DATA/npj_compt.mat_project_github/data/valid_llm_pn.jsonl",
        help="Validation JSONL file",
    )
    parser.add_argument("--max_seq_len", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache_dir", type=str, default="/DATA/gpt-oss/cache_eval_pn")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/DATA/gpt-oss/20b/evaluation_pn/validation_metrics_pn.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Decision threshold for P/N classification (log-diff)",
    )
    parser.add_argument(
        "--ckpt_regex",
        type=str,
        default=None,
        help="Regex filter for checkpoint names (optional)",
    )
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Skip base model evaluation (only adapters)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output_csv exists, skip already-evaluated checkpoints and append new rows (safe to re-run).",
    )

    args = parser.parse_args()

    evaluate_checkpoints(
        base_model=args.base_model,
        checkpoints_dir=args.checkpoints_dir,
        val_jsonl=args.val_jsonl,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        device=args.device,
        output_csv=args.output_csv,
        threshold=args.threshold,
        ckpt_regex=args.ckpt_regex,
        include_baseline=not args.no_baseline,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
