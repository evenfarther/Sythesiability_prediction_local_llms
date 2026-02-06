#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_qwen3_14b_pn_focal_random_init_qlora.py
--------------------------------------------
Stable SFT trainer for Qwen3-14B random-init control using P/N-only focal loss (QLoRA 4bit).

Difference vs pretrained control:
- base model is loaded from a local random-init checkpoint directory (FP16/BF16),
  then quantized to 4-bit for QLoRA training.
"""

import os
import sys
import gc
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# ======= Environment knobs (safe, ASCII-only) =======
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128",
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ======= Optional Unsloth imports =======
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel  # type: ignore

    UNSLOTH_AVAILABLE = True
except Exception:
    pass

# ======= HF ecosystem =======
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    __version__ as TRANSFORMERS_VERSION,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
from trl import SFTTrainer, SFTConfig  # type: ignore

# ======= Defaults (ENV-overridable) =======
DEFAULT_HF = "unsloth/Qwen3-14B-unsloth-bnb-4bit"

MODEL_NAME_OR_PATH = os.environ.get("MODEL_NAME_OR_PATH", "").strip() or DEFAULT_HF
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "").strip()
if BASE_MODEL_PATH:
    MODEL_NAME_OR_PATH = BASE_MODEL_PATH

REQUIRE_RANDOM_BASE = os.environ.get("REQUIRE_RANDOM_BASE", "0").strip() == "1"
if REQUIRE_RANDOM_BASE:
    base_path = Path(MODEL_NAME_OR_PATH)
    if not base_path.exists() or not base_path.is_dir():
        raise RuntimeError(
            "REQUIRE_RANDOM_BASE=1 but MODEL_NAME_OR_PATH does not point to a local directory: "
            f"{MODEL_NAME_OR_PATH}"
        )
    if not (base_path / "config.json").exists():
        raise RuntimeError(f"Random-init base dir missing config.json: {MODEL_NAME_OR_PATH}")

TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH", "/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./qwen3_14b_pn_focal_v1_randinit")
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "180"))
BATCH_SIZE = int(os.environ.get("PER_DEVICE_BATCH", "64"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "20"))
NUM_EPOCHS = int(os.environ.get("EPOCHS", "10"))
LR = float(os.environ.get("LR", "2e-4"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
SAVE_STRATEGY = os.environ.get("SAVE_STRATEGY", "epoch")
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "0"))
SEED = int(os.environ.get("SEED", "3407"))

SUPPORTS_BF16 = (
    (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) if hasattr(torch.cuda, "is_bf16_supported") else False
)
USE_BF16 = os.environ.get("BF16", "1") == "1" and SUPPORTS_BF16
USE_FP16 = not USE_BF16
USE_GC = os.environ.get("USE_GC", "0") == "1"

OPTIM_NAME = os.environ.get("OPTIM", "adamw_torch_fused")

LORA_R = int(os.environ.get("LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "128"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))

FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))
FOCAL_ALPHA_P = float(os.environ.get("FOCAL_ALPHA_P", "7.5"))  # P minority (~12%)
FOCAL_ALPHA_N = float(os.environ.get("FOCAL_ALPHA_N", "1.0"))


# ======= Utility =======
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_set_use_cache_false(model):
    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception:
        pass


def print_versions():
    print("=" * 80)
    print("ENV / VERSION CHECK")
    print("=" * 80)
    print(f"Python:         {sys.version.split()[0]}")
    try:
        import trl

        print(f"TRL:            {trl.__version__}")
    except Exception as e:
        print(f"TRL:            unknown ({e})")
    try:
        import peft

        print(f"PEFT:           {peft.__version__}")
    except Exception as e:
        print(f"PEFT:           unknown ({e})")
    try:
        import bitsandbytes as bnb  # noqa

        print("bitsandbytes:   available")
    except Exception as e:
        print(f"bitsandbytes:   unavailable ({e})")
    print(f"Transformers:   {TRANSFORMERS_VERSION}")
    print(f"Torch:          {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device(s): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  - {i}: {props.name}, {props.total_memory/1024**3:.1f} GiB")


# ======= P/N-only Focal Loss =======
class PNOnlyFocalLoss(nn.Module):
    def __init__(self, tokenizer, gamma=2.0, alpha_p=7.5, alpha_n=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha_p = alpha_p
        self.alpha_n = alpha_n
        self.tokenizer = tokenizer

        def last_token_id(s: str):
            ids = tokenizer.encode(s, add_special_tokens=False)
            return [ids[-1]] if ids else []

        self.p_token_ids = sorted(set(last_token_id("P") + last_token_id(" P") + last_token_id("\nP")))
        self.n_token_ids = sorted(set(last_token_id("N") + last_token_id(" N") + last_token_id("\nN")))

        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, batch_size: Optional[int] = None):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        focal_factor = (1 - pt) ** self.gamma

        weights = torch.zeros_like(ce)
        pn_mask = torch.zeros_like(targets, dtype=torch.bool)

        if self.p_token_ids or self.n_token_ids:
            if self.p_token_ids:
                for p_id in self.p_token_ids:
                    pn_mask |= (targets == p_id)
            if self.n_token_ids:
                for n_id in self.n_token_ids:
                    pn_mask |= (targets == n_id)

            if batch_size is not None and batch_size > 0:
                seq_len = len(targets) // batch_size
                pn_mask_2d = pn_mask.view(batch_size, seq_len)
                has_pn = pn_mask_2d.any(dim=1)
                if has_pn.any():
                    rev_mask = torch.flip(pn_mask_2d, dims=[1]).float()
                    last_pos_from_end = rev_mask.argmax(dim=1)
                    last_pos = seq_len - 1 - last_pos_from_end
                    select_2d = torch.zeros_like(pn_mask_2d)
                    select_2d[torch.arange(batch_size, device=targets.device)[has_pn], last_pos[has_pn]] = True
                    select = select_2d.view(-1)
                    for p_id in self.p_token_ids:
                        weights[select & (targets == p_id)] = self.alpha_p
                    for n_id in self.n_token_ids:
                        weights[select & (targets == n_id)] = self.alpha_n
            else:
                for p_id in self.p_token_ids:
                    weights[targets == p_id] = self.alpha_p
                for n_id in self.n_token_ids:
                    weights[targets == n_id] = self.alpha_n

        focal_loss = weights * focal_factor * ce
        num_pn = (weights > 0).sum()
        if num_pn > 0:
            return focal_loss.sum() / num_pn
        mask = targets != -100
        return focal_loss[mask].mean() if mask.any() else focal_loss.mean()


# ======= Trainer subclass =======
class PNFocalSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tok = getattr(self, "tokenizer", None)
        self.focal_loss = PNOnlyFocalLoss(
            tokenizer=tok,
            gamma=FOCAL_GAMMA,
            alpha_p=FOCAL_ALPHA_P,
            alpha_n=FOCAL_ALPHA_N,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Avoid fp32 materialization of full [B, T, V] logits under Accelerate AMP.
        # We only need the final P/N classification token per sample, so we compute logits
        # for the last K positions only (via `logits_to_keep`) and apply focal loss on that token.
        labels = inputs.pop("labels")
        batch_size_local, seq_len = labels.shape

        pn_ids = self.focal_loss.p_token_ids + self.focal_loss.n_token_ids
        pn_mask = torch.zeros_like(labels, dtype=torch.bool)
        for t in pn_ids:
            pn_mask |= labels == t
        has_pn = pn_mask.any(dim=1)

        if not has_pn.any():
            try:
                outputs = model(**inputs, logits_to_keep=2)
            except TypeError:
                outputs = model(**inputs)
            loss = outputs.logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        last_from_end = torch.flip(pn_mask, dims=[1]).float().argmax(dim=1)
        last_pos = (seq_len - 1) - last_from_end
        logit_pos = last_pos - 1
        valid = has_pn & (logit_pos >= 0)

        if not valid.any():
            try:
                outputs = model(**inputs, logits_to_keep=2)
            except TypeError:
                outputs = model(**inputs)
            loss = outputs.logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        required_keep = (seq_len - logit_pos[valid]).max().item()
        logits_to_keep = int(max(2, min(seq_len, required_keep)))

        try:
            outputs = model(**inputs, logits_to_keep=logits_to_keep)
        except TypeError:
            outputs = model(**inputs)
            logits_to_keep = 0

        logits = outputs.logits
        batch_idx = torch.arange(batch_size_local, device=labels.device)

        if logits_to_keep > 0:
            start_pos = seq_len - logits_to_keep
            idx_in_kept = (logit_pos - start_pos).clamp(min=0, max=logits_to_keep - 1)
            sel_logits = logits[batch_idx[valid], idx_in_kept[valid], :]  # [Bv, V]
        else:
            sel_logits = logits[batch_idx[valid], logit_pos[valid], :]

        sel_labels = labels[batch_idx[valid], last_pos[valid]]  # [Bv]

        ce = torch.nn.functional.cross_entropy(sel_logits.float(), sel_labels, reduction="none")
        pt = torch.exp(-ce)
        focal_term = (1.0 - pt) ** self.focal_loss.gamma

        is_p = torch.zeros_like(sel_labels, dtype=torch.bool)
        for t in self.focal_loss.p_token_ids:
            is_p |= sel_labels == t
        weights = torch.where(
            is_p,
            torch.full_like(ce, self.focal_loss.alpha_p),
            torch.full_like(ce, self.focal_loss.alpha_n),
        )
        loss = (weights * focal_term * ce).mean()

        try:
            if getattr(self, "state", None) is not None and getattr(self.state, "global_step", 0) < 3:
                print(f"[PN Debug] step={self.state.global_step} loss.requires_grad={loss.requires_grad}")
        except Exception:
            pass
        return (loss, outputs) if return_outputs else loss


# ======= Loaders =======
def load_model_with_unsloth(model_path: str, max_seq_length: int):
    print("Loading model with Unsloth...")
    dtype_str = "bfloat16" if USE_BF16 else "float16"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype_str,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=("unsloth" if USE_GC else False),
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    safe_set_use_cache_false(model)
    return model, tokenizer


def load_model_with_transformers(model_path: str, max_seq_length: int):
    print("Loading model with transformers + PEFT...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=USE_GC)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    try:
        if USE_GC:
            model.gradient_checkpointing_enable()
    except Exception:
        pass
    safe_set_use_cache_false(model)
    return model, tokenizer


# ======= Data =======
def build_text_dataset(train_path: str, tokenizer) -> Any:
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    ds = load_dataset("json", data_files=train_path, split="train", keep_in_memory=False)

    eos = tokenizer.eos_token or ""

    def format_messages(batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = []
        msgs_list = batch.get("messages", None)
        if msgs_list is None:
            for t in batch.get("text", []):
                t = t if t is not None else ""
                if not t.endswith(eos):
                    t = t + eos
                texts.append(t)
            return {"text": texts}

        for msgs in msgs_list:
            messages = msgs
            if not (messages and len(messages) > 0 and messages[0].get("role") == "system"):
                system_msg = {
                    "role": "system",
                    "content": "You are a helpful assistant for P/N classification of synthesizability.",
                }
                messages = [system_msg] + messages
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if not text.endswith(eos):
                text = text + eos
            texts.append(text)
        return {"text": texts}

    print("Converting messages to plain text with EOS...")
    ds = ds.map(
        format_messages,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        num_proc=1,
        desc="Formatting messages to text",
    )
    print("Dataset prepared.")
    return ds


# ======= SFTConfig wrapper =======
def make_sft_config(**kwargs) -> SFTConfig:
    import inspect

    sig = inspect.signature(SFTConfig)
    valid = set(sig.parameters.keys())
    if "evaluation_strategy" in kwargs and "eval_strategy" not in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    clean = {k: v for k, v in kwargs.items() if k in valid}
    return SFTConfig(**clean)


# ======= Main =======
def main():
    print("=" * 80)
    print("Qwen3-14B — P/N Focal SFT (random-init base; QLoRA-only)")
    print("=" * 80)
    print_versions()

    model_path = MODEL_NAME_OR_PATH
    output_dir = OUTPUT_DIR
    checkpoint_dir = CHECKPOINT_DIR
    max_seq_length = MAX_SEQ_LEN
    batch_size = BATCH_SIZE
    grad_accum = GRAD_ACCUM
    num_epochs = NUM_EPOCHS
    lr = LR
    logging_steps = LOGGING_STEPS
    save_steps = SAVE_STEPS
    train_data_path = TRAIN_DATA_PATH

    print("Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Train data: {train_data_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Max seq len: {max_seq_length}")
    print(f"  Batch size: {batch_size}  |  Grad accum: {grad_accum}  |  Effective: {batch_size * grad_accum}")
    print(f"  LoRA r/alpha/dropout: {LORA_R}/{LORA_ALPHA}/{LORA_DROPOUT}")
    print(f"  LR: {lr}, Epochs: {num_epochs}, Logging steps: {logging_steps}")
    print(f"  Precision: {'bf16' if USE_BF16 else 'fp16'}  |  Grad checkpointing: {USE_GC}")
    print(f"  Seed: {SEED}")
    set_seed(SEED)
    cleanup_memory()

    # Load model (Unsloth -> fallback)
    try:
        if UNSLOTH_AVAILABLE:
            print("[Init] Attempting to load with Unsloth...")
            model, tokenizer = load_model_with_unsloth(model_path, max_seq_length)
            print("Unsloth path OK")
        else:
            raise RuntimeError("Unsloth not available")
    except Exception as e:
        print(f"Unsloth loading failed: {e}")
        print("Falling back to transformers + PEFT")
        model, tokenizer = load_model_with_transformers(model_path, max_seq_length)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({(trainable_params/total_params*100):.2f}%)")

    ds = build_text_dataset(train_data_path, tokenizer)

    steps_per_epoch = math.ceil(len(ds) / max(1, batch_size * grad_accum))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(10, int(total_steps * 0.06))

    print("Training statistics:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Warmup steps:    {warmup_steps}")

    train_args = make_sft_config(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        fp16=USE_FP16,
        bf16=USE_BF16,
        logging_steps=logging_steps,
        save_strategy=SAVE_STRATEGY,
        save_steps=save_steps,
        save_total_limit=10,
        load_best_model_at_end=False,
        optim=OPTIM_NAME,
        seed=SEED,
        report_to="none",
        group_by_length=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        max_length=max_seq_length,
        packing=False,
        dataset_text_field="text",
        do_eval=False,
    )

    trainer = PNFocalSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=train_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        train_out = trainer.train()
        final_model_path = f"{output_dir}/final_model"
        print(f"Saving final adapter to {final_model_path}")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        summary = {
            "train_data": train_data_path,
            "output_dir": output_dir,
            "train_seed": SEED,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum,
            "epochs": num_epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "lr": lr,
            "bf16": USE_BF16,
            "fp16": USE_FP16,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
            "focal": {"gamma": FOCAL_GAMMA, "alpha_p": FOCAL_ALPHA_P, "alpha_n": FOCAL_ALPHA_N},
            "metrics": getattr(train_out, "metrics", {}),
            "base_model_path": model_path,
            "require_random_base": REQUIRE_RANDOM_BASE,
        }
        try:
            import json

            with open(Path(output_dir) / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not write training_summary.json: {e}")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"Final adapter: {final_model_path}")
        print(f"Checkpoints:   {checkpoint_dir}")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cleanup_memory()
        print(f"Script ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
