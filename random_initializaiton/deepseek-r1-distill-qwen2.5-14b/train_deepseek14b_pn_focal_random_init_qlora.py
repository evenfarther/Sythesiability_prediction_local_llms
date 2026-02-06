#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P/N-only focal training for DeepSeek-R1-Distill-Qwen-14B (QLoRA 4bit)
using a **random-initialized** base checkpoint directory.

This keeps the training recipe as close as possible to the pretrained control:
- same dataset formatting (messages -> chat template text)
- same PN-only focal loss (last classification token)
- same LoRA config and optimizer defaults

Difference:
- base model is loaded from a local random-init checkpoint directory (FP16/BF16),
  then quantized to 4-bit (`load_in_4bit=True`) for QLoRA training.
"""

import os
import gc
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Environment defaults
# --------------------------------------------------------------------------- #
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128",
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Flex attention / Unsloth knobs
os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
os.environ["TORCH_COMPILE"] = "0"

# Patch Unsloth's fix_untrained_tokens to be safe when tokenizer is None
try:
    import unsloth_zoo.tokenizer_utils as _utok  # type: ignore

    if not hasattr(_utok, "_orig_fix_untrained_tokens"):
        _utok._orig_fix_untrained_tokens = _utok.fix_untrained_tokens  # type: ignore[attr-defined]

        def _safe_fix_untrained_tokens(model, tokenizer, train_dataset, ignored_names, eps=1e-16):
            if tokenizer is None:
                print("Unsloth: tokenizer is None in fix_untrained_tokens; skipping this step.")
                return
            return _utok._orig_fix_untrained_tokens(model, tokenizer, train_dataset, ignored_names, eps=eps)

        _utok.fix_untrained_tokens = _safe_fix_untrained_tokens  # type: ignore[attr-defined]
        print("Patched unsloth_zoo.tokenizer_utils.fix_untrained_tokens with None-safe wrapper.")
except Exception as e:  # pragma: no cover
    print(f"Warning: could not patch fix_untrained_tokens: {e}")

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
# Prefer Unsloth if available (matching previous successful DeepSeek scripts)
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel  # type: ignore

    UNSLOTH_AVAILABLE = True
except Exception:
    pass

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
from trl import SFTTrainer

# --------------------------------------------------------------------------- #
# Config (env overridable)
# --------------------------------------------------------------------------- #
DEFAULT_HF = "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit"

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

TRAIN_DATA_PATH = os.environ.get("PN_DATA_PATH", "/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl")
OUTPUT_SUFFIX = os.environ.get("PN_OUTPUT_SUFFIX", "pn-focal-v1-randinit")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"./deepseek_14b_{OUTPUT_SUFFIX}")
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip() or None

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "180"))
BATCH_SIZE = int(os.environ.get("PER_DEVICE_BATCH", "64"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))
NUM_EPOCHS = int(os.environ.get("EPOCHS", "10"))
LR = float(os.environ.get("LR", "2e-4"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
TRAIN_SEED = int(os.environ.get("SEED", "3407"))
OPTIM_NAME = os.environ.get("OPTIM", "adamw_torch_fused")

USE_BF16 = os.environ.get("BF16", "1") == "1" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = not USE_BF16

LORA_R = int(os.environ.get("LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "128"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))

FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))
FOCAL_ALPHA_P = float(os.environ.get("FOCAL_ALPHA_P", "7.5"))  # P minority (~12%)
FOCAL_ALPHA_N = float(os.environ.get("FOCAL_ALPHA_N", "1.0"))

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_texts(tokenizer, dataset):
    def fmt(batch):
        texts = []
        for msgs in batch["messages"]:
            if not msgs or msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": "You are a materials science assistant. Answer only 'P' or 'N'."}] + msgs
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    return dataset.map(
        fmt,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        num_proc=1,
        desc="Formatting messages",
    )


# --------------------------------------------------------------------------- #
# Loss: P/N-only focal (last P/N token per sample)
# --------------------------------------------------------------------------- #
class PNOnlyFocalLoss(nn.Module):
    def __init__(self, tokenizer, gamma=2.0, alpha_p=7.5, alpha_n=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha_p = alpha_p
        self.alpha_n = alpha_n

        def last_token_id(s: str):
            ids = tokenizer.encode(s, add_special_tokens=False)
            return [ids[-1]] if ids else []

        self.p_token_ids = sorted(set(last_token_id("P") + last_token_id(" P") + last_token_id("\nP")))
        self.n_token_ids = sorted(set(last_token_id("N") + last_token_id(" N") + last_token_id("\nN")))
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, batch_size: Optional[int] = None):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma
        weights = torch.zeros_like(ce)

        pn_mask = torch.zeros_like(targets, dtype=torch.bool)
        for t in self.p_token_ids:
            pn_mask |= (targets == t)
        for t in self.n_token_ids:
            pn_mask |= (targets == t)

        if batch_size:
            seq_len = len(targets) // batch_size
            pn_mask_2d = pn_mask.view(batch_size, seq_len)
            has_pn = pn_mask_2d.any(dim=1)
            if has_pn.any():
                rev = torch.flip(pn_mask_2d, dims=[1]).float()
                last_pos_from_end = rev.argmax(dim=1)
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

        focal_loss = weights * focal * ce
        num = (weights > 0).sum()
        if num > 0:
            return focal_loss.sum() / num
        mask = targets != -100
        if mask.sum() > 0:
            return focal_loss.sum() / mask.sum()
        return focal_loss.mean()


# --------------------------------------------------------------------------- #
# Trainer subclass to apply focal loss
# --------------------------------------------------------------------------- #
class PNFocalSFTTrainer(SFTTrainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self.focal_loss = PNOnlyFocalLoss(
            tokenizer=tokenizer, gamma=FOCAL_GAMMA, alpha_p=FOCAL_ALPHA_P, alpha_n=FOCAL_ALPHA_N
        )
        self.answers_seen = 0
        self.samples_seen = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Avoid fp32 materialization of full [B, T, V] logits under Accelerate AMP.
        # These runs only need the final P/N classification token per sample, so we compute logits
        # for the last K positions only (via `logits_to_keep`) and apply focal loss on the selected token.
        labels = inputs.pop("labels")
        batch_size_local, seq_len = labels.shape

        pn_ids = self.focal_loss.p_token_ids + self.focal_loss.n_token_ids
        pn_mask = torch.zeros_like(labels, dtype=torch.bool)
        for t in pn_ids:
            pn_mask |= labels == t
        has_pn = pn_mask.any(dim=1)

        final_pn_count = int(has_pn.sum().item())
        self.answers_seen += final_pn_count
        self.samples_seen += batch_size_local

        if not has_pn.any():
            # Skip batch safely (no P/N labels detected) without OOMing.
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
        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------------------------- #
# Model loader (Unsloth if available; fallback to HF+PEFT)
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer():
    if UNSLOTH_AVAILABLE:
        # Load model via Unsloth, but use HF tokenizer explicitly to avoid None issues
        model, _ = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME_OR_PATH,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=TRAIN_SEED,
            use_rslora=False,
            loftq_config=None,
        )
        return model, tokenizer

    # HF + PEFT path
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 80)
    print("DeepSeek 14B P/N Focal Training (random-init base; QLoRA-only)")
    print("=" * 80)
    print(f"Model path:       {MODEL_NAME_OR_PATH}")
    print(f"Train data:       {TRAIN_DATA_PATH}")
    print(f"Output dir:       {OUTPUT_DIR}")
    print(f"Checkpoint dir:   {CHECKPOINT_DIR}")
    print(f"Train seed:       {TRAIN_SEED}")
    print(f"Max seq len:      {MAX_SEQ_LEN}")
    print(f"Batch x GradAcc:  {BATCH_SIZE} x {GRAD_ACCUM} (effective {BATCH_SIZE * GRAD_ACCUM})")
    print(f"Epochs:           {NUM_EPOCHS}")
    print(f"LR:               {LR}")
    print(f"Optim:            {OPTIM_NAME}")
    print(f"BF16/FP16:        {USE_BF16}/{USE_FP16}")
    print(f"LoRA r/alpha/do:  {LORA_R}/{LORA_ALPHA}/{LORA_DROPOUT}")
    print(f"Focal gamma:      {FOCAL_GAMMA}")
    print(f"Focal alpha P/N:  {FOCAL_ALPHA_P}/{FOCAL_ALPHA_N}")
    if RESUME_FROM_CHECKPOINT:
        print(f"Resume from:      {RESUME_FROM_CHECKPOINT}")

    # Set python/numpy/torch RNG seeds for reproducibility.
    set_seed(TRAIN_SEED)

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    cleanup_memory()

    model, tokenizer = load_model_and_tokenizer()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: {type(tokenizer)}; pad_token: {tokenizer.pad_token}")
    cleanup_memory()

    train_ds = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train", keep_in_memory=False)
    train_ds = prepare_texts(tokenizer, train_ds)

    steps_per_epoch = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(10, int(total_steps * 0.06))

    print(f"Steps/epoch: {steps_per_epoch}, total steps: {total_steps}, warmup: {warmup_steps}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        warmup_steps=warmup_steps,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        fp16=USE_FP16,
        bf16=USE_BF16,
        optim=OPTIM_NAME,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=TRAIN_SEED,
        dataloader_num_workers=0,
        max_grad_norm=0.5,
    )

    trainer = PNFocalSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
    )

    if RESUME_FROM_CHECKPOINT:
        trainer_stats = trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    else:
        trainer_stats = trainer.train()
    final_model_path = Path(OUTPUT_DIR) / "final_model"
    trainer.save_model(final_model_path)

    summary = {
        "train_data": TRAIN_DATA_PATH,
        "output_dir": OUTPUT_DIR,
        "train_seed": TRAIN_SEED,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "epochs": NUM_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "lr": LR,
        "optim": OPTIM_NAME,
        "bf16": USE_BF16,
        "fp16": USE_FP16,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "focal": {"gamma": FOCAL_GAMMA, "alpha_p": FOCAL_ALPHA_P, "alpha_n": FOCAL_ALPHA_N},
        "metrics": getattr(trainer_stats, "metrics", {}),
        "answers_seen": getattr(trainer, "answers_seen", None),
        "samples_seen": getattr(trainer, "samples_seen", None),
        "base_model_path": MODEL_NAME_OR_PATH,
        "require_random_base": REQUIRE_RANDOM_BASE,
    }
    with open(Path(OUTPUT_DIR) / "training_summary.json", "w") as f:
        import json

        json.dump(summary, f, indent=2)

    print("Training complete. Summary saved to training_summary.json")


if __name__ == "__main__":
    main()
