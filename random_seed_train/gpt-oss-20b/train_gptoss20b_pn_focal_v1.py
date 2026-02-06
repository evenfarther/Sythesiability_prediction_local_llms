#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final Training Script (P/N) with classification-token-only focal loss for GPT-OSS 20B
Creates a dedicated output directory to avoid overlapping with prior P/U runs.
"""

import os
import json
import torch
import torch.nn as nn
import gc
import time
import math
from pathlib import Path

# Environment setup before heavy imports
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from unsloth import FastLanguageModel
from peft import PeftModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback, set_seed
from datasets import load_dataset

# Hard-disable flex attention torch.compile path (conflicts with bitsandbytes + proxy tensor)
try:
    import unsloth_zoo.flex_attention.attention_sink as fas  # type: ignore
    import unsloth_zoo.temporary_patches.gpt_oss as gpt_oss  # type: ignore

    _orig_flex_attention_with_sink = fas.flex_attention_with_sink

    def flex_attention_no_compile(self_attn, query, key, value, scale=None, sliding_window=None, compile=False):
        # Ignore caller's compile flag entirely; always use uncompiled path
        return _orig_flex_attention_with_sink(
            self_attn,
            query,
            key,
            value,
            scale=scale,
            sliding_window=sliding_window,
            compile=False,
        )

    # Also force default compile flag to False on the original, in case other references call it
    try:
        defaults = list(_orig_flex_attention_with_sink.__defaults__ or ())
        if defaults and defaults[-1] is True:
            defaults[-1] = False
            _orig_flex_attention_with_sink.__defaults__ = tuple(defaults)
    except Exception:
        pass

    # Override in both modules so GptOssAttention forward uses the non-compiled path
    fas.flex_attention_with_sink = flex_attention_no_compile
    gpt_oss.flex_attention_with_sink = flex_attention_no_compile

    # Global kill-switch: ensure torch.nn.attention.flex_attention.flex_attention points to uncompiled version
    try:
        import torch.nn.attention.flex_attention as flex  # type: ignore
        flex.flex_attention = flex.uncompiled_flex_attention
        print("Rebound torch.nn.attention.flex_attention.flex_attention -> uncompiled_flex_attention")
    except Exception:
        pass

    # Reset env flag back to 0 after gpt_oss sets it to 1
    os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
    print("Patched flex_attention_with_sink to run without torch.compile (compile=False) [fas & gpt_oss]")
    try:
        print("flex_attention_with_sink defaults:", gpt_oss.flex_attention_with_sink.__defaults__)
    except Exception:
        pass
except Exception as e:  # pragma: no cover - best-effort patch
    print(f"Warning: flex attention patch failed: {e}")

try:
    import bitsandbytes as bnb  # noqa: F401
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

print("=" * 80)
print("FINAL GPT-OSS 20B TRAINING WITH P/N-ONLY FOCAL LOSS (v1)")
print("Materials synthesis classification (P vs N)")
print("Loss applied only to P/N classification tokens (last token per sample)")
print("=" * 80)

# ---------------------------
# Configuration
# ---------------------------
max_seq_length = 180
dtype = None
batch_size = int(os.environ.get("PER_DEVICE_BATCH", "64"))
gradient_accumulation_steps = int(os.environ.get("GRAD_ACCUM", "20"))
lora_r = 64
lora_alpha = 128
learning_rate = float(os.environ.get("LR", "2e-4"))
train_seed = int(os.environ.get("SEED", "3407"))
num_epochs = int(os.environ.get("EPOCHS", "10"))
logging_steps = 10
epoch_offset = int(os.environ.get("EPOCH_OFFSET", "0"))
resume_adapter_dir = os.environ.get("RESUME_ADAPTER_DIR", "").strip() or None
final_model_dirname = os.environ.get("FINAL_MODEL_DIRNAME", "final_model")
training_summary_name = os.environ.get("TRAINING_SUMMARY_NAME", "training_summary.json")
warmup_steps_override = os.environ.get("WARMUP_STEPS", "").strip()
warmup_ratio = float(os.environ.get("WARMUP_RATIO", "0.06"))

# Focal loss parameters (P minority: ~12%)
focal_gamma = 2.0
focal_alpha_p = 7.5  # Weight for P (positive)
focal_alpha_n = 1.0  # Weight for N (negative)

# Data / output (env overrides allow safe re-runs without clobbering)
output_suffix = os.environ.get("PN_OUTPUT_SUFFIX", "pn-v1")
train_data_path = os.environ.get("PN_DATA_PATH", "/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl")
output_dir = os.environ.get("OUTPUT_DIR", f"./gpt-oss-20b-focal-{output_suffix}")
checkpoint_dir = f"{output_dir}/checkpoints"

use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

print(f"\nConfiguration:")
print(f"  Output directory: {output_dir}")
print(f"  Output suffix: {output_suffix}")
print(f"  Checkpoint directory: {checkpoint_dir}")
print(f"  Training data: {train_data_path}")
print(f"  Max seq length: {max_seq_length}")
print(f"  Batch size: {batch_size}")
print(f"  Gradient accumulation: {gradient_accumulation_steps}")
print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
print(f"  LoRA rank: {lora_r}")
print(f"  Learning rate: {learning_rate}")
print(f"  Epochs: {num_epochs}")
if epoch_offset:
    print(f"  Epoch offset (global numbering): {epoch_offset}")
if resume_adapter_dir:
    print(f"  Resume adapter dir: {resume_adapter_dir}")
print(f"  Final model dirname: {final_model_dirname}")
print(f"  Training summary name: {training_summary_name}")
print(f"  Train seed: {train_seed}")
optim_name = os.environ.get("OPTIM", "").strip()
optim_name = optim_name or ("adamw_bnb_8bit" if BNB_AVAILABLE else "adamw_torch")
print(f"  Optim: {optim_name}")
print(f"  Focal Loss gamma: {focal_gamma}")
print(f"  Class weights - P: {focal_alpha_p}, N: {focal_alpha_n}")
print(f"  BF16 support: {use_bf16}")
print(f"  bitsandbytes available: {BNB_AVAILABLE}")

Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

# Set all RNG seeds (python/numpy/torch) for reproducibility.
set_seed(train_seed)


# ---------------------------
# Memory cleanup helper
# ---------------------------
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------
# Custom Focal Loss (P/N-only)
# ---------------------------
class PNOnlyFocalLoss(nn.Module):
    """Focal loss that applies only to P/N tokens, ignoring others."""

    def __init__(self, tokenizer, gamma=2.0, alpha_p=7.5, alpha_n=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha_p = alpha_p
        self.alpha_n = alpha_n

        def last_token_id(s):
            ids = tokenizer.encode(s, add_special_tokens=False)
            return [ids[-1]] if ids else []

        self.p_token_ids = sorted(set(
            last_token_id("P") + last_token_id(" P") + last_token_id("\nP")
        ))
        self.n_token_ids = sorted(set(
            last_token_id("N") + last_token_id(" N") + last_token_id("\nN")
        ))

        print(f"P token IDs (last tokens only): {self.p_token_ids}")
        print(f"N token IDs (last tokens only): {self.n_token_ids}")

        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    def forward(self, logits, targets, batch_size=None):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma

        weights = torch.zeros_like(ce_loss)

        if batch_size is not None:
            pn_mask = torch.zeros_like(targets, dtype=torch.bool)
            p_tensor = torch.tensor(self.p_token_ids, device=targets.device, dtype=targets.dtype)
            n_tensor = torch.tensor(self.n_token_ids, device=targets.device, dtype=targets.dtype)

            for p_id in p_tensor:
                pn_mask |= (targets == p_id)
            for n_id in n_tensor:
                pn_mask |= (targets == n_id)

            seq_len = len(targets) // batch_size
            pn_mask_2d = pn_mask.view(batch_size, seq_len)

            has_pn = pn_mask_2d.any(dim=1)
            if has_pn.any():
                rev_mask = torch.flip(pn_mask_2d, dims=[1])
                last_pos_from_end = rev_mask.float().argmax(dim=1)
                last_pos = seq_len - 1 - last_pos_from_end

                batch_indices = torch.arange(batch_size, device=targets.device)
                select_2d = torch.zeros_like(pn_mask_2d)
                select_2d[batch_indices[has_pn], last_pos[has_pn]] = True
                select = select_2d.view(-1)

                for p_id in p_tensor:
                    p_mask = select & (targets == p_id)
                    weights[p_mask] = self.alpha_p
                for n_id in n_tensor:
                    n_mask = select & (targets == n_id)
                    weights[n_mask] = self.alpha_n
        else:
            p_tensor = torch.tensor(self.p_token_ids, device=targets.device, dtype=targets.dtype)
            n_tensor = torch.tensor(self.n_token_ids, device=targets.device, dtype=targets.dtype)
            for p_id in p_tensor:
                weights[targets == p_id] = self.alpha_p
            for n_id in n_tensor:
                weights[targets == n_id] = self.alpha_n

        focal_loss = weights * focal_term * ce_loss

        num_pn_tokens = (weights > 0).sum().item()
        if num_pn_tokens > 0:
            return focal_loss.sum() / num_pn_tokens
        mask = targets != -100
        if mask.sum() > 0:
            return focal_loss.sum() / mask.sum()
        return focal_loss.mean()


# ---------------------------
# P/N Token Monitoring Callback
# ---------------------------
class PNTokenMonitorCallback(TrainerCallback):
    """Monitor P/N token ratio during training."""

    def __init__(self, trainer_ref):
        self.trainer = trainer_ref

    def on_log(self, args, state, control, logs=None, **kwargs):
        samples = getattr(self.trainer, "samples_seen", 0)
        answers = getattr(self.trainer, "answers_seen", 0)
        if samples > 0:
            pn_ratio = answers / samples
            if logs is not None:
                logs["pn_answer_ratio"] = pn_ratio
            if state.global_step % args.logging_steps == 0:
                print(f"  P/N answer ratio: {pn_ratio:.2%} ({answers}/{samples} samples with P/N)")
        return None


# ---------------------------
# Checkpoint callback (per-epoch)
# ---------------------------
class EpochCheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_dir, epoch_offset: int = 0):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_offset = int(epoch_offset)
        self.trainer = None

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) + self.epoch_offset
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}"
        print(f"\nSaving checkpoint for Epoch {epoch} to {checkpoint_path}")

        try:
            if self.trainer is not None:
                self.trainer.save_model(checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
            else:
                model = kwargs.get("model", None)
                tokenizer = kwargs.get("tokenizer", None)
                if model is not None and tokenizer is not None:
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    print(f"Model & tokenizer saved via fallback to {checkpoint_path}")
                else:
                    print("Warning: Cannot save checkpoint - trainer/model/tokenizer not accessible")
        except Exception as e:
            print(f"Error while saving checkpoint: {e}")

        state_dict = {
            "epoch": epoch,
            "global_step": getattr(state, "global_step", None),
            "training_loss": state.log_history[-1].get("loss", None) if state.log_history else None,
        }
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

        cleanup_memory()


# ---------------------------
# Load model & tokenizer
# ---------------------------
print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

cleanup_memory()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully")
cleanup_memory()


# ---------------------------
# Add / load LoRA adapters
# ---------------------------
if resume_adapter_dir:
    print("\nLoading existing LoRA adapter (no stacking) ...")
    print(f"  Adapter path: {resume_adapter_dir}")
    # Avoid get_peft_model() here; it can allocate a new adapter on top of the loaded one.
    model = PeftModel.from_pretrained(model, resume_adapter_dir, is_trainable=True)
    print("LoRA adapter loaded and set trainable")
else:
    print("\nAdding LoRA adapters with gradient checkpointing...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=train_seed,
        use_rslora=False,
        loftq_config=None,
    )
    print("LoRA adapters added")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
cleanup_memory()


# ---------------------------
# Load and prepare dataset
# ---------------------------
print("\n" + "=" * 80)
print("LOADING AND PREPARING DATASET")
print("=" * 80)

print("\nLoading training dataset...")
train_dataset = load_dataset(
    "json",
    data_files=train_data_path,
    split="train",
    keep_in_memory=False,
)

print(f"Training samples: {len(train_dataset)}")


def format_messages(batch):
    texts = []
    for msgs in batch["messages"]:
        if msgs and len(msgs) > 0 and msgs[0].get("role") == "system":
            pass
        else:
            system_msg = {
                "role": "system",
                "content": "You are a materials science assistant. Given a chemical composition, answer only with 'P' (positive) or 'N' (negative).",
            }
            msgs = [system_msg] + msgs

        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


print("Converting messages to text format...")
train_dataset = train_dataset.map(
    format_messages,
    batched=True,
    batch_size=1000,
    remove_columns=train_dataset.column_names,
    num_proc=1,
    desc="Formatting messages",
)
print("Dataset prepared with 'text' column")


# ---------------------------
# Training stats
# ---------------------------
steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * gradient_accumulation_steps))
total_steps = steps_per_epoch * num_epochs
if warmup_steps_override:
    warmup_steps = int(warmup_steps_override)
else:
    warmup_steps = max(10, int(total_steps * warmup_ratio))

print(f"\nTraining statistics:")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print(f"  Warmup steps: {warmup_steps}")


# ---------------------------
# Custom trainer with P/N focal loss
# ---------------------------
class PNOnlyFocalSFTTrainer(SFTTrainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = PNOnlyFocalLoss(tokenizer=tokenizer, gamma=focal_gamma, alpha_p=focal_alpha_p, alpha_n=focal_alpha_n)
        self.answers_seen = 0
        self.samples_seen = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        batch_size = labels.shape[0]

        labels_flat = shift_labels.view(-1)
        pn_mask = torch.zeros_like(labels_flat, dtype=torch.bool)
        for t in self.focal_loss.p_token_ids + self.focal_loss.n_token_ids:
            pn_mask |= (labels_flat == t)

        seq_len = labels_flat.shape[0] // batch_size
        final_pn_count = 0
        for i in range(batch_size):
            sample_start = i * seq_len
            sample_end = (i + 1) * seq_len
            sample_pn = pn_mask[sample_start:sample_end].nonzero().flatten()
            if len(sample_pn) > 0:
                final_pn_count += 1

        self.answers_seen += final_pn_count
        self.samples_seen += batch_size

        if hasattr(self, "state") and self.state.global_step < 2:
            print(f"[Debug] Step {self.state.global_step}: batch_size={batch_size}, samples_with_pn={final_pn_count}")

        loss = self.focal_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            labels_flat,
            batch_size=batch_size,
        )

        return (loss, outputs) if return_outputs else loss


# ---------------------------
# Training arguments and trainer
# ---------------------------
print("\n" + "=" * 80)
print("SETTING UP TRAINING")
print("=" * 80)

if not BNB_AVAILABLE:
    print("Warning: bitsandbytes is not available. Using standard AdamW optimizer.")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    bf16=use_bf16,
    fp16=not use_bf16,
    logging_steps=logging_steps,
    save_strategy="no",
    optim=optim_name,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    seed=train_seed,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.5,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    remove_unused_columns=True,
    eval_strategy="no",
    do_eval=False,
)

trainer = PNOnlyFocalSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
)


# ---------------------------
# Callbacks and training start
# ---------------------------
epoch_callback = EpochCheckpointCallback(checkpoint_dir, epoch_offset=epoch_offset)
epoch_callback.trainer = trainer
trainer.add_callback(epoch_callback)

monitor_callback = PNTokenMonitorCallback(trainer)
trainer.add_callback(monitor_callback)

print("\n" + "=" * 80)
print(f"STARTING TRAINING ({num_epochs} EPOCHS)")
if epoch_offset:
    print(f"Global epoch numbering: starts at epoch_{epoch_offset + 1}")
if resume_adapter_dir:
    print("Resume mode: adapter-only (no optimizer/scheduler state)")
print("Key settings:")
print("  - P/N-only focal loss (non-P/N tokens have zero weight)")
print("  - Loss applied only to last P/N classification token per sample")
print(f"  - Batch size: {batch_size}")
print("  - Gradient accumulation for stability")
print("  - Checkpoint after each epoch (custom callback)")
print("=" * 80)

start_time = time.time()
trainer_stats = trainer.train()
training_time = time.time() - start_time

# Save final model
print("\nSaving final model...")
final_model_path = Path(output_dir) / final_model_dirname
trainer.save_model(final_model_path)
print(f"Final model saved to {final_model_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)


# ---------------------------
# Save training summary
# ---------------------------
p_ids = getattr(getattr(trainer, "focal_loss", None), "p_token_ids", [47])
n_ids = getattr(getattr(trainer, "focal_loss", None), "n_token_ids", [52])

summary = {
    "configuration": {
        "output_suffix": output_suffix,
        "train_data_path": train_data_path,
        "train_seed": train_seed,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio if not warmup_steps_override else None,
        "num_epochs": num_epochs,
        "epoch_offset": epoch_offset,
        "resume_adapter_dir": resume_adapter_dir,
        "final_model_dirname": final_model_dirname,
        "training_summary_name": training_summary_name,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "optim": optim_name,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "loss_type": "P/N-only focal loss (v1)",
    },
    "focal_loss_config": {
        "gamma": focal_gamma,
        "alpha_p": focal_alpha_p,
        "alpha_n": focal_alpha_n,
        "p_token_ids": p_ids,
        "n_token_ids": n_ids,
        "loss_strategy": "Only P/N classification tokens contribute to loss (last token per sample)",
    },
    "training_results": {
        "final_loss": trainer_stats.metrics.get("train_loss", "N/A") if hasattr(trainer_stats, "metrics") else "N/A",
        "total_steps": trainer_stats.metrics.get("train_steps", total_steps) if hasattr(trainer_stats, "metrics") else total_steps,
        "training_time_seconds": training_time,
        "training_time_hours": training_time / 3600,
        "total_samples_processed": trainer.samples_seen,
        "samples_with_pn_answer": trainer.answers_seen,
        "pn_answer_ratio": trainer.answers_seen / trainer.samples_seen if trainer.samples_seen > 0 else 0,
    },
    "checkpoints": [f"epoch_{epoch_offset + i}" for i in range(1, num_epochs + 1)] + [final_model_dirname],
    "critical_fixes_applied": [
        "P/N-only focal loss with zero weights for non-P/N tokens",
        "Loss applied to last P/N token per sample",
        "CrossEntropyLoss with ignore_index=-100",
        "Messages to text conversion with chat template",
        "Balanced class weights for P minority",
        "P/N token counting and ratio monitoring",
    ],
}

summary_path = Path(output_dir) / training_summary_name
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nTraining summary saved to: {summary_path}")

print("\nTraining Summary:")
print(f"  Total training time: {training_time/3600:.2f} hours")
print(f"  Final loss: {summary['training_results']['final_loss']}")
print(f"  Total steps: {summary['training_results']['total_steps']}")
print(f"  Samples with P/N answer: {summary['training_results']['samples_with_pn_answer']:,}")
print(f"  Total samples processed: {summary['training_results']['total_samples_processed']:,}")
print(f"  P/N answer ratio: {summary['training_results']['pn_answer_ratio']:.2%}")
print("\nCheckpoints saved:")
for checkpoint in summary["checkpoints"]:
    print(f"  - {output_dir}/{checkpoint}")

print("\n" + "=" * 80)
print("All training completed successfully!")
print("P/N-only focal loss training finished.")
print("=" * 80)
