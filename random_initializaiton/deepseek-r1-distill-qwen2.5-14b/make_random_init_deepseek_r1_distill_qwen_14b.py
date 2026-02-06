#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a random-initialized DeepSeek-R1-Distill-Qwen-14B checkpoint (FP16/BF16).

Loads config/tokenizer assets from `--base_model_id` and initializes model weights from
the config (no pretrained weights). Writes a sharded safetensors checkpoint plus the
tokenizer files needed for consistent prompting.
"""

import argparse
import os
import shutil

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed
from huggingface_hub import hf_hub_download


def copy_if_exists(repo_id: str, filename: str, out_dir: str) -> None:
    """Best-effort copy of auxiliary files (chat templates, generation config, etc.)."""
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy(path, os.path.join(out_dir, filename))
        print(f"[copy] {filename}")
    except Exception as e:
        print(f"[skip] {filename}: {e}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Make random-init DeepSeek-R1-Distill-Qwen-14B checkpoint (no pretrained weights)."
    )
    p.add_argument(
        "--base_model_id",
        type=str,
        default="unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
        help="HF repo id to pull config/tokenizer/template from",
    )
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for the random-init checkpoint")
    p.add_argument("--seed", type=int, required=True, help="RNG seed for model weight initialization")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_shard_size", type=str, default="2GB")
    p.add_argument(
        "--drop_quant_config",
        action="store_true",
        help="Remove quantization_config from config.json (recommended for a clean FP16 random-init checkpoint)",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders (safe if model uses custom code).",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Seed control (python/numpy/torch via transformers utility + torch.manual_seed).
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[init] base_model_id={args.base_model_id}")
    print(f"[init] out_dir={args.out_dir}")
    print(f"[init] seed={args.seed}")
    print(f"[init] dtype={args.dtype}")
    print(f"[init] max_shard_size={args.max_shard_size}")
    print(f"[init] trust_remote_code={bool(args.trust_remote_code)}")

    # 1) Load config ONLY (no weights).
    config = AutoConfig.from_pretrained(args.base_model_id, trust_remote_code=args.trust_remote_code)

    if args.drop_quant_config and hasattr(config, "quantization_config"):
        # `PretrainedConfig.to_dict()` may treat `quantization_config` as a dict-like object.
        # Remove the attribute to keep config serialization stable for FP16 checkpoints.
        print("[info] Dropping quantization_config from config for random-init checkpoint.")
        try:
            delattr(config, "quantization_config")
        except Exception:
            # Fallback: keep it as an empty dict so config.to_dict() stays valid.
            config.quantization_config = {}

    # 2) Save tokenizer and related artifacts to keep chat template consistent.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(args.out_dir)

    for fname in [
        "chat_template.jinja",
        "chat_template.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "README.md",
    ]:
        copy_if_exists(args.base_model_id, fname, args.out_dir)

    # 3) Build model from config (RANDOM WEIGHTS).
    # Avoid float32 -> float16 conversion peak by instantiating directly with the
    # requested default dtype.
    old_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        print("[build] AutoModelForCausalLM.from_config(config) (random init; no pretrained weights)")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
        # Defensive: ensure weight init ran.
        model.init_weights()
    finally:
        torch.set_default_dtype(old_default)

    # 4) Save (sharded) safetensors checkpoint.
    print(f"[save] Saving random-init checkpoint to {args.out_dir} (max_shard_size={args.max_shard_size})")
    model.save_pretrained(
        args.out_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    config.save_pretrained(args.out_dir)

    # 5) Quick sanity stats.
    with torch.no_grad():
        p0 = next(model.parameters()).detach().float().cpu()
        print("[sanity] first_param mean/std:", p0.mean().item(), p0.std().item())

    # Record seed and dtype for reproducibility.
    with open(os.path.join(args.out_dir, "random_init_meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"dtype={args.dtype}\n")
        f.write(f"base_model_id={args.base_model_id}\n")
        f.write(f"drop_quant_config={bool(args.drop_quant_config)}\n")
        f.write(f"trust_remote_code={bool(args.trust_remote_code)}\n")

    print("[done] Random-init checkpoint created.")


if __name__ == "__main__":
    main()
