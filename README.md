# HEM Synthesizability (P/N) — Revision Repro Package

This repository contains code and Slurm submit scripts used for the *npj Computational Materials* revision.
The task is **HEM (High-Entropy Materials) synthesizability binary classification (P/N)**.

## Layout

- `data/`: fixed PN datasets used in this package (`train`, `valid`, `valid_hem_only`)
- `random_seed_train/`: pretrained open-weight LLMs (multi-seed reproducibility; QLoRA)
- `random_initializaiton/`: random-init control LLMs (isolate pretraining effect; QLoRA-only)
- `ml_models/`: classical ML baselines (composition-only)

## Datasets

The datasets are included under `data/` and are consumed by both LLM and classical ML pipelines:

- train: `/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl`
- valid (final test): `/DATA/npj_compt.mat_project_github/data/valid_llm_pn.jsonl`
- valid_hem_only: `/DATA/npj_compt.mat_project_github/data/valid_hem_only_llm_pn.jsonl`

The submit scripts hard-code the absolute path above. If you place the repo elsewhere, update the JSONL paths
in the relevant submit scripts and/or training scripts.

## Quick start (Slurm)

LLM runs are organized by model directories. For any model in either `random_seed_train/` or
`random_initializaiton/`:

1. Train (creates `seed_*` run directories at runtime):
   - `cd <group>/<model>`
   - `sbatch submit_seeds.sbatch`
2. Evaluate intermediate checkpoints:
   - `sbatch submit_eval.sbatch`
3. Collect per-seed evaluation CSVs:
   - `seed_i/validation_metrics_pn.csv`

For classical ML baselines, see `ml_models/README.md`.

## LLM base models

The LLM experiments fine-tune LoRA adapters on top of the following open-weight base models
(typically loaded in 4-bit for QLoRA):

- GPT-OSS-20B: `unsloth/gpt-oss-20b-unsloth-bnb-4bit` (https://huggingface.co/unsloth/gpt-oss-20b-unsloth-bnb-4bit)
- Qwen3-14B: `unsloth/Qwen3-14B-unsloth-bnb-4bit` (https://huggingface.co/unsloth/Qwen3-14B-unsloth-bnb-4bit)
- DeepSeek-R1-Distill-Qwen-14B: `unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit` (https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit)

Some submit scripts refer to local filesystem mirrors of these models; the model family should match
the identifiers above.

## Evaluation rules

- `valid_llm_pn.jsonl` is treated as a **final test** split (no tuning / threshold selection on it).
- LLM evaluation is **token-logits based** (P vs N token logits), not parsing generated text.
- For LLMs we evaluate **intermediate checkpoints only** (do **not** evaluate `seed_i/final_model`).
- `valid_hem_only` currently has `n_neg=0`, so only **recall(TPR)** is meaningful.

## Artifacts

- LLM checkpoints/adapters are not stored in this GitHub copy; training/evaluation jobs write them under
  per-seed run directories.
