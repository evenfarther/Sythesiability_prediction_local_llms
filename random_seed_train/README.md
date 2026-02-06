# random_seed_train (pretrained LLMs)

Pretrained LLM multi-seed reproducibility runs (QLoRA) for:
- `gpt-oss-20b`
- `deepseek-r1-distill-qwen2.5-14b`
- `qwen3-14b`

## Reproduce

0. Dataset paths (fixed):
   - train: `/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl`
   - valid (final test): `/DATA/npj_compt.mat_project_github/data/valid_llm_pn.jsonl`
   - valid_hem_only: `/DATA/npj_compt.mat_project_github/data/valid_hem_only_llm_pn.jsonl`
1. Define shared seeds in `random_seed_train/common/seeds.sh`.
2. Submit training per model:
   - `cd random_seed_train/<model>`
   - `sbatch submit_seeds.sbatch`
3. Current policy in this GitHub copy:
   - `submit_seeds.sbatch` uses `export EPOCHS=10` for all 3 models.
4. Submit checkpoint evaluation:
   - `sbatch submit_eval.sbatch`

## Outputs

Training produces per-seed run directories (created at runtime, not stored in this GitHub copy). Expected layout:

- GPT-OSS: `seed_i/checkpoints/epoch_*/` (intermediate checkpoints)
- DeepSeek/Qwen3: `seed_i/checkpoint-*/` (intermediate checkpoints)
- Common markers/logs:
  - `seed_i/DONE` (created by training when finished)
  - `seed_i/eval.log` (created by evaluation)
  - `seed_i/validation_metrics_pn.csv` (created by evaluation)

## Evaluation rules

- Use P/N token logits (not generated text parsing).
- Evaluate checkpoints only.
- Do not evaluate `seed_i/final_model`.
- Keep `valid_llm_pn.jsonl` as final test only (no tuning/threshold selection).

## Cluster knobs

Per-model submit scripts set GPU type, partition, memory, cache directories, and base model path/ID.
If you run on a different cluster or filesystem layout, adjust `submit_seeds.sbatch` and `submit_eval.sbatch`
inside each model directory.
