# random_initializaiton (random-init control)

Random-init control runs to separate supervised-learning effect from pretraining effect.
Models:
- `gpt-oss-20b`
- `deepseek-r1-distill-qwen2.5-14b`
- `qwen3-14b`

## Reproduce

0. Dataset paths (fixed):
   - train: `/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl`
   - valid (final test): `/DATA/npj_compt.mat_project_github/data/valid_llm_pn.jsonl`
   - valid_hem_only: `/DATA/npj_compt.mat_project_github/data/valid_hem_only_llm_pn.jsonl`
1. Define shared seeds in `random_initializaiton/common/seeds.sh`.
2. Submit training per model:
   - `cd random_initializaiton/<model>`
   - `sbatch submit_seeds.sbatch`
3. Current policy in this GitHub copy:
   - `submit_seeds.sbatch` uses `export EPOCHS=10` for all 3 models.
   - random-init bases are created automatically if missing.
4. Submit checkpoint evaluation:
   - `sbatch submit_eval.sbatch`

## Outputs

Each model directory keeps random-initialized base checkpoints under:

- `random_base_fp16/seed_<seed>/`

These base checkpoints are created on-demand by `submit_seeds.sbatch` using `make_random_init_*.py` and are
not stored in this GitHub copy.

Training/evaluation produces per-seed run directories (created at runtime). Expected layout:

- GPT-OSS: `seed_i/checkpoints/epoch_*/`
- DeepSeek/Qwen3: `seed_i/checkpoint-*/`
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

As in `random_seed_train/`, per-model submit scripts set GPU type, partition, memory, cache directories, and
base model path/ID. If you run on a different cluster or filesystem layout, adjust `submit_seeds.sbatch` and
`submit_eval.sbatch` inside each model directory.
