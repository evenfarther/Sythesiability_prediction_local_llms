# ml_models reproducibility

This directory contains composition-only classical ML baselines for HEM synthesizability (P/N):
- `svm_linearsvc`
- `xgb`
- `mlp_small`, `mlp_medium`, `mlp_large`
- `roost_aviary` (optional)

## Fixed data policy

Use the same fixed files:
- train: `/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl`
- valid (final test): `/DATA/npj_compt.mat_project_github/data/valid_llm_pn.jsonl`
- valid_hem_only: `/DATA/npj_compt.mat_project_github/data/valid_hem_only_llm_pn.jsonl`

Do not use `valid_llm_pn.jsonl` for tuning, early stopping, or threshold selection.

## Environment

Recommended execution pattern:
- `CONDA_NO_PLUGINS=true conda run -n hem-ml ...`

## Reproduce from scratch

1. Build dataset cache and features:
   - `python ml_models/scripts/01_extract_jsonl_to_table.py --overwrite`
   - `python ml_models/scripts/02_parse_and_audit.py --overwrite`
2. Train/evaluate models (repeat per seed):
   - SVM: `python ml_models/scripts/04_train_eval_svm.py --seed <seed> --inner_split_seed <seed> --overwrite`
   - XGB: `python ml_models/scripts/05_train_eval_xgb.py --seed <seed> --inner_split_seed <seed> --overwrite`
   - MLP: `python ml_models/scripts/07_train_eval_mlp.py --arch <small|medium|large> --seed <seed> --inner_split_seed <seed> --overwrite`
3. Aggregate per-model seed runs:
   - `python ml_models/scripts/08_aggregate_seed_runs_json.py --run_dir ml_models/runs/<model> --overwrite`
4. Build total summary:
   - `python ml_models/scripts/09_make_total_summary.py --overwrite`

## Seeds and splits

- `seed` controls model randomness (and is used to name `seed=<seed>/` output directories).
- `inner_split_seed` controls the internal train/tune split used for selecting thresholds.

Keep `inner_split_seed` fixed if you want to compare models under the same tuning split. In the included runs,
we use `inner_split_seed=<seed>` by default.

## Output locations

- Artifacts: `ml_models/artifacts/`
- Per-model runs: `ml_models/runs/<model>/seed=<seed>/`
- Per-model summaries: `ml_models/runs/<model>/seed_summary.json`
- Combined summary: `ml_models/runs/total_summary.json`, `ml_models/runs/total_summary.csv`

## Using existing runs

This GitHub copy includes `ml_models/runs/` for convenience. If you add new seeds or re-run a model,
rebuild summaries with:

- `python ml_models/scripts/08_aggregate_seed_runs_json.py --run_dir ml_models/runs/<model> --overwrite`
- `python ml_models/scripts/09_make_total_summary.py --overwrite`

## Metric notes

- `valid_hem_only` has `n_neg=0`; recall(TPR) is the primary meaningful metric.
- For LLM-side comparison consistency, keep final-test discipline and report metrics transparently.
