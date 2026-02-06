#!/usr/bin/env bash
# Evaluate checkpoints only (skip seed_i/final_model).
#
# Outputs per seed:
#   - seed_i/validation_metrics_pn.csv
#   - seed_i/eval.log
#
# Shared evaluator in this repo:
#   /DATA/npj_compt.mat_project_github/random_seed_train/common/eval_pn_checkpoints.py

#SBATCH --job-name=gptoss20b_seed_eval
#SBATCH --output=gptoss20b_seed_eval.%N.%j.out
#SBATCH --error=gptoss20b_seed_eval.%N.%j.err
##SBATCH --time=100:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G

set -euo pipefail

HERE="$(pwd)"
source "${HERE}/../common/seeds.sh"

source /TGM/Apps/ANACONDA/2024.10/etc/profile.d/conda.sh
conda activate llm

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/DATA/gpt-oss/20b/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export PYTHONUNBUFFERED=1

export UNSLOTH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE=0
export UNSLOTH_ENABLE_FLEX_ATTENTION=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128"

VAL_JSONL="/DATA/npj_compt.mat_project_github/data/valid_llm_pn.jsonl"
BASE_MODEL="unsloth/gpt-oss-20b-unsloth-bnb-4bit"
EVAL_SCRIPT="/DATA/npj_compt.mat_project_github/random_seed_train/common/eval_pn_checkpoints.py"

echo "============================================================"
echo "Eval (logit-based P/N) for pretrained GPT-OSS-20B"
echo "Seeds: ${SEEDS[*]}"
echo "Valid: ${VAL_JSONL}"
echo "Base:  ${BASE_MODEL}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "============================================================"

for i in "${!SEEDS[@]}"; do
  IDX=$((i + 1))
  SEED="${SEEDS[$i]}"
  RUN_DIR="${HERE}/seed_${IDX}"
  CKPTS_DIR="${RUN_DIR}/checkpoints"
  OUT_CSV="${RUN_DIR}/validation_metrics_pn.csv"

  if [[ ! -f "${RUN_DIR}/DONE" ]]; then
    echo "Seed ${IDX} (${SEED}): missing DONE -> skipping: ${RUN_DIR}"
    continue
  fi
  if [[ -f "${OUT_CSV}" ]]; then
    echo "Seed ${IDX} (${SEED}): output already exists -> skipping: ${OUT_CSV}"
    continue
  fi
  if [[ ! -d "${CKPTS_DIR}" ]]; then
    echo "Seed ${IDX} (${SEED}): missing checkpoints dir -> skipping: ${CKPTS_DIR}"
    continue
  fi
  if ! ls -d "${CKPTS_DIR}/epoch_"* >/dev/null 2>&1; then
    echo "Seed ${IDX} (${SEED}): no epoch_* dirs -> skipping: ${CKPTS_DIR}"
    continue
  fi

  echo "------------------------------------------------------------"
  echo "Seed ${IDX}/${#SEEDS[@]}: ${SEED}"
  echo "Ckpts: ${CKPTS_DIR} (regex: ^epoch_)"
  echo "Out:   ${OUT_CSV}"
  echo "Time:  $(date)"
  echo "------------------------------------------------------------"

  python "${EVAL_SCRIPT}" \
    --base_model "${BASE_MODEL}" \
    --checkpoints_dir "${CKPTS_DIR}" \
    --ckpt_regex '^epoch_' \
    --val_jsonl "${VAL_JSONL}" \
    --max_seq_len 180 \
    --batch_size 32 \
    --cache_dir /DATA/gpt-oss/cache_eval_pn \
    --output_csv "${OUT_CSV}" \
    --no_baseline \
    2>&1 | tee "${RUN_DIR}/eval.log"
done

echo "============================================================"
echo "Eval done."
echo "End: $(date)"
echo "============================================================"
