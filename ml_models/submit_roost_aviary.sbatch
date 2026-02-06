#!/usr/bin/env bash
# Train + evaluate Roost (composition-only GNN) via `aviary-models` and write run artifacts.
#
# Usage:
#   cd /DATA/npj_compt.mat_project/ml_models
#   sbatch submit_roost_aviary.sbatch
#
# Optional env overrides:
#   SEED=3409 sbatch submit_roost_aviary.sbatch                      # run 1 seed only
#   SEEDS="3408 3409 3410 3411 3412" sbatch submit_roost_aviary.sbatch  # run multiple seeds (default)
#   INNER_SPLIT_SEED=3408 sbatch submit_roost_aviary.sbatch          # fix tune split seed (applies to all)
#   DEVICE=cpu EPOCHS=10 BATCH_SIZE=256 sbatch submit_roost_aviary.sbatch
#   SMOKE=1 sbatch submit_roost_aviary.sbatch
#   OVERWRITE=1 sbatch submit_roost_aviary.sbatch

#SBATCH --job-name=ml_roost_aviary
#SBATCH --output=ml_roost_aviary.%N.%j.out
#SBATCH --error=ml_roost_aviary.%N.%j.err
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --nodes=1
##SBATCH --time=48:00:00

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/ml_models/scripts" ]]; then
  ROOT="${SUBMIT_DIR}"                # submitted from repo root
elif [[ -d "${SUBMIT_DIR}/scripts" && -f "${SUBMIT_DIR}/README.md" ]]; then
  ROOT="$(realpath "${SUBMIT_DIR}/..")"  # submitted from ml_models
else
  echo "[error] Could not locate repo root from submit dir: ${SUBMIT_DIR}" >&2
  echo "Submit from repo root (/DATA/npj_compt.mat_project) or ml_models directory." >&2
  exit 2
fi
cd "${ROOT}"

source /TGM/Apps/ANACONDA/2024.10/etc/profile.d/conda.sh
export CONDA_NO_PLUGINS=true
conda activate hem-ml

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
ELEM_EMBEDDING="${ELEM_EMBEDDING:-matscholar200}"
OUT_ROOT="${OUT_ROOT:-ml_models/runs/roost_aviary}"

# Seed selection:
# - If SEEDS is set, use it (space-separated).
# - Else if SEED is set, run just that one seed.
# - Else default to 5 seeds: 3408..3412
SEEDS_STR=""
if [[ -n "${SEEDS:-}" ]]; then
  SEEDS_STR="${SEEDS}"
elif [[ -n "${SEED:-}" ]]; then
  SEEDS_STR="${SEED}"
else
  SEEDS_STR="3408 3409 3410 3411 3412"
fi
read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi
SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE_FLAG="--smoke"
  # Avoid clobbering full runs unless explicitly overridden.
  if [[ "${OUT_ROOT}" == "ml_models/runs/roost_aviary" ]]; then
    OUT_ROOT="ml_models/runs/roost_aviary_smoke"
  fi
fi

echo "============================================================"
echo "Job: Roost (aviary-models) baseline"
echo "Seeds: ${SEEDS_ARR[*]}"
echo "Inner split seed (override): ${INNER_SPLIT_SEED:-<per-seed default>}"
echo "Device: ${DEVICE}"
echo "Epochs: ${EPOCHS} (patience=${PATIENCE})"
echo "Batch: ${BATCH_SIZE} (eval=${EVAL_BATCH_SIZE})"
echo "Num workers: ${NUM_WORKERS}"
echo "Elem embedding: ${ELEM_EMBEDDING}"
echo "Out root: ${OUT_ROOT}"
echo "Smoke: ${SMOKE:-0}"
echo "Overwrite: ${OVERWRITE:-0}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "============================================================"

REQ_FILES=(
  config.json
  threshold.json
  metrics_valid.json
  metrics_valid_hem_only.json
  predictions_valid.parquet
  predictions_valid_hem_only.parquet
  roc_valid.npz
  model.pt
  training_history.json
)

for SEED in "${SEEDS_ARR[@]}"; do
  if [[ "${OUT_ROOT}" == /* ]]; then
    RUN_DIR="${OUT_ROOT}/seed=${SEED}"
  else
    RUN_DIR="${ROOT}/${OUT_ROOT}/seed=${SEED}"
  fi
  INNER="${INNER_SPLIT_SEED:-${SEED}}"

  if [[ "${OVERWRITE:-0}" != "1" ]]; then
    all_present=1
    for f in "${REQ_FILES[@]}"; do
      if [[ ! -f "${RUN_DIR}/${f}" ]]; then
        all_present=0
        break
      fi
    done
    if [[ "${all_present}" == "1" ]]; then
      echo "------------------------------------------------------------"
      echo "Seed ${SEED}: outputs already present -> skipping: ${RUN_DIR}"
      continue
    fi
  fi

  mkdir -p "${RUN_DIR}"

  echo "------------------------------------------------------------"
  echo "Seed: ${SEED}"
  echo "Inner split seed: ${INNER}"
  echo "Run dir: ${RUN_DIR}"
  echo "Time: $(date)"
  echo "------------------------------------------------------------"

  python -u ml_models/scripts/10_train_eval_roost_aviary.py \
    --seed "${SEED}" \
    --inner_split_seed "${INNER}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --elem_embedding "${ELEM_EMBEDDING}" \
    --out_root "${OUT_ROOT}" \
    ${SMOKE_FLAG} \
    ${OVERWRITE_FLAG} \
    2>&1 | tee "${RUN_DIR}/train_eval.log"
done

echo "============================================================"
echo "All seeds finished."
echo "End: $(date)"
echo "============================================================"
