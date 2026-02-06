#!/usr/bin/env bash
# Train + evaluate SVM (LinearSVC baseline) and write run artifacts.
#
# Usage:
#   cd /DATA/npj_compt.mat_project/ml_models
#   sbatch submit_svm.sbatch
#
# Optional env overrides:
#   SEED=3409 INNER_SPLIT_SEED=3409 sbatch submit_svm.sbatch
#   OVERWRITE=1 sbatch submit_svm.sbatch

#SBATCH --job-name=ml_svm
#SBATCH --output=ml_svm.%N.%j.out
#SBATCH --error=ml_svm.%N.%j.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --partition=gpu
#SBATCH --nodes=1
##SBATCH --time=24:00:00

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

SEED="${SEED:-3412}"
INNER_SPLIT_SEED="${INNER_SPLIT_SEED:-3412}"

OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

echo "============================================================"
echo "Job: SVM LinearSVC baseline"
echo "Seed: ${SEED}"
echo "Inner split seed: ${INNER_SPLIT_SEED}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Overwrite: ${OVERWRITE:-0}"
echo "============================================================"

python ml_models/scripts/04_train_eval_svm.py \
  --seed "${SEED}" \
  --inner_split_seed "${INNER_SPLIT_SEED}" \
  ${OVERWRITE_FLAG}

echo "============================================================"
echo "DONE"
echo "End: $(date)"
echo "============================================================"
