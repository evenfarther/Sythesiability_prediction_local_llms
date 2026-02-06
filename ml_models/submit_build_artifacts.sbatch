#!/usr/bin/env bash
# Build full ML artifacts (parquet cache + parsed/features/audit).
#
# Usage:
#   cd /DATA/npj_compt.mat_project/ml_models
#   sbatch submit_build_artifacts.sbatch
#
# Optional env overrides:
#   OVERWRITE=1 sbatch submit_build_artifacts.sbatch

#SBATCH --job-name=ml_build_artifacts
#SBATCH --output=ml_build_artifacts.%N.%j.out
#SBATCH --error=ml_build_artifacts.%N.%j.err
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

OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

echo "============================================================"
echo "Job: build artifacts (01_extract -> 02_parse_and_audit)"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Overwrite: ${OVERWRITE:-0}"
echo "============================================================"

python ml_models/scripts/01_extract_jsonl_to_table.py ${OVERWRITE_FLAG}
python ml_models/scripts/02_parse_and_audit.py ${OVERWRITE_FLAG}

echo "============================================================"
echo "DONE"
echo "End: $(date)"
echo "============================================================"
