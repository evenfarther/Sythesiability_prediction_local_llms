#!/usr/bin/env bash
# Aggregate run metrics into Reviewer 1 table.
#
# Usage:
#   cd /DATA/npj_compt.mat_project/ml_models
#   sbatch submit_make_table.sbatch

#SBATCH --job-name=ml_make_table
#SBATCH --output=ml_make_table.%N.%j.out
#SBATCH --error=ml_make_table.%N.%j.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --nodes=1
##SBATCH --time=01:00:00

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

echo "============================================================"
echo "Job: make Reviewer 1 table"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "============================================================"

python ml_models/scripts/06_make_reviewer1_table.py

echo "============================================================"
echo "DONE"
echo "End: $(date)"
echo "============================================================"
