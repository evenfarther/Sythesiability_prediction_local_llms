#!/usr/bin/env bash
#SBATCH --job-name=qwen314b_randinit_seed
#SBATCH --output=qwen314b_randinit_seed.%N.%j.out
#SBATCH --error=qwen314b_randinit_seed.%N.%j.err
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

export MODEL_NAME_OR_PATH="/DATA/qwen3_4b/models/Qwen3-14B"  # overridden per-seed below
export TRAIN_DATA_PATH="/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl"
export MAX_SEQ_LEN=180
export LR=2e-4
export EPOCHS=10
export LOGGING_STEPS=10

export FOCAL_GAMMA=2.0
export FOCAL_ALPHA_P=7.5
export FOCAL_ALPHA_N=1.0

export LORA_R=64
export LORA_ALPHA=128
export LORA_DROPOUT=0.0

export BF16=1
export USE_GC=0
export OPTIM=adamw_torch_fused

export SAVE_STRATEGY=epoch
export SAVE_STEPS=0

export PER_DEVICE_BATCH=64
export GRAD_ACCUM=20

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/DATA/qwen3_4b/base_evaluation/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

BASES_DIR="${HERE}/random_base_fp16"

echo "============================================================"
echo "Arch: Qwen3-14B random-init base + QLoRA-only (10-epoch)"
echo "Seeds: ${SEEDS[*]}"
echo "Epochs per seed: ${EPOCHS}"
echo "Random bases: ${BASES_DIR}/seed_<seed>/"
echo "Host: $(hostname)"
echo "Job start: $(date)"
echo "============================================================"

for i in "${!SEEDS[@]}"; do
  IDX=$((i + 1))
  SEED="${SEEDS[$i]}"
  RUN_DIR="${HERE}/seed_${IDX}"
  BASE_DIR="${BASES_DIR}/seed_${SEED}"

  mkdir -p "${RUN_DIR}"
  echo "${SEED}" > "${RUN_DIR}/seed.txt"

  if [[ -f "${RUN_DIR}/DONE" ]]; then
    echo "Seed ${IDX} (${SEED}) already DONE -> skipping."
    continue
  fi

  if [[ ! -f "${BASE_DIR}/DONE" ]]; then
    echo "Seed ${IDX} (${SEED}): random-init base missing -> creating: ${BASE_DIR}"
    mkdir -p "${BASE_DIR}"
    echo "${SEED}" > "${BASE_DIR}/seed.txt"
    python "${HERE}/make_random_init_qwen3_14b.py" \
      --out_dir "${BASE_DIR}" \
      --seed "${SEED}" \
      --dtype float16 \
      --max_shard_size 2GB \
      --drop_quant_config \
      --trust_remote_code \
      2>&1 | tee "${BASE_DIR}/make_random_init.log"
    echo "DONE" > "${BASE_DIR}/DONE"
  fi

  export OUTPUT_DIR="${RUN_DIR}"
  export SEED="${SEED}"
  export BASE_MODEL_PATH="${BASE_DIR}"
  export MODEL_NAME_OR_PATH="${BASE_DIR}"
  export REQUIRE_RANDOM_BASE=1

  cat > "${RUN_DIR}/run_config.txt" <<EOF
arch=qwen3-14b
mode=random_init_base + qlora_only
seed=${SEED}
epochs=${EPOCHS}
per_device_batch=${PER_DEVICE_BATCH}
grad_accum=${GRAD_ACCUM}
effective_batch=$((PER_DEVICE_BATCH * GRAD_ACCUM))
lr=${LR}
max_seq_len=${MAX_SEQ_LEN}
lora_r=${LORA_R}
lora_alpha=${LORA_ALPHA}
lora_dropout=${LORA_DROPOUT}
optim=${OPTIM}
bf16=${BF16}
train_data=${TRAIN_DATA_PATH}
base_model_path=${BASE_MODEL_PATH}
EOF

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${HERE}/train_qwen3_14b_pn_focal_random_init_qlora.py" > "${RUN_DIR}/train_script.sha256"
    sha256sum "${HERE}/../common/seeds.sh" > "${RUN_DIR}/seeds.sha256"
  fi

  echo "------------------------------------------------------------"
  echo "Seed ${IDX}/${#SEEDS[@]}: ${SEED}"
  echo "Run dir:  ${RUN_DIR}"
  echo "Base dir: ${BASE_DIR}"
  echo "Start:    $(date)"
  echo "------------------------------------------------------------"

  python "${HERE}/train_qwen3_14b_pn_focal_random_init_qlora.py" 2>&1 | tee "${RUN_DIR}/train.log"

  echo "DONE" > "${RUN_DIR}/DONE"
  echo "End: $(date)"
done

echo "============================================================"
echo "All seeds finished (10-epoch)."
echo "Job end: $(date)"
echo "============================================================"
