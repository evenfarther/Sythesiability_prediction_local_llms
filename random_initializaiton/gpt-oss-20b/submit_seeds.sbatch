#!/usr/bin/env bash
#SBATCH --job-name=gptoss20b_randinit_seed
#SBATCH --output=gptoss20b_randinit_seed.%N.%j.out
#SBATCH --error=gptoss20b_randinit_seed.%N.%j.err
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

# Match the original GPT-OSS PN environment knobs.
export UNSLOTH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/DATA/gpt-oss/20b/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export UNSLOTH_ENABLE_FLEX_ATTENTION=0
export TORCH_COMPILE=0

# Run config (10-epoch).
export PN_DATA_PATH="/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl"
export EPOCHS=10
export PER_DEVICE_BATCH=64
export GRAD_ACCUM=20
export OPTIM=adamw_torch_fused

BASES_DIR="${HERE}/random_base_fp16"

echo "============================================================"
echo "Arch: GPT-OSS-20B random-init base + QLoRA-only (10-epoch)"
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
    python "${HERE}/make_random_init_gpt_oss_20b.py" \
      --out_dir "${BASE_DIR}" \
      --seed "${SEED}" \
      --dtype float16 \
      --max_shard_size 2GB \
      --drop_quant_config \
      2>&1 | tee "${BASE_DIR}/make_random_init.log"
    echo "DONE" > "${BASE_DIR}/DONE"
  fi

  export OUTPUT_DIR="${RUN_DIR}"
  export SEED="${SEED}"
  export BASE_MODEL_PATH="${BASE_DIR}"
  export REQUIRE_RANDOM_BASE=1

  cat > "${RUN_DIR}/run_config.txt" <<EOF
arch=gpt-oss-20b
mode=random_init_base + qlora_only
seed=${SEED}
epochs=${EPOCHS}
per_device_batch=${PER_DEVICE_BATCH}
grad_accum=${GRAD_ACCUM}
effective_batch=$((PER_DEVICE_BATCH * GRAD_ACCUM))
lr=2e-4
max_seq_len=180
lora_r=64
lora_alpha=128
lora_dropout=0.0
optim=${OPTIM}
train_data=${PN_DATA_PATH}
base_model_path=${BASE_MODEL_PATH}
EOF

  echo "------------------------------------------------------------"
  echo "Seed ${IDX}/${#SEEDS[@]}: ${SEED}"
  echo "Run dir:  ${RUN_DIR}"
  echo "Base dir: ${BASE_DIR}"
  echo "Start:    $(date)"
  echo "------------------------------------------------------------"

  python "${HERE}/train_gptoss20b_pn_focal_random_init_qlora.py" 2>&1 | tee "${RUN_DIR}/train.log"

  echo "DONE" > "${RUN_DIR}/DONE"
  echo "End: $(date)"
done

echo "============================================================"
echo "All seeds finished (10-epoch)."
echo "Job end: $(date)"
echo "============================================================"
