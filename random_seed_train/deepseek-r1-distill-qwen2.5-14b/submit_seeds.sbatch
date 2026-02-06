#!/usr/bin/env bash
#SBATCH --job-name=deepseek14b_seed
#SBATCH --output=deepseek14b_seed.%N.%j.out
#SBATCH --error=deepseek14b_seed.%N.%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --gres=gpu:a6000:1
#SBATCH --partition=gpu
#SBATCH --nodes=1
##SBATCH --time=100:00:00  # walltime is effectively unlimited on this cluster

set -euo pipefail

HERE="$(pwd)"
source "${HERE}/../common/seeds.sh"

source /TGM/Apps/ANACONDA/2024.10/etc/profile.d/conda.sh
conda activate llm

# Mirror the original DeepSeek PN env settings.
export MODEL_NAME_OR_PATH="/DATA/deepseek_r1_distilled_qwen3_14b/models/DeepSeek-R1-Distill-Qwen3-14B"
export PN_DATA_PATH="/DATA/npj_compt.mat_project_github/data/train_llm_pn.jsonl"
export MAX_SEQ_LEN=180
export PER_DEVICE_BATCH=64
export GRAD_ACCUM=20
export EPOCHS=10
export LR=2e-4
export LOGGING_STEPS=10

export FOCAL_GAMMA=2.0
export FOCAL_ALPHA_P=7.5
export FOCAL_ALPHA_N=1.0

export LORA_R=64
export LORA_ALPHA=128
export LORA_DROPOUT=0.0

export BF16=1
export OPTIM=adamw_torch_fused
export USE_GC=0

export SAVE_STRATEGY=epoch

export UNSLOTH_ENABLE_FLEX_ATTENTION=0
export TORCH_COMPILE=0
export TORCHDYNAMO_DISABLE=1
export UNSLOTH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/DATA/deepseek_r1_distilled_qwen3_14b/base_evaluation/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export OPTIM=adamw_torch_fused

echo "============================================================"
echo "Arch: DeepSeek-R1-Distill-Qwen-14B (multi-seed sequential)"
echo "Seeds: ${SEEDS[*]}"
echo "Epochs per seed: ${EPOCHS}"
echo "Host: $(hostname)"
echo "Job start: $(date)"
echo "============================================================"

for i in "${!SEEDS[@]}"; do
  IDX=$((i + 1))
  SEED="${SEEDS[$i]}"
  RUN_DIR="${HERE}/seed_${IDX}"
  mkdir -p "${RUN_DIR}"
  echo "${SEED}" > "${RUN_DIR}/seed.txt"

  if [[ -f "${RUN_DIR}/DONE" ]]; then
    echo "Seed ${IDX} (${SEED}) already DONE -> skipping."
    continue
  fi

  export OUTPUT_DIR="${RUN_DIR}"
  export SEED="${SEED}"

  cat > "${RUN_DIR}/run_config.txt" <<EOF
arch=deepseek-r1-distill-qwen-14b
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
train_data=${PN_DATA_PATH}
model_name_or_path=${MODEL_NAME_OR_PATH}
EOF

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${HERE}/train_deepseek14b_pn_focal_v1.py" > "${RUN_DIR}/train_script.sha256"
    sha256sum "${HERE}/../common/seeds.sh" > "${RUN_DIR}/seeds.sha256"
  fi

  echo "------------------------------------------------------------"
  echo "Seed ${IDX}/${#SEEDS[@]}: ${SEED}"
  echo "Run dir: ${RUN_DIR}"
  echo "Start: $(date)"
  echo "------------------------------------------------------------"

  python "${HERE}/train_deepseek14b_pn_focal_v1.py" 2>&1 | tee "${RUN_DIR}/train.log"

  echo "DONE" > "${RUN_DIR}/DONE"
  echo "End: $(date)"
done

echo "============================================================"
echo "All seeds finished."
echo "Job end: $(date)"
echo "============================================================"
