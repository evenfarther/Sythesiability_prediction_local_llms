#!/usr/bin/env bash
set -euo pipefail

# Use the same seed list as the pretrained runs; keep the first 3 for cost.
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
source "${SCRIPT_DIR}/../../random_seed_train/common/seeds.sh"

# Use first 3 seeds for random-init runs.
SEEDS=("${SEEDS[@]:0:3}")
