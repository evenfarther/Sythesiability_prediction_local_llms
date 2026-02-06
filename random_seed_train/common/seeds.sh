#!/usr/bin/env bash
set -euo pipefail

# Base seed used in existing single-seed trainings (see original scripts: seed=3407).
BASE_SEED=3407

# Training seeds (must be identical across architectures).
# Currently: 3 seeds derived from BASE_SEED (cost control).
SEEDS=(
  $((BASE_SEED + 1))
  $((BASE_SEED + 2))
  $((BASE_SEED + 3))
)
