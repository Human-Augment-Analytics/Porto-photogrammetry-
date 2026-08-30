#!/bin/bash
# Setup for NVIDIA B200 (compute capability 10.0) — the repo's original target.
# Requires CUDA 12.8+; torch versions follow the README. Not validated here (no B200).
# Override CUDA_MODULE if your site uses a different name.
set -euo pipefail
export GPU_LABEL="B200"
export GPU_ARCH="10.0"
export CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
export TORCH_SPEC="torch==2.9.1 torchvision==0.24.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
export NUMPY_GENERATION="2"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
