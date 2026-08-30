#!/bin/bash
# Setup for NVIDIA L40S (compute capability 8.9). Validated combo.
set -euo pipefail
export GPU_LABEL="L40S"
export GPU_ARCH="8.9"
export CUDA_MODULE="cuda/12.1.1"
export TORCH_SPEC="torch==2.3.1 torchvision==0.18.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
export NUMPY_GENERATION="1"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
