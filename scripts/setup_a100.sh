#!/bin/bash
# Setup for NVIDIA A100 (compute capability 8.0), both 40 GB and 80 GB.
# Same CUDA 12.1 / torch 2.3.1 toolchain as L40S.
set -euo pipefail
export GPU_LABEL="A100"
export GPU_ARCH="8.0"
export CUDA_MODULE="cuda/12.1.1"
export TORCH_SPEC="torch==2.3.1 torchvision==0.18.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
export NUMPY_GENERATION="1"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
