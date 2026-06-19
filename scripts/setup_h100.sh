#!/bin/bash
# Setup for NVIDIA H100 (compute capability 9.0). Same CUDA 12.1 / torch 2.3.1 toolchain.
set -euo pipefail
export GPU_LABEL="H100"
export GPU_ARCH="9.0"
export CUDA_MODULE="cuda/12.1.1"
export TORCH_SPEC="torch==2.3.1 torchvision==0.18.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
