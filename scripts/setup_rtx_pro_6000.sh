#!/bin/bash
# Setup for NVIDIA RTX Pro 6000 Blackwell (compute capability 12.0).
# CUDA 12.8+ toolkit / torch 2.9.1 (cu130 wheels). Not validated here (no RTX Pro 6000).
# Override CUDA_MODULE if your site uses a different module name.
set -euo pipefail
export GPU_LABEL="RTX-Pro-6000-Blackwell"
export GPU_ARCH="12.0"
export CUDA_MODULE="${CUDA_MODULE:-cuda/13.0.2}"
export TORCH_SPEC="torch==2.9.1 torchvision==0.24.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
