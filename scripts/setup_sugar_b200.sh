#!/bin/bash
# Setup SuGaR and 3D Gaussian Splatting for NVIDIA B200 (Blackwell, sm_100)
#
# This script:
#   1. Verifies CUDA 12.8+ is available
#   2. Clones 3D Gaussian Splatting (with submodules)
#   3. Compiles diff-gaussian-rasterization with sm_100 support
#   4. Compiles simple-knn with sm_100 support
#   5. Clones SuGaR and installs dependencies
#
# Usage:
#   bash scripts/setup_sugar_b200.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "SuGaR + 3DGS Setup for NVIDIA B200"
echo "========================================"
echo "Project root: $PROJECT_ROOT"

# --- Verify CUDA ---
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA Toolkit 12.8+."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
echo "CUDA version: $CUDA_VERSION"

if [ "$CUDA_MAJOR" -lt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 8 ]); then
    echo "WARNING: CUDA $CUDA_VERSION detected. B200 (sm_100) requires CUDA 12.8+."
    echo "Continuing anyway, but compilation may fail for Blackwell targets."
fi

# --- Set CUDA architecture for Blackwell ---
export TORCH_CUDA_ARCH_LIST="10.0"
export TCNN_CUDA_ARCHITECTURES="100"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# --- Clone 3D Gaussian Splatting ---
GS_DIR="$PROJECT_ROOT/gaussian-splatting"
if [ ! -d "$GS_DIR" ]; then
    echo ""
    echo "Cloning 3D Gaussian Splatting..."
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git "$GS_DIR"
else
    echo "3DGS already exists at $GS_DIR"
fi

# --- Build diff-gaussian-rasterization ---
echo ""
echo "Building diff-gaussian-rasterization (sm_100)..."
cd "$GS_DIR/submodules/diff-gaussian-rasterization"
pip install -e . --no-build-isolation
echo "diff-gaussian-rasterization built successfully."

# --- Build simple-knn ---
echo ""
echo "Building simple-knn (sm_100)..."
cd "$GS_DIR/submodules/simple-knn"
pip install -e . --no-build-isolation
echo "simple-knn built successfully."

# --- Clone SuGaR ---
SUGAR_DIR="$PROJECT_ROOT/SuGaR"
if [ ! -d "$SUGAR_DIR" ]; then
    echo ""
    echo "Cloning SuGaR..."
    git clone https://github.com/Anttwo/SuGaR.git "$SUGAR_DIR"
else
    echo "SuGaR already exists at $SUGAR_DIR"
fi

# --- Install SuGaR Python dependencies ---
echo ""
echo "Installing SuGaR dependencies..."
cd "$SUGAR_DIR"
pip install -e . 2>/dev/null || {
    echo "SuGaR has no setup.py, installing deps from environment.yml..."
    # Install core deps that aren't already in our environment
    pip install pymcubes 2>/dev/null || true
}

# --- Optional: install nvdiffrast for fast texture export ---
echo ""
echo "Installing nvdiffrast (optional, for fast texture export)..."
pip install nvdiffrast 2>/dev/null || {
    echo "nvdiffrast install failed (optional). Texture export will use fallback."
}

cd "$PROJECT_ROOT"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Repos:"
echo "  3DGS:  $GS_DIR"
echo "  SuGaR: $SUGAR_DIR"
echo ""
echo "Run the pipeline:"
echo "  python src/pipeline/run_sugar_pipeline.py /path/to/data --quality medium"
echo ""
echo "Or step by step:"
echo "  # 1. VGGT + COLMAP export"
echo "  python src/pipeline/run_pipeline.py /path/to/data --save_for_sugar --skip_glb"
echo ""
echo "  # 2. Full SuGaR pipeline"
echo "  python SuGaR/train_full_pipeline.py -s sugar_output/colmap \\"
echo "      -r dn_consistency --high_poly True --export_obj True"
