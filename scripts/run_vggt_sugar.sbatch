#!/bin/bash
#SBATCH --job-name=vggt-sugar
#SBATCH --output=vggt_sugar_%j.log
#SBATCH --error=vggt_sugar_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --gres=gpu:b200:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

# ============================================================
# VGGT -> SuGaR Pipeline (B200)
# ============================================================
#
# Usage:
#   sbatch scripts/run_vggt_sugar.sbatch
#
# If your cluster uses a different partition for B200s, update
# the --partition and --gres lines above accordingly.
# ============================================================

set -e

# --- Paths ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="/blue/arthur.porto-biocosmos/data"
IMAGES_DIR="${DATA_DIR}/images"
OUTPUT_DIR="${DATA_DIR}/sugar_output"

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Project:      $PROJECT_DIR"
echo "Images:       $IMAGES_DIR"
echo "Output:       $OUTPUT_DIR"
echo ""

# Verify images exist
if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found: $IMAGES_DIR"
    exit 1
fi
NUM_IMAGES=$(ls "$IMAGES_DIR" | wc -l)
echo "Found $NUM_IMAGES images"

# --- Environment ---
module load conda 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true

conda activate augenblick

# B200 CUDA arch for any JIT compilation
export TORCH_CUDA_ARCH_LIST="10.0"
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Cache HuggingFace models to blue storage (avoids re-download)
export TORCH_HOME="${DATA_DIR}/.cache/torch"
export HF_HOME="${DATA_DIR}/.cache/huggingface"
mkdir -p "$TORCH_HOME" "$HF_HOME"

echo ""
echo "Python:  $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo ""

# ============================================================
# Stage 1: VGGT inference + COLMAP export
# ============================================================
echo "=========================================="
echo "Stage 1: VGGT -> COLMAP"
echo "=========================================="

python "${PROJECT_DIR}/src/pipeline/run_pipeline.py" \
    "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --save_for_sugar \
    --skip_glb \
    --max_points3d 200000 \
    --point_conf_threshold 70.0

COLMAP_DIR="${OUTPUT_DIR}/colmap"

# ============================================================
# Stage 2: 3DGS training
# ============================================================
GS_REPO="${PROJECT_DIR}/gaussian-splatting"
GS_OUTPUT="${OUTPUT_DIR}/gs_model"

if [ -d "$GS_REPO" ]; then
    echo ""
    echo "=========================================="
    echo "Stage 2: 3D Gaussian Splatting"
    echo "=========================================="

    python "${GS_REPO}/train.py" \
        -s "$COLMAP_DIR" \
        -m "$GS_OUTPUT" \
        --iterations 15000
else
    echo ""
    echo "Skipping 3DGS: repo not found at $GS_REPO"
    echo "Run scripts/setup_sugar_b200.sh first, then resubmit."
fi

# ============================================================
# Stage 3: SuGaR mesh extraction
# ============================================================
SUGAR_REPO="${PROJECT_DIR}/SuGaR"
SUGAR_OUTPUT="${OUTPUT_DIR}/sugar_mesh"

if [ -d "$SUGAR_REPO" ] && [ -d "$GS_OUTPUT" ]; then
    echo ""
    echo "=========================================="
    echo "Stage 3: SuGaR Mesh Extraction"
    echo "=========================================="

    python "${SUGAR_REPO}/train_full_pipeline.py" \
        -s "$COLMAP_DIR" \
        -c "$GS_OUTPUT" \
        -o "$SUGAR_OUTPUT" \
        -r dn_consistency \
        --high_poly True \
        --refinement_time medium \
        --export_obj True
else
    echo ""
    echo "Skipping SuGaR: repo not found at $SUGAR_REPO or no 3DGS checkpoint"
    echo "Run scripts/setup_sugar_b200.sh first, then resubmit."
fi

# ============================================================
# Done
# ============================================================
echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "  COLMAP:     $COLMAP_DIR"
echo "  3DGS:       $GS_OUTPUT"
echo "  SuGaR mesh: $SUGAR_OUTPUT"
echo "  Job time:   $SECONDS seconds"
