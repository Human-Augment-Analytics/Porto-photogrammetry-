#!/bin/bash
# Run COLMAP sparse reconstruction from images.
#
# Produces the same directory layout as the VGGT-based pipeline
# (src/pipeline/vggt_to_colmap.py), so the output can be used
# directly with 3D Gaussian Splatting and SuGaR.
#
# Output structure:
#   <output_path>/
#   ├── images/            # Undistorted images
#   └── sparse/
#       └── 0/
#           ├── cameras.bin
#           ├── images.bin
#           ├── points3D.bin
#           └── points3D.ply
#
# Usage:
#   bash scripts/run_colmap.sh --image_path /path/to/images --output_path /path/to/output
#
# All flags:
#   --image_path      Path to source images (required)
#   --output_path     Path for COLMAP output (required)
#   --camera_model    Camera model: PINHOLE, SIMPLE_PINHOLE, OPENCV, etc. (default: PINHOLE)
#   --single_camera   Use a shared camera for all images: 0 or 1 (default: 1)
#   --no_gpu          Disable GPU acceleration for SIFT
#   --matcher         Matching strategy: exhaustive, sequential, vocab_tree (default: exhaustive)
#   --colmap          Path to colmap binary (default: colmap)

set -e

# ============================================================
# Defaults
# ============================================================
CAMERA_MODEL="PINHOLE"
SINGLE_CAMERA=1
USE_GPU=1
MATCHER="exhaustive"
COLMAP_BIN="colmap"
IMAGE_PATH=""
OUTPUT_PATH=""

# ============================================================
# Parse arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image_path)   IMAGE_PATH="$2";   shift 2 ;;
        --output_path)  OUTPUT_PATH="$2";  shift 2 ;;
        --camera_model) CAMERA_MODEL="$2"; shift 2 ;;
        --single_camera) SINGLE_CAMERA="$2"; shift 2 ;;
        --no_gpu)       USE_GPU=0;         shift   ;;
        --matcher)      MATCHER="$2";      shift 2 ;;
        --colmap)       COLMAP_BIN="$2";   shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --image_path <path> --output_path <path> [options]"
            exit 1 ;;
    esac
done

if [ -z "$IMAGE_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "ERROR: --image_path and --output_path are required."
    echo "Usage: $0 --image_path <path> --output_path <path> [options]"
    exit 1
fi

if [ ! -d "$IMAGE_PATH" ]; then
    echo "ERROR: Image directory not found: $IMAGE_PATH"
    exit 1
fi

WORK_DIR="${OUTPUT_PATH}/distorted"
DATABASE="${WORK_DIR}/database.db"

echo "=========================================="
echo "COLMAP Sparse Reconstruction"
echo "=========================================="
echo "  Images:        $IMAGE_PATH"
echo "  Output:        $OUTPUT_PATH"
echo "  Camera model:  $CAMERA_MODEL"
echo "  Single camera: $SINGLE_CAMERA"
echo "  GPU:           $USE_GPU"
echo "  Matcher:       $MATCHER"
echo "  COLMAP:        $COLMAP_BIN"
echo ""

# ============================================================
# Step 1: Feature Extraction
# ============================================================
echo "=========================================="
echo "Step 1/6: Feature Extraction"
echo "=========================================="

mkdir -p "${WORK_DIR}/sparse"

$COLMAP_BIN feature_extractor \
    --database_path "$DATABASE" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.single_camera "$SINGLE_CAMERA" \
    --ImageReader.camera_model "$CAMERA_MODEL" \
    --SiftExtraction.use_gpu "$USE_GPU"

# ============================================================
# Step 2: Feature Matching
# ============================================================
echo ""
echo "=========================================="
echo "Step 2/6: Feature Matching ($MATCHER)"
echo "=========================================="

case "$MATCHER" in
    exhaustive)
        $COLMAP_BIN exhaustive_matcher \
            --database_path "$DATABASE" \
            --SiftMatching.use_gpu "$USE_GPU"
        ;;
    sequential)
        $COLMAP_BIN sequential_matcher \
            --database_path "$DATABASE" \
            --SiftMatching.use_gpu "$USE_GPU"
        ;;
    vocab_tree)
        $COLMAP_BIN vocab_tree_matcher \
            --database_path "$DATABASE" \
            --SiftMatching.use_gpu "$USE_GPU"
        ;;
    *)
        echo "ERROR: Unknown matcher: $MATCHER"
        exit 1 ;;
esac

# ============================================================
# Step 3: Sparse Reconstruction (Mapper)
# ============================================================
echo ""
echo "=========================================="
echo "Step 3/6: Sparse Reconstruction (Mapper)"
echo "=========================================="

$COLMAP_BIN mapper \
    --database_path "$DATABASE" \
    --image_path "$IMAGE_PATH" \
    --output_path "${WORK_DIR}/sparse" \
    --Mapper.ba_global_function_tolerance 0.000001

# ============================================================
# Step 4: Image Undistortion
# ============================================================
echo ""
echo "=========================================="
echo "Step 4/6: Image Undistortion"
echo "=========================================="

$COLMAP_BIN image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "${WORK_DIR}/sparse/0" \
    --output_path "$OUTPUT_PATH" \
    --output_type COLMAP

# ============================================================
# Step 5: Move sparse files into sparse/0/
# ============================================================
echo ""
echo "=========================================="
echo "Step 5/6: Reorganize sparse/0/"
echo "=========================================="

# image_undistorter writes directly into sparse/; 3DGS and SuGaR
# expect the reconstruction inside sparse/0/.
mkdir -p "${OUTPUT_PATH}/sparse/0"
for f in cameras.bin images.bin points3D.bin; do
    if [ -f "${OUTPUT_PATH}/sparse/$f" ]; then
        mv "${OUTPUT_PATH}/sparse/$f" "${OUTPUT_PATH}/sparse/0/"
    fi
done

echo "Done."

# ============================================================
# Step 6: Export PLY point cloud
# ============================================================
echo ""
echo "=========================================="
echo "Step 6/6: Export points3D.ply"
echo "=========================================="

$COLMAP_BIN model_converter \
    --input_path "${OUTPUT_PATH}/sparse/0" \
    --output_path "${OUTPUT_PATH}/sparse/0/points3D.ply" \
    --output_type PLY

# ============================================================
# Cleanup
# ============================================================
echo ""
echo "Removing intermediate files..."
rm -rf "$WORK_DIR"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "COLMAP reconstruction complete!"
echo "=========================================="
echo "  Output:   $OUTPUT_PATH"
echo "  Images:   $(ls "$OUTPUT_PATH/images" 2>/dev/null | wc -l) undistorted images"
echo "  Sparse:   $OUTPUT_PATH/sparse/0/"
echo ""
echo "Next steps:"
echo "  # Train 3D Gaussian Splatting:"
echo "  python gaussian-splatting/train.py -s $OUTPUT_PATH"
echo ""
echo "  # Or run full SuGaR pipeline:"
echo "  python SuGaR/train_full_pipeline.py -s $OUTPUT_PATH \\"
echo "      -r dn_consistency --high_poly True --export_obj True"
