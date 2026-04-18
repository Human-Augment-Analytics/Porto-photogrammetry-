#!/bin/bash
# Run COLMAP sparse reconstruction from images.
#
# Produces directory layout that can be used
# directly with 3D Gaussian Splatting, SuGaR, and 2DGS.
#
# Output structure:
#   <output_path>/
#   ├── images/            # Undistorted images
#   ├── masks/             # Image masks
#   └── sparse/
#       └── 0/
#           ├── cameras.bin
#           ├── images.bin
#           ├── points3D.bin
#           └── points3D.ply
#
# Usage:
#   bash scripts/run_colmap.sh --input_path /path/to/dataset --output_path /path/to/output
#
# All flags:
#   --input_path      Path to source dataset (required)
#   --output_path     Path for COLMAP output (required)
#   --camera_model    Camera model: SIMPLE_RADIAL, PINHOLE, OPENCV, etc. (default: PINHOLE)
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

print_usage() {
    echo "Usage: $0 --image_path <path> --output_path <path> [options]"
    echo ""
    echo "Options:"
    echo "  --camera_model <model>      Camera model (default: PINHOLE)"
    echo "  --single_camera <0|1>       Share intrinsics across all images (default: 1)"
    echo "  --no_gpu                    Disable GPU acceleration for SIFT"
    echo "  --matcher <name>            exhaustive, sequential, vocab_tree (default: exhaustive)"
    echo "  --colmap <path>             Path to the COLMAP binary (default: colmap)"
}

# ============================================================
# Parse arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)      print_usage;       exit 0 ;;
        --input_path)   INPUT_PATH="$2";   shift 2 ;;
        --output_path)  OUTPUT_PATH="$2";  shift 2 ;;
        --camera_model) CAMERA_MODEL="$2"; shift 2 ;;
        --single_camera) SINGLE_CAMERA="$2"; shift 2 ;;
        --no_gpu)       USE_GPU=0;         shift   ;;
        --matcher)      MATCHER="$2";      shift 2 ;;
        --colmap)       COLMAP_BIN="$2";   shift 2 ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1 ;;
    esac
done

if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "ERROR: --image_path and --output_path are required."
    print_usage
    exit 1
fi

IMAGE_PATH="${INPUT_PATH}/images"
MASK_PATH="${INPUT_PATH}/masks"
if [ ! -d "$IMAGE_PATH" ]; then
    echo "ERROR: Image directory not found: $IMAGE_PATH"
    exit 1
fi

WORK_DIR="${OUTPUT_PATH}/distorted"
DATABASE="${WORK_DIR}/database.db"
MODEL_PATH="${WORK_DIR}/sparse/0"

echo "=========================================="
echo "COLMAP Sparse Reconstruction"
echo "=========================================="
echo "  Images:        $IMAGE_PATH"
echo "  Masks:         $MASK_PATH"
echo "  Output:        $OUTPUT_PATH"
echo "  Camera model:  $CAMERA_MODEL"
echo "  Single camera: $SINGLE_CAMERA"
echo "  GPU:           $USE_GPU"
echo "  Matcher:       $MATCHER"
echo "  COLMAP:        $COLMAP_BIN"
echo ""

PIPELINE_START=$(date +%s)

# ============================================================
# Step 1: Feature Extraction
# ============================================================
echo "=========================================="
echo "Step 1/6: Feature Extraction"
echo "=========================================="
STEP_START=$(date +%s)

mkdir -p "${WORK_DIR}/sparse"

$COLMAP_BIN feature_extractor \
    --database_path "$DATABASE" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.single_camera "$SINGLE_CAMERA" \
    --ImageReader.camera_model "$CAMERA_MODEL" \
    --SiftExtraction.use_gpu "$USE_GPU" \
    --SiftExtraction.num_threads 4

FEATURE_EXTRACT_TIME=$(( $(date +%s) - STEP_START ))
echo "Feature extraction completed in ${FEATURE_EXTRACT_TIME}s"

# ============================================================
# Step 2: Feature Matching
# ============================================================
echo ""
echo "=========================================="
echo "Step 2/6: Feature Matching ($MATCHER)"
echo "=========================================="
STEP_START=$(date +%s)

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

MATCHING_TIME=$(( $(date +%s) - STEP_START ))
echo "Feature matching completed in ${MATCHING_TIME}s"

# ============================================================
# Step 3: Sparse Reconstruction (Mapper)
# ============================================================
echo ""
echo "=========================================="
echo "Step 3/6: Sparse Reconstruction (Mapper)"
echo "=========================================="
STEP_START=$(date +%s)

$COLMAP_BIN mapper \
    --database_path "$DATABASE" \
    --image_path "$IMAGE_PATH" \
    --output_path "${WORK_DIR}/sparse" \
    --Mapper.ba_global_function_tolerance 0.000001

MAPPER_TIME=$(( $(date +%s) - STEP_START ))
echo "Mapping completed in ${MAPPER_TIME}s"

# ============================================================
# Step 4: Image Undistortion
# ============================================================
echo ""
echo "=========================================="
echo "Step 4/6: Image Undistortion"
echo "=========================================="
STEP_START=$(date +%s)

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Refined sparse model not found at $MODEL_PATH before image undistortion"
    exit 1
fi

$COLMAP_BIN image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --output_type COLMAP

UNDISTORT_TIME=$(( $(date +%s) - STEP_START ))
echo "Image undistortion completed in ${UNDISTORT_TIME}s"

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

# ============================================================
# Step 6: Copy masks (if any)
# ============================================================
echo ""
echo "=========================================="
echo "Step 6/6: Copy masks from input directory (if any)"
echo "=========================================="

if [ -d "$MASK_PATH" ]; then
    mkdir -p "$OUTPUT_PATH/masks"
    cp "$MASK_PATH/"* "$OUTPUT_PATH/masks/"
fi

echo "Done."

# ============================================================
# Cleanup
# ============================================================
echo ""
echo "Removing intermediate files..."
rm -rf "$WORK_DIR"

# ============================================================
# Summary
# ============================================================
TOTAL_TIME=$(( $(date +%s) - PIPELINE_START ))
echo ""
echo "=========================================="
echo "COLMAP reconstruction complete!"
echo "=========================================="
echo "  Output:            $OUTPUT_PATH"
echo "  Images:            $(ls "$OUTPUT_PATH/images" 2>/dev/null | wc -l) undistorted images"
echo "  Sparse:            $OUTPUT_PATH/sparse/0/"
echo "  Feature extract:   ${FEATURE_EXTRACT_TIME}s"
echo "  Feature matching:  ${MATCHING_TIME}s"
echo "  Mapper:            ${MAPPER_TIME}s"
echo "  Undistortion:      ${UNDISTORT_TIME}s"
echo "  Total:             ${TOTAL_TIME}s"
echo ""
