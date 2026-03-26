# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Augenblick** is a photogrammetry pipeline that replaces classical COLMAP feature-matching with a feed-forward transformer (VGGT) and feeds its outputs into neural surface-reconstruction backends. Three export paths exist from a single VGGT inference run:

| Path | Entry point | Output |
|------|-------------|--------|
| GLB preview | `run_pipeline.py` | `.glb` point-cloud scene |
| COLMAP → 3DGS → SuGaR | `run_pipeline.py --save_for_sugar` + 3DGS + SuGaR | Textured mesh (`.obj`/`.ply`) |
| NeuS2 | `run_pipeline.py --save_for_neus2` | `transform.json` + watertight mesh |

## Environment Setup

```bash
conda env create -f environment.yml   # Python 3.10, PyTorch 2.6, CUDA 12.8
conda activate augenblick
cd src && pip install -e .            # installs the vggt package

# One-time setup for 3DGS + SuGaR (Blackwell/B200 GPUs):
bash scripts/setup_sugar_b200.sh
```

Required: `pycolmap==3.10.0` (pinned in `requirements.txt`). The VGGT model weights (~4 GB) are downloaded automatically from `facebook/VGGT-1B` on HuggingFace on first run.

## Running the Pipeline

### VGGT inference (main entry point)
```bash
# Input must have an images/ subdirectory; masks/ is optional
python src/pipeline/run_pipeline.py <input_dir> \
    [--output_dir <dir>] \
    [--use_masks] \
    [--save_for_sugar]      # writes COLMAP binary to <output_dir>/colmap/
    [--save_for_neus2]      # writes transform.json pickle
    [--save_ply]            # writes point cloud .ply
    [--skip_glb]            # skip GLB generation (faster)
    [--conf_thres 50]       # confidence percentile for GLB (0-100)
    [--point_conf_threshold 70]  # confidence percentile for COLMAP/PLY export
    [--max_points3d 200000]
    [--load_mode {crop,pad}]
    [--prediction_mode "Depthmap and Camera Branch"]
```

### VGGT + COLMAP BA (alternative entry point)
```bash
python src/pipeline/demo_colmap.py \
    --input_dir <dir>   \   # must contain images/ subdirectory
    --output_dir <dir>  \
    [--use_masks]       \
    [--use_ba]          \   # enables VGGSfM tracker + bundle adjustment
    [--camera_type SIMPLE_PINHOLE]
```

### Full VGGT → 3DGS → SuGaR pipeline
```bash
# Local
python src/pipeline/run_sugar_pipeline.py <input_dir> \
    [--quality {draft,medium,high}]   # 7k/15k/30k 3DGS iterations

# HPC (SLURM, B200 node)
sbatch scripts/run_vggt_sugar.sbatch
```

### Classical COLMAP only
```bash
bash scripts/run_colmap.sh \
    --image_path <images_dir> \
    --output_path <output_dir> \
    [--camera_model PINHOLE] \
    [--single_camera] \
    [--matcher {exhaustive,sequential,vocab_tree}] \
    [--no_gpu]
```

## Architecture

### Data flow

```
Input images/
    │
    ▼
load_and_preprocess_images_square()   ← src/vggt/utils/load_fn.py
  - Square-pad to max(W,H), resize to 1024 (load) then 518 (VGGT input)
  - Optionally composite masks (zero out background)
  - Returns: images [N,3,1024,1024], original_coords [N,6], masks list
    │
    ▼
VGGT model (facebook/VGGT-1B)        ← src/vggt/models/vggt.py
  - Aggregator: alternating frame-blocks + global-blocks, 2D RoPE
  - camera_head → pose_enc [B,S,9]  (translation + quaternion + FoV)
  - depth_head  → depth_map, depth_conf [B,S,H,W]
  - pointmap_head → world_points [B,S,H,W,3]  (optional)
  - track_head  → tracks (used only with --use_ba)
    │
    ▼
pose_encoding_to_extri_intri()        ← src/vggt/utils/pose_enc.py
  - Converts 9-D encoding → extrinsic [3×4] + intrinsic [3×3]
  - Intrinsics are in 518-pixel space at this point
    │
    ├──► vggt_to_colmap.py           ← src/pipeline/
    │     - Rescales intrinsics to original image resolution
    │     - Confidence-filters and spatially subsamples world points
    │     - Writes cameras.bin / images.bin / points3D.bin
    │
    ├──► vggt_to_neus2_converter.py  ← src/pipeline/
    │     - OpenCV → NeRF coords (180° x-rotation)
    │     - Computes scene scale/offset from point-cloud bbox
    │     - Writes transform.json
    │
    └──► predictions_to_glb()        ← src/visual_util.py
          - Confidence-filters points, optionally runs sky segmentation (ONNX)
          - Exports trimesh scene as .glb
```

### Coordinate system conventions

- **VGGT outputs**: OpenCV convention (x-right, y-down, z-forward), camera-from-world (`[R|t]`)
- **Intrinsics from `pose_encoding_to_extri_intri`**: initially in 518×518 pixel space; rescaled to original image space by `rename_colmap_recons_and_rescale_camera` (in `demo_colmap.py`) or by `vggt_to_colmap.py`
- **NeuS2 export**: applies a 180° x-rotation to convert to NeRF convention (y-up, z-backward)
- **COLMAP format**: 1-indexed image/camera/point IDs (there is a +1 offset between batch index and COLMAP ID throughout `np_to_pycolmap.py`)

### Input directory structure

```
<scene>/
├── images/     ← required; all image files go here
└── masks/      ← optional; PNG masks named <image_stem>.png
                   White = foreground, black = background
```

### Key source files

| File | Role |
|------|------|
| `src/pipeline/run_pipeline.py` | Main CLI; VGGT inference + export orchestration |
| `src/pipeline/demo_colmap.py` | Alternative CLI; adds optional VGGSfM bundle adjustment |
| `src/pipeline/run_sugar_pipeline.py` | End-to-end VGGT → 3DGS → SuGaR orchestrator |
| `src/pipeline/vggt_to_colmap.py` | VGGT predictions → COLMAP binary format |
| `src/pipeline/vggt_to_neus2_converter.py` | VGGT predictions → NeuS2 `transform.json` |
| `src/vggt/models/vggt.py` | VGGT model class |
| `src/vggt/utils/load_fn.py` | Image loading and square-pad preprocessing |
| `src/vggt/utils/pose_enc.py` | Pose encoding ↔ extrinsic/intrinsic conversion |
| `src/vggt/utils/geometry.py` | Depth unprojection, SE(3) inverse |
| `src/vggt/dependency/np_to_pycolmap.py` | NumPy arrays → pycolmap Reconstruction objects |
| `src/visual_util.py` | GLB export, sky segmentation (ONNX) |

## Submodules

`src/light_glue/`, `src/pytorch3d/`, `src/sugar/` are git submodules. After cloning, run:
```bash
git submodule update --init --recursive
```

## GPU Notes

- Mixed precision: bfloat16 on Ampere+ (SM ≥ 8.0), float16 otherwise
- Blackwell (B200, SM ≥ 10.0): `torch.compile(mode="max-autotune")` is applied automatically; the SLURM script targets `sm_100` architecture
- Crash dumps (`core.colmap-*.ufhpc.*`) in the repo root are artifacts from HPC runs and can be ignored/deleted
