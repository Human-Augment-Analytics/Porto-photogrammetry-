# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Augenblick** is a photogrammetry pipeline for producing 3D meshes from multi-view images. It replaces classical COLMAP feature-matching with a feed-forward transformer (VGGT) and feeds its outputs into several neural surface-reconstruction backends. The project supports two SfM paths (VGGT, classical COLMAP) and three Gaussian-splatting-based reconstruction backends (SuGaR, PGSR, 2DGS), all of which consume COLMAP-format sparse reconstructions.

| Pipeline stage | Options |
|----------------|---------|
| Data preparation | `pipeline/preparation/prepare_uf_dataset.py` |
| SfM | VGGT (`pipeline/sfm/run_vggt_to_colmap.py`) or COLMAP (`pipeline/sfm/run_colmap.sh`) |
| Reconstruction | SuGaR (`pipeline/reconstruction/run_sugar.py`), 2DGS (`pipeline/reconstruction/run_2dgs.py`), PGSR (`pipeline/reconstruction/run_pgsr.py`) |
| Baseline | Meshroom (`baseline/benchmark_meshroom.py`) |

## Environment Setup

```bash
conda env create -f environment.yml   # Python 3.10, PyTorch 2.6, CUDA 12.8
conda activate augenblick
cd src/vggt && pip install -e .       # installs the vggt package
```

> **Note:** The vggt Python package lives at `src/vggt/vggt/` (i.e. `src/vggt/` is the project root containing `pyproject.toml`, and the importable package is the inner `vggt/` directory). If you have an old editable install from `cd src && pip install -e .`, reinstall via `cd src/vggt && pip install -e .`.

## Submodules

`src/light_glue/`, `src/pytorch3d/`, `src/sugar/` are git submodules. After cloning:
```bash
git submodule update --init --recursive
```

## GPU Notes

- Mixed precision: bfloat16 on Ampere+ (SM >= 8.0), float16 otherwise
- Blackwell (B200, SM >= 10.0): `torch.compile(mode="max-autotune")` is applied automatically
- Crash dumps (`core.colmap-*.ufhpc.*`) in the repo root are HPC artifacts and can be ignored/deleted

---

## pipeline/ -- Orchestration Scripts

The `pipeline/` directory contains the canonical entry points for dataset preparation, structure-from-motion, and reconstruction. These are the scripts that should be used for new experiments.

### Data Preparation

```bash
python pipeline/preparation/prepare_uf_dataset.py <input_dir> \
    [--out <output_dir>] \
    [--mode {copy,move,symlink}] \
    [--include-unmatched]
```

Separates a flat directory of mixed images and masks (naming convention: `<stem>.JPG` + `<stem>.jpg.mask.png`) into the standard `images/` + `masks/` structure. Images are normalised to `.jpg`, masks to `.png` named after the image stem. Supports copy, move, and symlink modes.

### VGGT SfM (VGGT to COLMAP)

```bash
python pipeline/sfm/run_vggt_to_colmap.py \
    --input_dir <dir>       \   # must contain images/ subdirectory
    --output_dir <dir>      \
    [--use_masks]           \   # apply masks from masks/ to depth confidence
    [--use_ba]              \   # enable VGGSfM tracker + bundle adjustment
    [--camera_type SIMPLE_PINHOLE] \
    [--conf_thres_value 5.0]    # depth confidence threshold (no-BA mode)
```

Runs VGGT inference, converts predictions to a COLMAP sparse reconstruction (`sparse/0/`), copies images to the output directory, and exports a `points.ply` visualization. Masks are always copied to `<output>/masks/` when they exist in `<input>/masks/`, regardless of `--use_masks` (the flag only controls whether masks weight the depth confidence during reconstruction). The output directory is ready to be consumed by any of the reconstruction backends. Logs per-stage runtimes (model load, VGGT inference, optional tracking + BA) and a total-time summary.

**Two modes:**
- **Without BA (default):** Uses VGGT's depth maps and camera predictions directly. Filters 3D points by `conf_thres_value`, randomly subsamples to 100k points, writes COLMAP with PINHOLE camera model at 518px resolution, then rescales to original resolution.
- **With BA (`--use_ba`):** Additionally runs VGGSfM tracker for correspondence prediction, then runs `pycolmap.bundle_adjustment()`. Operates at 1024px resolution internally, supports SIMPLE_PINHOLE and shared camera modes.

### Classical COLMAP SfM

```bash
bash pipeline/sfm/run_colmap.sh \
    --input_path <dataset_dir> \
    --output_path <output_dir> \
    [--camera_model PINHOLE] \
    [--single_camera 1] \
    [--matcher {exhaustive,sequential,vocab_tree}] \
    [--no_gpu]
```

`--input_path` is a dataset directory containing an `images/` subdirectory (and optionally `masks/`); the script derives `IMAGE_PATH=<input>/images` and `MASK_PATH=<input>/masks` internally. Runs COLMAP feature extraction, matching, sparse reconstruction, and image undistortion. Outputs undistorted images in `images/` and sparse reconstruction in `sparse/0/`. If `<input>/masks/` exists, it is copied to `<output>/masks/`. The intermediate `distorted/` working directory is cleaned up automatically. Prints per-step and total runtimes (via `date +%s`) in the final summary.

### Reconstruction Scripts

Wrapper scripts that orchestrate training and mesh extraction for each backend. All accept a COLMAP scene directory and an output directory.

```bash
# SuGaR: vanilla 3DGS training + coarse → mesh → refine → textured mesh
python pipeline/reconstruction/run_sugar.py <scene_dir> <output_dir> \
    [--gs_iterations 20000] [--iteration_to_load 7000] \
    [--regularization dn_consistency] [--high_poly] [--refinement_time long]

# 2DGS: training + TSDF mesh extraction
python pipeline/reconstruction/run_2dgs.py <scene_dir> <output_dir> \
    [--iterations 30000] [--voxel_size -1.0] [--depth_trunc -1.0] \
    [--num_cluster 50] [--unbounded]

# PGSR: copies scene, flattens sparse/0/ → sparse/, trains, TSDF mesh extraction
python pipeline/reconstruction/run_pgsr.py <scene_dir> <output_dir> \
    [--iterations 30000] [--max_depth 10.0] [--voxel_size 0.001] \
    [--max_abs_split_points 0] [--opacity_cull_threshold 0.05]
```

Each script runs the underlying `src/` training and rendering scripts via `subprocess.run()` with `cwd` set to the backend's source directory. PGSR's `run_pgsr.py` handles the sparse directory flattening automatically (PGSR expects `sparse/` not `sparse/0/`).

---

## baseline/ -- Baseline Wrappers

Thin wrappers around third-party photogrammetry tools used as qualitative comparisons. They match the logging style of `pipeline/reconstruction/run_2dgs.py` (banner, `subprocess.run`, total-time summary).

### Meshroom (AliceVision)

```bash
python baseline/benchmark_meshroom.py <input_images> <output_dir> \
    [--save_file <path.mg>] \
    [--meshroom_root <path>]
```

Resolves `meshroom_batch` from `$MESHROOM_ROOT` (or `--meshroom_root`), invokes the hardcoded `photogrammetry` pipeline template, and logs total runtime. Prepends `$MESHROOM_ROOT` to `PYTHONPATH` before invoking `meshroom_batch` so its Python modules resolve. Installation/env-var setup is documented in `meshroom-setup.md` (the setup doc includes a dedicated `meshroom` conda env for the batch CLI).

---

## src/vggt/ -- VGGT Model

Meta's Visual Geometry Grounded Transformer. Weights (~4 GB) auto-download from `facebook/VGGT-1B` on HuggingFace.

### Model Architecture (`src/vggt/vggt/models/`)

**`vggt.py` -- VGGT class:**
- Input: images `[B, S, 3, H, W]` in `[0, 1]`, optional query points `[B, N, 2]`
- Aggregator produces token lists via alternating frame/global attention
- Four prediction heads, each consuming the aggregated tokens:
  - `camera_head` -> `pose_enc [B, S, 9]` (translation[3] + quaternion[4] + FoV[2])
  - `depth_head` -> `depth [B, S, H, W, 1]`, `depth_conf [B, S, H, W]`
  - `point_head` -> `world_points [B, S, H, W, 3]`, `world_points_conf [B, S, H, W]`
  - `track_head` -> `track [B, S, N, 2]`, `vis [B, S, N]`, `conf [B, S, N]` (only if query_points given)

**`aggregator.py` -- Aggregator class:**
- DINOv2 ViT-L/14 patch embedding (frozen weights from `dinov2_vitl14_reg`)
- 24 alternating blocks: frame attention (per-frame `[B*S, P, C]`) + global attention (across frames `[B, S*P, C]`)
- 2D RoPE positional encoding (frequency=100)
- Special tokens: 1 camera token + 4 register tokens prepended to each frame's patch tokens
- Outputs concatenated frame+global intermediates `[B, S, P, 2*C]` (2048-dim) for each block pair
- Gradient checkpointing enabled during training

### Utilities (`src/vggt/vggt/utils/`)

**`load_fn.py`:**
- `load_and_preprocess_images_square()`: Main loader for the pipeline. Square-pads to `max(W,H)`, resizes to `target_size` (default 1024). Returns images `[N, 3, target_size, target_size]`, `original_coords [N, 6]` (x1, y1, x2, y2, W, H for undo-padding), and transformed masks. Optionally composites masks onto images (zeroes background).
- `load_and_preprocess_images()`: Simpler loader with `crop` and `pad` modes at 518px. Used by the model's quick-start demo.

**`pose_enc.py`:**
- `pose_encoding_to_extri_intri()`: Decodes `[B, S, 9]` -> extrinsic `[B, S, 3, 4]` + intrinsic `[B, S, 3, 3]`. Intrinsics are in the resolution of `image_size_hw` (typically 518x518). Principal point assumed at image center.
- `extri_intri_to_pose_encoding()`: Inverse operation.

**`geometry.py`:**
- `unproject_depth_map_to_point_map()`: Depth `[S, H, W]` + extrinsics + intrinsics -> world points `[S, H, W, 3]`
- `closed_form_inverse_se3()`: Batch SE(3) inverse (works on both numpy and torch)

**`helper.py`:**
- `randomly_limit_trues()`: Subsamples True entries in a boolean mask to a budget
- `create_pixel_coordinate_grid()`: Creates `[S, H, W, 3]` grid of (x, y, frame_idx)

### COLMAP Conversion (`src/vggt/vggt/dependency/np_to_pycolmap.py`)

- `batch_np_matrix_to_pycolmap()`: Full conversion with tracks, used with BA. Applies reprojection error filtering, builds pycolmap Reconstruction with proper Point2D<->Point3D associations.
- `batch_np_matrix_to_pycolmap_wo_track()`: Lightweight conversion without tracks, used for feed-forward mode. Points are assigned to the frame they were unprojected from. **Do NOT use this for BA.**
- `pycolmap_to_batch_np_matrix()`: Inverse (reconstruction -> numpy arrays)

**COLMAP ID convention:** All IDs are 1-indexed. There is a `+1` offset between batch index and COLMAP image/camera ID throughout.

### Coordinate System Conventions

- **VGGT outputs**: OpenCV convention (x-right, y-down, z-forward), camera-from-world `[R|t]`
- **Intrinsics from `pose_encoding_to_extri_intri`**: initially in 518x518 pixel space; `run_vggt_to_colmap.py` rescales them to original image resolution via `rename_colmap_recons_and_rescale_camera()`
- **NeuS2 export**: applies 180deg x-rotation to convert to NeRF convention (y-up, z-backward)
- **COLMAP format**: 1-indexed image/camera/point IDs

---

## src/sugar/ -- SuGaR (Surface-Aligned Gaussians)

A git submodule. Extracts textured triangle meshes from 3D Gaussian Splatting by regularising Gaussians to align with an implicit surface, then binding them to an explicit mesh.

### Scene Input Format

All SuGaR scripts expect a COLMAP-format scene directory:
```
<scene_path>/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

SuGaR also requires a pre-trained vanilla 3DGS checkpoint (from `src/sugar/gaussian_splatting/train.py`).

### Training Pipeline (`src/sugar/train.py`)

Run from `src/sugar/`:
```bash
python train.py \
    -s <scene_path>          \
    -c <3dgs_checkpoint>     \
    [-i <iteration_to_load>] \   # default: 7000
    [-r {sdf,density,dn_consistency}] \  # regularization type
    [-o <output_path>]       \   # organises coarse/coarse_mesh/refined/refined_mesh
    [-v <n_vertices>]        \   # default: 1M
    [-g <gaussians_per_tri>] \   # default: 1
    [-f <refinement_iters>]  \   # default: 15000
    [--low_poly]             \   # 200k vertices, 6 gaussians/triangle
    [--high_poly]            \   # 1M vertices, 1 gaussian/triangle
    [--refinement_time {short,medium,long}]  # 2k/7k/15k iterations
```

**Four sequential stages:**
1. **Coarse SuGaR training** (`sugar_trainers/`): Loads a 3DGS checkpoint and trains with surface regularization (`sdf`, `density`, or `dn_consistency`). `dn_consistency` is recommended for best mesh quality.
2. **Coarse mesh extraction** (`sugar_extractors/coarse_mesh.py`): Extracts a triangle mesh at the given surface level (default 0.3), decimates to target vertex count.
3. **Refined SuGaR training** (`sugar_trainers/refine.py`): Binds Gaussians to the mesh and refines with normal consistency loss.
4. **Textured mesh extraction** (`sugar_extractors/refined_mesh.py`): UV-unwraps the refined mesh and bakes a texture atlas (`.obj` output).

### Mesh Extraction (Standalone)

```bash
# Coarse mesh from a trained coarse SuGaR model
python extract_mesh.py -s <scene> -c <3dgs_ckpt> -m <coarse_model_path> \
    [-l <surface_level>] [-d <decimation_target>]

# Textured mesh from a refined SuGaR model
python extract_refined_mesh_with_texture.py -s <scene> -c <3dgs_ckpt> \
    -m <refined_model_path> [--coarse_mesh_dir <dir>] [-o <output_dir>]
```

### Key Components

| Directory | Role |
|-----------|------|
| `gaussian_splatting/` | Embedded vanilla 3DGS (train, render, scene loading). The 3DGS checkpoint is the starting point for SuGaR. |
| `gsplat_wrapper/` | Alternative rasterization backend using gsplat instead of diff-gaussian-rasterization |
| `sugar_trainers/` | Coarse training (density, SDF, DN consistency regularization) and refinement |
| `sugar_extractors/` | Mesh extraction from coarse and refined SuGaR models |
| `sugar_scene/` | SuGaR model definition (`sugar_model.py`), Gaussian model (`gs_model.py`), optimizers, densifiers |
| `sugar_utils/` | Mesh rasterization, loss functions, spherical harmonics, nvdiffrast utilities |
| `configs/` | YAML configurations for different training presets |

### SuGaR's Embedded 3DGS

SuGaR includes its own copy of 3D Gaussian Splatting at `src/sugar/gaussian_splatting/`. This is a standard 3DGS implementation with `ModelParams`, `PipelineParams`, `OptimizationParams` in `arguments/__init__.py`. Scene loading (`scene/__init__.py`) auto-detects COLMAP vs Blender format by checking for `sparse/` or `transforms_train.json`.

```bash
# Train vanilla 3DGS (prerequisite for SuGaR)
cd src/sugar
python gaussian_splatting/train.py -s <scene_path> -m <model_output_path> \
    [--iterations 7000]  # SuGaR loads from iteration 7000 by default
```

---

## src/pgsr/ -- PGSR (Planar-based Gaussian Splatting Reconstruction)

Extends 3DGS with planar Gaussian primitives for improved geometry. Uses `diff-plane-rasterization` (custom CUDA rasterizer in `submodules/`).

### Training

```bash
cd src/pgsr
python train.py -s <scene_path> -m <model_output_path> \
    [--iterations 30000] \
    [--test_iterations 7000 30000] \
    [--save_iterations 7000 30000]
```

**Key differences from vanilla 3DGS:**
- **Plane-based rasterization**: Renders `plane_depth`, `rendered_normal`, `rendered_distance` alongside RGB
- **AppModel** (`scene/app_model.py`): Per-image appearance compensation (exposure, color shift)
- **Single-view loss**: Normal consistency between rendered normals and depth-derived normals (enabled after iter 7000)
- **Multi-view loss**: Geometric consistency (reprojection error) + photometric consistency (NCC patch matching) between neighboring views (enabled after iter 7000)
- **Virtual camera augmentation**: Generates perturbed camera poses for multi-view training
- **Multi-view trimming**: Prunes Gaussians observed by fewer than 2 cameras (every 1000 iterations)
- **Nearest-view computation**: Scene init computes nearest cameras per frame based on distance and viewing angle

### Rendering + Mesh Extraction

```bash
cd src/pgsr
python render.py -m <model_path> \
    [--iteration -1]         \   # -1 = latest
    [--skip_train] [--skip_test] \
    [--max_depth 5.0]        \
    [--voxel_size 0.002]     \   # TSDF voxel size
    [--num_cluster 1]        \   # connected components to keep
    [--use_depth_filter]         # filter by view angle
```

Renders all views, runs TSDF fusion on rendered depth maps, and extracts a triangle mesh (`tsdf_fusion.ply` + `tsdf_fusion_post.ply`). Post-processing removes small disconnected clusters.

### Key Parameters (`arguments/__init__.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `multi_view_num` | 8 | Nearest views for multi-view loss |
| `multi_view_max_angle` | 30 | Max angle (degrees) for nearest view selection |
| `multi_view_max_dis` | 1.5 | Max distance for nearest view selection |
| `single_view_weight` | 0.015 | Normal consistency weight |
| `multi_view_ncc_weight` | 0.15 | NCC patch-matching weight |
| `multi_view_geo_weight` | 0.03 | Geometric consistency weight |
| `scale_loss_weight` | 100.0 | Min-scale regularization weight |
| `ncc_scale` | 1.0 | Resolution scale for NCC patch sampling |

---

## src/2dgs/ -- 2DGS (2D Gaussian Splatting)

Represents scenes with 2D surfel primitives instead of 3D Gaussians, yielding better surface geometry. Uses `diff-surfel-rasterization` (custom CUDA rasterizer in `submodules/`).

### Training

```bash
cd src/2dgs
python train.py -s <scene_path> -m <model_output_path> \
    [--iterations 30000] \
    [--test_iterations 7000 30000] \
    [--save_iterations 7000 30000]
```

**Key differences from vanilla 3DGS:**
- **Surfel rasterization**: Rasterizes 2D oriented disks instead of 3D ellipsoids. Produces `rend_normal`, `rend_alpha`, `surf_depth`, `surf_normal`, `rend_dist`.
- **Normal consistency loss** (`lambda_normal`, default 0.05): Penalises disagreement between the rendered normal and the pseudo-surface normal derived from depth (enabled after iter 7000).
- **Distortion loss** (`lambda_dist`, default 0.0): Ray-along-distortion regularization for tighter depth distributions (enabled after iter 3000).
- **`depth_ratio`** (PipelineParams, default 0.0): Blends between expected depth (0.0) and median depth (1.0). Use median (1.0) for bounded scenes, expected (0.0) for unbounded scenes.
- **Mask support**: If `gt_alpha_mask` is present, background pixels are set to `bg_color` in the GT and alpha is concatenated as a 4th channel for supervision.

### Rendering + Mesh Extraction

```bash
cd src/2dgs
python render.py -m <model_path> \
    [--iteration -1]         \
    [--skip_train] [--skip_test] [--skip_mesh] \
    [--voxel_size -1.0]     \   # auto-computed if negative
    [--depth_trunc -1.0]    \   # auto = 2 * bounding_sphere_radius
    [--sdf_trunc -1.0]      \   # auto = 5 * voxel_size
    [--num_cluster 50]       \   # connected components to keep
    [--unbounded]            \   # marching cubes with contraction (experimental)
    [--mesh_res 1024]        \   # resolution for unbounded mesh
    [--render_path]              # render a smooth camera trajectory video
```

**Mesh extraction** uses `GaussianExtractor` (`utils/mesh_utils.py`):
- **Bounded mode** (default): TSDF fusion over rendered depth maps. Auto-estimates bounding sphere from camera poses. Writes `fuse.ply` + `fuse_post.ply`.
- **Unbounded mode** (`--unbounded`): Marching cubes with spatial contraction (`utils/mcube_utils.py`). Writes `fuse_unbounded.ply`.

### Key Parameters (`arguments/__init__.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_normal` | 0.05 | Normal consistency loss weight |
| `lambda_dist` | 0.0 | Distortion loss weight |
| `opacity_cull` | 0.05 | Opacity threshold for pruning |
| `depth_ratio` | 0.0 | Expected (0) vs median (1) depth blend |
| `render_items` | RGB, Alpha, Normal, Depth, Edge, Curvature | Visualisable quantities in GUI |

---

## Common Scene Format

All reconstruction backends (SuGaR, PGSR, 2DGS) consume the same COLMAP-format scene:

```
<scene>/
├── images/           # required; source images
├── masks/            # optional; binary PNGs (white=foreground, black=background)
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

Scene loading in all backends auto-detects COLMAP format by checking for `sparse/` directory. Each backend's `scene/__init__.py` reads cameras and 3D points via `dataset_readers.py` -> `colmap_loader.py`.

---

## Data Flow

```
Raw data (mixed images + masks)
    │
    ▼
prepare_uf_dataset.py ──► images/ + masks/
    │
    ├──► run_vggt_to_colmap.py ──► sparse/0/ (COLMAP format)
    │        VGGT → depth + cameras → optional BA → COLMAP binary
    │
    └──► run_colmap.sh ──► sparse/0/ (COLMAP format)
             SIFT → matching → mapper → undistortion
    │
    ▼
COLMAP scene (images/ + sparse/0/)
    │
    ├──► run_sugar.py → 3DGS checkpoint → coarse → mesh → refine → textured mesh (.obj)
    │
    ├──► run_2dgs.py → 2DGS model → TSDF fusion → mesh (.ply)
    │
    └──► run_pgsr.py → copy + flatten sparse/ → PGSR model → TSDF fusion → mesh (.ply)
```
