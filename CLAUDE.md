# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Augenblick** is a photogrammetry pipeline for producing 3D meshes from multi-view images. It replaces classical COLMAP feature-matching with a feed-forward transformer (VGGT) and feeds its outputs into several neural surface-reconstruction backends. The project supports two SfM paths (VGGT, classical COLMAP) and four Gaussian-splatting-based reconstruction backends (SuGaR, PGSR, 2DGS, Gaussian Wrapping), all of which consume COLMAP-format sparse reconstructions.

| Pipeline stage | Options |
|----------------|---------|
| Data preparation | `pipeline/preparation/prepare_uf_dataset.py` |
| SfM | VGGT (`pipeline/sfm/run_vggt_to_colmap.py`) or COLMAP (`pipeline/sfm/run_colmap.sh`) |
| Reconstruction | SuGaR (`pipeline/reconstruction/run_sugar.py`), 2DGS (`pipeline/reconstruction/run_2dgs.py`), PGSR (`pipeline/reconstruction/run_pgsr.py`), Gaussian Wrapping (`pipeline/reconstruction/run_gw.py`) |
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

# Gaussian Wrapping: train (--rasterizer ours) → pivot mesh extraction → texture refinement
python pipeline/reconstruction/run_gw.py <scene_dir> <output_dir> \
    [--iterations 30000] [--sh_degree 3] [--max_gaussians 6000000] \
    [--n_pivots 2] [--std_factor 3.0] [--n_binary_steps 10] [--isosurface_value 0.0] \
    [--use_searched_pivots] [--use_smallest_axis_as_normal] \
    [--no-postprocess] [--no-filter_large_edges] \
    [--texture_n_iter 1000] [--texture_lr 0.0025] \
    [-r 2]                       # image downsample (used for metrics)
```

Each script runs the underlying `src/` training and rendering scripts via `subprocess.run()`; SuGaR / 2DGS / PGSR set `cwd` to the backend's source directory. PGSR's `run_pgsr.py` handles the sparse directory flattening automatically (PGSR expects `sparse/` not `sparse/0/`). `run_gw.py` is the exception: it invokes `train.py`, `pivot_based_mesh_extraction.py`, and `texture_mesh.py` (all under `src/gaussian_wrapping/`) by absolute path without setting `cwd`; module imports like `from scene.gaussian_model import ...` resolve because Python prepends the script's directory to `sys.path`. Unrecognised CLI flags on `run_gw.py` are forwarded only to the training step (via `parse_known_args()`).

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

## src/gaussian_wrapping/ -- Gaussian Wrapping ("Blobs to Spokes")

Reconstructs watertight, textured surface meshes by interpreting 3D Gaussians as stochastic oriented surface elements (Gomez et al., 2026, arXiv:2604.07337). Multiple rasterizer backends (`ours` median-depth, `radegs`, `sof`) and SDF modes (`ours`, `exact_computation`) are available; the canonical pipeline uses `--rasterizer ours` + `--sdf_mode ours`.

> **Note on layout:** Internal modules import as `from scene.gaussian_model import ...`, `from gaussian_renderer.ours import ...`, etc. — i.e. `src/gaussian_wrapping/` is the package root. `pipeline/reconstruction/run_gw.py` calls each script by absolute path; the script's directory is prepended to `sys.path` automatically by Python so the imports resolve. (Unlike the other reconstruction wrappers, `run_gw.py` does **not** set `cwd` on its `subprocess.run` calls.)

### Three-stage pipeline

The wrapper `pipeline/reconstruction/run_gw.py` orchestrates:

1. **Training** (`train.py`): hardcoded `--rasterizer ours`, `--exposure_compensation`, `--data_device cpu`, `--N_max_gaussians 6000000`. Multi-view NCC + geometric consistency losses, normal-field densification (iters ~22k–26k), depth-normal regularization (`mask_depth_normal=True` is auto-set when `--rasterizer ours`).
2. **Pivot-based mesh extraction** (`pivot_based_mesh_extraction.py`): hardcoded `--sdf_mode ours`, `--dtype int32`, `--use_valid_mask`, `--isosurface_value 0.0`, `--n_binary_steps 10`. Marching-tetrahedra over Delaunay tetrahedralisation of pivot points; binary search refinement; optional `--postprocess` (default on, strips floaters) and `--filter_large_edges` (default on).
3. **Texture refinement** (`texture_mesh.py`): bakes per-vertex colors from rendered Gaussian views via L1 + fused-SSIM, default 1000 iters, `sh_degree_for_texturing=0`. Output: `{mesh_stem}_texture_refined_{iter-1}.ply`.

Mesh filename convention: `mesh_{sdf_mode}_{n_pivots}pivots[_post].ply` written under `<output_dir>` (the model_path). The texture stage appends `_texture_refined_{iter}` to the stem. The driver computes these paths via `get_mesh_path()` / `get_textured_mesh_path()`.

### Key components

| Path | Role |
|------|------|
| `train.py` | Full optimisation loop. CLI in `__main__` (lines 729-889) selects rasterizer, loads YAML configs for multiview/MILo/depth-order/normal-field, and forwards into `training()`. |
| `pivot_based_mesh_extraction.py` | `marching_tetrahedra_with_binary_search()` is the entry point. SDF mode dispatches to `integrate_ours` (default) or SOF transmittance. `compute_valid_mask` reprojects pivots through every camera and ANDs `gt_mask` where present. |
| `texture_mesh.py` | Loads Gaussians + mesh, optimises `_verts_colors` against rendered views. Mesh rasterisation via `ScalableMeshRenderer` / `MeshRenderer` (nvdiffrast). |
| `primal_adaptive_meshing_extraction.py` | Alternative extraction: samples candidate points from an existing mesh, refines onto the occupancy isosurface via gradient descent, reconstructs via Delaunay. Supports `--bounding_box_method {scene,ground_truth,blender}`. |
| `scripts/train_and_extract_gw_ours.py` | Upstream end-to-end driver with hardcoded `train_and_extract_gw_ours.py` flags. The repo's `pipeline/reconstruction/run_gw.py` is a more configurable wrapper around the same three stages. |
| `scripts/train_and_extract_gw_radegs.py` | Same flow with the RaDe-GS rasterizer (slower, used for qualitative MipNeRF360 results upstream). |
| `scripts/benchmark_{tnt,mip360}_gw_{ours,radegs}.py` | Dataset-specific batch benchmark wrappers. |
| `scene/gaussian_model.py` | `GaussianModel` with `learn_occupancy`, `n_pivots_per_gaussian`, 3D Mip filter (`compute_3D_filter`), exposure compensation, RaDe-GS densification (`densify_and_prune_radegs`). |
| `scene/__init__.py` | `Scene` auto-detects COLMAP (`sparse/`) vs Blender (`transforms_train.json`); loads checkpoints or calls `create_from_pcd`. |
| `scene/mesh.py` | `Meshes`, `MeshRasterizer`, `MeshRenderer`, `ScalableMeshRenderer`, QEM utilities, `return_delaunay_tets(method="tetranerf")`. |
| `gaussian_renderer/ours.py` | `render_ours`, `integrate_ours`, `sample_depth_with_ours`. Backed by `diff_gaussian_rasterization_gw`. |
| `gaussian_renderer/radegs.py` | `render_radegs`, `integrate_radegs`. Top-level import is guarded by `try/except ImportError`. |
| `gaussian_renderer/sof.py` | `render_sof`, vacancy/transmittance evaluators. Used only by `--sdf_mode exact_computation` or `--milo`. |
| `extraction/pivots.py` | `get_intersecting_pivots_from_normals` (default), `get_pivots_by_scores`, `sample_random_pivots`, `get_searched_pivots`. |
| `extraction/mesh.py` | `extract_mesh`, `compute_isosurface_value_from_depth`. |
| `regularization/sdf/learnable.py` | `refine_intersections_with_binary_search`, SDF↔occupancy conversions. |
| `regularization/sdf/depth_fusion.py` | `AdaptiveTSDF`, `evaluate_mesh_colors_all_vertices`, `frustum_cull_mesh`. |
| `regularization/regularizer/multiview.py` | NCC patch-matching + geometric consistency across nearest views (`--multiview` defaults to True). |
| `regularization/regularizer/mesh_in_the_loop.py` | MILo depth/normal/occupancy losses against a Delaunay mesh rebuilt every `reset_delaunay_every` iters. |
| `regularization/regularizer/normal_field.py` | Normal-field initialisation, regularization, densification, non-maximal pruning. |
| `regularization/regularizer/depth_order.py` | Depth-Anything-V2 supervision; off by default, enabled with `--depth_order`. |
| `arguments/__init__.py` | `ModelParams`, `PipelineParams`, `OptimizationParams`. `get_combined_args` merges CLI with the `cfg_args` file from `model_path`. |
| `configs/` | YAML presets for `mesh_in_the_loop/`, `multiview/`, `normal_field/`, `depth_order/`, `mesh/`. |

### CUDA/C++ submodules (under `submodules/`)

The repo-level `python -m pip install ... --no-build-isolation` step in [README.md](README.md) installs the four wheels GW needs alongside the other backends. The CGAL-based `tetra_triangulation` is built separately (via `cmake` + `make` + `pip install -e .`) because it requires `CPATH` to be set to the CUDA include directory.

| Submodule | Purpose | When loaded |
|-----------|---------|-------------|
| `diff-gaussian-rasterization-gw` | Median-depth rasterizer for `render_ours` / `integrate_ours` | Always on the `ours` path |
| `diff-gaussian-rasterization-ms` | Mini-Splatting2; provides fused-SSIM `_C` binding and `render_depth`/`render_simp` for normal-field densification | Always (top-level import in `utils/loss_utils.py`) |
| `fused-ssim` | Fast SSIM for the photometric loss | Always |
| `warp-patch-ncc` | NCC patch matching for the multiview regularizer | Always (`--multiview` defaults to True) |
| `tetra_triangulation` | CGAL Delaunay tetrahedralisation (`return_delaunay_tets(method="tetranerf")`) | Pivot extraction stage |
| `nvdiffrast` (vendored) | Mesh rasterisation backing `MeshRasterizer` | Texture refinement stage |
| `Depth-Anything-V2` | Monocular depth prior | Only with `--depth_order` (off by default) |
| `diff-gaussian-rasterization` (RaDe-GS) | RaDe-GS rasterizer | Only with `--rasterizer radegs`; top-level import is `try/except`-guarded |
| `diff-gaussian-rasterization-sof` | SOF transmittance rasterizer | Only with `--sdf_mode exact_computation` or `--milo` |

The repo currently installs `diff-gaussian-rasterization-gw`, `diff-gaussian-rasterization-ms`, `fused-ssim`, `warp-patch-ncc`, and `tetra_triangulation`. RaDe-GS, SOF, and Depth-Anything-V2 are not built in the canonical install (the `ours` path's imports are guarded so this is safe).

### Output filenames under `<output_dir>`

```
<output_dir>/
├── point_cloud/iteration_<N>/point_cloud.ply   # trained Gaussians
├── cfg_args, time.txt, cameras.json, input.ply # scene metadata
├── mesh_ours_2pivots.ply                       # raw pivot-based mesh
├── mesh_ours_2pivots_post.ply                  # post-processed (floaters stripped)
└── mesh_ours_2pivots_post_texture_refined_<iter-1>.ply  # final textured mesh
```

`<iter-1>` because `texture_mesh.py` writes the iteration index, not count (e.g. `texture_n_iter=1000` → file suffix `_999`).

### Inclusion notes

The driver path (`--rasterizer ours` + `--sdf_mode ours`) avoids RaDe-GS and SOF, which require additional CUDA kernels. Top-level imports of those backends in `gaussian_renderer/__init__.py` and `pivot_based_mesh_extraction.py` are wrapped in `try/except ImportError` or pushed inside `if args.rasterizer == "radegs":` branches, so the install can skip those wheels safely. See `gaussian-wrapping-inclusion.md` (repo root) for a full audit.

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
    ├──► run_pgsr.py → copy + flatten sparse/ → PGSR model → TSDF fusion → mesh (.ply)
    │
    └──► run_gw.py → GW (ours) train → pivot marching-tetrahedra extract → texture refine → textured mesh (.ply)
```
