# `src/libs/sugar/` — SuGaR (Surface-Aligned Gaussians)

Extracts textured triangle meshes from 3DGS by regularising Gaussians onto an implicit surface,
then binding them to an explicit mesh. Vendored in-tree (**not** a git submodule, despite older
docs). Requires a pre-trained vanilla 3DGS checkpoint from
`src/libs/sugar/gaussian_splatting/train.py`; `pipeline/reconstruction/run_sugar.py` produces it.

## Training pipeline (`src/libs/sugar/train.py`, run from `src/libs/sugar/`)

```bash
python train.py \
    -s <scene_path> -c <3dgs_checkpoint> \
    [-i <iteration_to_load>]              \  # default 7000
    [-r {sdf,density,dn_consistency}]     \  # regularization type
    [-o <output_path>]                    \  # organises coarse/coarse_mesh/refined/refined_mesh
    [-v <n_vertices>]                     \  # default 1M
    [-g <gaussians_per_tri>]              \  # default 1
    [-f <refinement_iters>]               \  # default 15000
    [--low_poly] [--high_poly]            \  # 200k verts/6 gpt | 1M verts/1 gpt
    [--refinement_time {short,medium,long}]  # 2k / 7k / 15k iterations
```

Four sequential stages:
1. **Coarse SuGaR training** (`sugar_trainers/`) — loads the 3DGS checkpoint, trains with
   surface regularization. `dn_consistency` gives the best mesh quality.
2. **Coarse mesh extraction** (`sugar_extractors/coarse_mesh.py`) — mesh at the surface level
   (default 0.3), decimated to the target vertex count.
3. **Refined SuGaR training** (`sugar_trainers/refine.py`) — binds Gaussians to the mesh,
   refines with a normal-consistency loss.
4. **Textured mesh extraction** (`sugar_extractors/refined_mesh.py`) — UV-unwrap + texture atlas
   bake, `.obj` output.

## Standalone extraction

```bash
python extract_mesh.py -s <scene> -c <3dgs_ckpt> -m <coarse_model_path> \
    [-l <surface_level>] [-d <decimation_target>]

python extract_refined_mesh_with_texture.py -s <scene> -c <3dgs_ckpt> \
    -m <refined_model_path> [--coarse_mesh_dir <dir>] [-o <output_dir>]
```

## Components

| Directory | Role |
|-----------|------|
| `gaussian_splatting/` | Embedded vanilla 3DGS (train, render, scene loading) — the starting checkpoint |
| `gsplat_wrapper/` | Alternative rasterization backend using gsplat |
| `sugar_trainers/` | Coarse training (density / SDF / DN-consistency) + refinement |
| `sugar_extractors/` | Mesh extraction from coarse and refined models |
| `sugar_scene/` | `sugar_model.py`, `gs_model.py`, optimizers, densifiers |
| `sugar_utils/` | Mesh rasterization, losses, spherical harmonics, nvdiffrast utils |
| `configs/` | YAML training presets |

Embedded 3DGS is a standard implementation (`ModelParams`/`PipelineParams`/`OptimizationParams`
in `arguments/__init__.py`; COLMAP-vs-Blender auto-detection in `scene/__init__.py`):

```bash
cd src/libs/sugar
python gaussian_splatting/train.py -s <scene_path> -m <model_output_path> [--iterations 7000]
```
