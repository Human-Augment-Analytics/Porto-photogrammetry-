# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

**Keep this file short.** It is orientation only: what the project is, which entry point to
reach for, and the gotchas that change what you type. All depth — full flag lists, backend
internals, parameter tables, algorithm walk-throughs — lives in `.claude/MEMORY/`. When you
document something new, put the detail in the matching MEMORY file (or add one and index it
below); only add to CLAUDE.md if it changes how the repo is invoked or navigated.

## MEMORY index

| File | Contents |
|------|----------|
| [environment-and-gpu.md](MEMORY/environment-and-gpu.md) | Install via `scripts/`, per-GPU wrappers, numpy/build gotchas, submodules, GPU notes |
| [pipeline-sfm.md](MEMORY/pipeline-sfm.md) | Data prep + all four SfM entry points with full flags (incl. turntable algorithm) |
| [pipeline-reconstruction.md](MEMORY/pipeline-reconstruction.md) | The four reconstruction wrappers, their flags, and output paths |
| [scene-format.md](MEMORY/scene-format.md) | COLMAP scene layout, mask naming, end-to-end data flow |
| [backend-vggt.md](MEMORY/backend-vggt.md) | VGGT architecture, utils, COLMAP conversion, coordinate conventions |
| [backend-sugar.md](MEMORY/backend-sugar.md) | SuGaR four-stage pipeline, extraction, components |
| [backend-2dgs.md](MEMORY/backend-2dgs.md) | 2DGS surfels, losses, TSDF/marching-cubes extraction |
| [backend-pgsr.md](MEMORY/backend-pgsr.md) | PGSR planar Gaussians, multi-view losses, TSDF meshing |
| [backend-gaussian-wrapping.md](MEMORY/backend-gaussian-wrapping.md) | GW three stages, components, CUDA submodule matrix |
| [baseline-meshroom.md](MEMORY/baseline-meshroom.md) | Meshroom wrapper + reference runtimes |
| [repo-conventions.md](MEMORY/repo-conventions.md) | Doc rule, repo layout, legacy code, ID/naming and commit conventions |

## Project overview

**Augenblick** is a two-stage photogrammetry pipeline producing 3D meshes from multi-view
images: an SfM initialiser feeds a Gaussian-primitive-based surface reconstructor. The research
question is how different SfM initialisations interact with each mesh extractor.

| Stage | Options |
|-------|---------|
| Data preparation | `pipeline/preparation/prepare_uf_dataset.py` |
| SfM | `pipeline/sfm/run_vggt_to_colmap.py` (± `--use_ba`), `run_colmap.sh`, `run_masked_colmap.py`, `run_turntable_to_colmap.py` |
| Reconstruction | `pipeline/reconstruction/run_{sugar,2dgs,pgsr,gw}.py` |
| Baselines | Meshroom (`baseline/benchmark_meshroom.py`), RealityScan (external) |

Everything between the stages is a COLMAP scene: `images/` + optional `masks/` + `sparse/0/`.

## Quick start

```bash
conda create --name augenblick python=3.10 && conda activate augenblick
git submodule update --init --recursive
bash scripts/auto_setup.sh                  # detects GPU, dispatches to setup_<gpu>.sh

python pipeline/sfm/run_vggt_to_colmap.py --input_dir <scene> --output_dir <sfm> --use_ba
python pipeline/reconstruction/run_2dgs.py <sfm> <out>
```

There is **no `environment.yml`** — `scripts/auto_setup.sh` (or the manual pip sequence in
README) is the install path. Details and knobs (`BACKENDS`, `SKIP_TETRA`, numpy<2 pin, stale
`build/` dirs): [environment-and-gpu.md](MEMORY/environment-and-gpu.md).

## Gotchas worth knowing before you type

- `run_colmap.sh` takes `--input_dir` / `--output_dir` (renamed from `--input_path` /
  `--output_path`), pointing at the dataset dir, not at `images/`.
- `run_colmap.sh` copies masks to the output but does **not** feed them to SIFT — use
  `run_masked_colmap.py` for mask-restricted features.
- `run_turntable_to_colmap.py` refines an **existing** COLMAP scene; it is a post-SfM step, not a
  standalone SfM.
- COLMAP wants masks named `<image_name>.png` (`foo.jpg.png`); the pycolmap scripts build a
  `masks_colmap/` symlink dir to satisfy this.
- `run_gw.py` invokes its subprocesses by absolute path with **no `cwd`**, and forwards unknown
  flags to the training step only; its boolean flags use `--no-<flag>` spellings.
- `run_pgsr.py` flattens `sparse/0/` → `sparse/` because PGSR expects no `0/`.
- COLMAP IDs are 1-indexed — `+1` offset from VGGT batch indices.
- `src/pipeline/` is legacy first-generation code; use `pipeline/` for new work.
- Only `src/light_glue`, `src/pytorch3d`, and `src/gaussian_wrapping/submodules/Depth-Anything-V2`
  are git submodules — `src/sugar` and the other backends are vendored in-tree.
