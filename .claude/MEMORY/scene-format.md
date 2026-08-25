# Common Scene Format and Data Flow

## Scene layout

Every reconstruction backend consumes the same COLMAP-format scene:

```
<scene>/
├── images/           # required; source images
├── masks/            # optional; binary PNGs (white = foreground, black = background)
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

Each backend's `scene/__init__.py` auto-detects COLMAP format by the presence of `sparse/`
(vs. Blender's `transforms_train.json`) and reads via `dataset_readers.py` → `colmap_loader.py`.
PGSR is the exception at the wrapper level: `run_pgsr.py` flattens `sparse/0/` → `sparse/`.

COLMAP mask naming quirk: COLMAP looks for `<image_name>.png` (e.g. `foo.jpg.png`), which is why
`run_masked_colmap.py` and `run_turntable_to_colmap.py` build a `masks_colmap/` symlink dir.

## Data flow

```
Raw data (mixed images + masks)
    │
    ▼
prepare_uf_dataset.py ──► images/ + masks/
    │
    ├──► run_vggt_to_colmap.py ──► sparse/0/    (VGGT → depth + cameras → optional BA)
    ├──► run_colmap.sh          ──► sparse/0/    (SIFT → matching → mapper → undistortion)
    └──► run_masked_colmap.py   ──► sparse/0/    (mask-restricted SIFT, pycolmap API)
                 │
                 └──► run_turntable_to_colmap.py ──► sparse/0/  (rig prior refinement of an
                                                                 existing scene)
    │
    ▼
COLMAP scene (images/ + sparse/0/)
    │
    ├──► run_sugar.py → 3DGS ckpt → coarse → mesh → refine → textured mesh (.obj)
    ├──► run_2dgs.py  → 2DGS model → TSDF fusion → mesh (.ply)
    ├──► run_pgsr.py  → flatten sparse/ → PGSR model → TSDF fusion → mesh (.ply)
    └──► run_gw.py    → GW train → pivot marching-tetrahedra → texture refine → mesh (.ply)
    │
    ▼
Baseline comparison: baseline/benchmark_meshroom.py (see baseline-meshroom.md)
```
