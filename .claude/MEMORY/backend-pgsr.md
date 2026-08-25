# `src/pgsr/` — PGSR (Planar-based Gaussian Splatting Reconstruction)

3DGS extended with planar Gaussian primitives. Custom CUDA rasterizer
`diff-plane-rasterization` under `submodules/`.

## Training

```bash
cd src/pgsr
python train.py -s <scene_path> -m <model_output_path> \
    [--iterations 30000] [--test_iterations 7000 30000] [--save_iterations 7000 30000]
```

Differences from vanilla 3DGS:
- **Plane-based rasterization** — renders `plane_depth`, `rendered_normal`,
  `rendered_distance` alongside RGB.
- **AppModel** (`scene/app_model.py`) — per-image appearance compensation (exposure, colour shift).
- **Single-view loss** — normal consistency between rendered and depth-derived normals
  (after iter 7000).
- **Multi-view loss** — geometric consistency (reprojection error) + photometric consistency
  (NCC patch matching) between neighbouring views (after iter 7000).
- **Virtual camera augmentation** — perturbed camera poses for multi-view training.
- **Multi-view trimming** — prunes Gaussians seen by fewer than 2 cameras, every 1000 iters.
- **Nearest-view computation** — scene init picks nearest cameras per frame by distance and
  viewing angle.

## Rendering + mesh extraction

```bash
cd src/pgsr
python render.py -m <model_path> \
    [--iteration -1] [--skip_train] [--skip_test] \
    [--max_depth 5.0] [--voxel_size 0.002] [--num_cluster 1] [--use_depth_filter]
```

Renders all views, TSDF-fuses the rendered depths, writes `tsdf_fusion.ply` +
`tsdf_fusion_post.ply` (post-processing drops small disconnected clusters).
Note `run_pgsr.py` overrides the defaults with `--max_depth 10.0 --voxel_size 0.001`
(`voxel_size` default changed in `906b955`).

## Key parameters (`arguments/__init__.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `multi_view_num` | 8 | Nearest views for the multi-view loss |
| `multi_view_max_angle` | 30 | Max angle (deg) for nearest-view selection |
| `multi_view_max_dis` | 1.5 | Max distance for nearest-view selection |
| `single_view_weight` | 0.015 | Normal consistency weight |
| `multi_view_ncc_weight` | 0.15 | NCC patch-matching weight |
| `multi_view_geo_weight` | 0.03 | Geometric consistency weight |
| `scale_loss_weight` | 100.0 | Min-scale regularization weight |
| `ncc_scale` | 1.0 | Resolution scale for NCC patch sampling |
