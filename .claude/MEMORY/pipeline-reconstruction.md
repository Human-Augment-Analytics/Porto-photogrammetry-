# Reconstruction Wrappers (`pipeline/reconstruction/`)

Four wrappers, each taking a COLMAP scene dir and an output dir positionally, orchestrating the
underlying `src/` train + render scripts via `subprocess.run()` with a banner + total-time
summary.

```bash
# SuGaR: vanilla 3DGS → coarse SuGaR → mesh → refine → textured mesh (.obj)
python pipeline/reconstruction/run_sugar.py <scene_dir> <output_dir> \
    [--gs_iterations 20000] [--gs_densify_grad_threshold] [--gs_densify_until_iter] \
    [--gs_lambda_dssim] [--gs_sh_degree] [--iteration_to_load 7000] \
    [--regularization dn_consistency] [--surface_level] [--n_vertices] \
    [--gaussians_per_triangle] [--refinement_iterations] [--low_poly] [--high_poly] \
    [--refinement_time long] [--square_size] [--postprocess_mesh] [--white_background] [--gpu]

# 2DGS: training + TSDF mesh extraction
python pipeline/reconstruction/run_2dgs.py <scene_dir> <output_dir> \
    [--iterations 30000] [--lambda_dist] [--lambda_normal] [--depth_ratio] \
    [--densify_grad_threshold] [--densify_until_iter] [--opacity_cull] [--white_background] \
    [--voxel_size -1.0] [--depth_trunc -1.0] [--sdf_trunc -1.0] [--num_cluster 50] \
    [--unbounded] [--mesh_res 1024] [--skip_mesh]

# PGSR: copies scene, flattens sparse/0/ → sparse/, trains, TSDF mesh extraction
python pipeline/reconstruction/run_pgsr.py <scene_dir> <output_dir> \
    [--iterations 30000] [--max_abs_split_points 0] [--opacity_cull_threshold 0.05] \
    [--lambda_dssim] [--single_view_weight] [--multi_view_ncc_weight] [--multi_view_geo_weight] \
    [--multi_view_num] [--densify_grad_threshold] [--densify_until_iter] [--white_background] \
    [--max_depth 10.0] [--voxel_size 0.001] [--num_cluster] [--use_depth_filter] [--skip_mesh]

# Gaussian Wrapping: train (--rasterizer ours) → pivot mesh extraction → texture refinement
python pipeline/reconstruction/run_gw.py <scene_dir> <output_dir> \
    [--iterations 30000] [--sh_degree 3] [--max_gaussians 6000000] \
    [--densify_until_iter] [--densify_grad_threshold] [--lambda_depth_normal] \
    [--multiview_factor] [--<many>_lr ...] [--extract_iteration] \
    [--n_pivots 2] [--std_factor 3.0] [--n_binary_steps 10] [--isosurface_value 0.0] \
    [--use_searched_pivots] [--use_smallest_axis_as_normal] \
    [--no-postprocess] [--no-filter_large_edges] \
    [--texture_n_iter 1000] [--texture_lambda_dssim] [--texture_lr 0.0025] [--texture_sh_degree]
```

## Wrapper-specific behaviour

- **SuGaR / 2DGS / PGSR** set `cwd` to the backend source dir on each `subprocess.run`.
- **PGSR** copies the scene and flattens `sparse/0/` → `sparse/` (PGSR expects no `0/`).
- **`run_gw.py` is the exception**: it invokes `train.py`,
  `pivot_based_mesh_extraction.py`, and `texture_mesh.py` under `src/gaussian_wrapping/` by
  **absolute path with no `cwd`**; imports like `from scene.gaussian_model import ...` resolve
  because Python prepends the script's own directory to `sys.path`. Unrecognised flags are
  forwarded **only to the training step** (`parse_known_args()`). Boolean toggles use
  `argparse.BooleanOptionalAction`, hence the `--no-postprocess` / `--no-filter_large_edges`
  spellings (both default on).

## Output locations

| Backend | Output | Path |
|---------|--------|------|
| SuGaR | textured mesh `.obj` + `.ply` | `<output>/refined_mesh/<scene>/` |
| 2DGS | triangle mesh | `<model>/train/ours_<iter>/fuse_post.ply` |
| PGSR | triangle mesh | `<model>/mesh/tsdf_fusion_post.ply` |
| GW | raw / post / textured mesh | `<output>/mesh_ours_2pivots{,_post,_post_texture_refined_<iter-1>}.ply` |

Per-backend internals: [backend-sugar.md](backend-sugar.md), [backend-2dgs.md](backend-2dgs.md),
[backend-pgsr.md](backend-pgsr.md), [backend-gaussian-wrapping.md](backend-gaussian-wrapping.md).
