# Data Preparation and SfM Entry Points

All SfM paths emit the same COLMAP scene layout — see [scene-format.md](scene-format.md).

## Data preparation

```bash
python pipeline/preparation/prepare_uf_dataset.py <input_dir> \
    [--out <output_dir>] [--mode {copy,move,symlink}] [--include-unmatched]
```

Splits a flat directory of mixed images and masks (`<stem>.JPG` + `<stem>.jpg.mask.png`) into
`images/` + `masks/`. Images normalised to `.jpg`, masks to `.png` named after the image stem.

## VGGT → COLMAP (`pipeline/sfm/run_vggt_to_colmap.py`)

```bash
python pipeline/sfm/run_vggt_to_colmap.py \
    --input_dir <dir>            \  # must contain images/
    --output_dir <dir>           \
    [--use_masks] [--seed 42]    \
    [--conf_thres_value 1.0]     \  # depth-confidence threshold, no-BA mode
    [--use_ba]                   \  # VGGSfM tracker + pycolmap BA
    [--shared_camera] [--camera_type SIMPLE_PINHOLE] \
    [--max_reproj_error 8.0] [--vis_thresh 0.2] \
    [--query_frame_num 8] [--max_query_pts 4096] [--fine_tracking]
```

Runs VGGT inference, writes `sparse/0/`, copies images, exports `points.ply`. Masks are always
copied to `<output>/masks/` when `<input>/masks/` exists — `--use_masks` only controls whether
masks weight the depth confidence. Logs per-stage runtimes (model load, inference, tracking+BA)
and a total.

- **No BA (default):** VGGT depth + camera predictions directly; filter by `conf_thres_value`,
  random-subsample to 100k points, write PINHOLE cameras at 518 px, then rescale to original
  resolution.
- **With `--use_ba`:** VGGSfM tracker for correspondences, then `pycolmap.bundle_adjustment()`.
  Operates at 1024 px internally; supports SIMPLE_PINHOLE and shared-camera modes.

README's benchmarked BA invocation overrides the defaults:
`--use_ba --shared_camera --max_reproj_error 32 --max_query_pts 1048576 --query_frame_num 8`.

Model/geometry internals: [backend-vggt.md](backend-vggt.md).

## Classical COLMAP (`pipeline/sfm/run_colmap.sh`)

```bash
bash pipeline/sfm/run_colmap.sh \
    --input_dir <dataset_dir> --output_dir <output_dir> \
    [--camera_model PINHOLE] [--single_camera 1] \
    [--matcher {exhaustive,sequential,vocab_tree}] [--no_gpu] [--colmap <bin>]
```

Flags are `--input_dir` / `--output_dir` (renamed from `--input_path` / `--output_path` in
`b1d9420` for consistency with the VGGT script). The script derives
`IMAGE_PATH=<input>/images` and `MASK_PATH=<input>/masks` internally. Runs feature extraction →
matching → mapper → `image_undistorter`; outputs undistorted `images/` + `sparse/0/`, cleans up
the intermediate `distorted/` working dir, and prints per-step and total runtimes.

Masks here are **only copied** to `<output>/masks/` at the end — SIFT extraction in this script
does *not* use them. For mask-restricted features use `run_masked_colmap.py`.

## Masked COLMAP (`pipeline/sfm/run_masked_colmap.py`, added `15f1816`)

```bash
python pipeline/sfm/run_masked_colmap.py \
    --input_dir <dir> --output_dir <dir> \
    [--max_image_size 2400] [--camera_model SIMPLE_PINHOLE]
```

pycolmap-based (not the `colmap` CLI): `extract_features` with
`ImageReaderOptions.mask_path` → `match_exhaustive` → `incremental_mapping`, then writes the
model with the most registered images to `sparse/0/`.

- `images/` and `masks/` are **symlinked** into the output dir, not copied.
- COLMAP expects a mask named `<image_name>.png` (i.e. `foo.jpg.png`), so the script builds a
  `masks_colmap/` dir of symlinks renamed `<stem>.jpg.png`. Same trick in the turntable script.
- `camera_mode=PER_IMAGE`, `num_threads=8`, no undistortion step.
- Prints `COLMAP_FAIL` and exits 2 if no model reconstructs; otherwise `COLMAP_DONE`.

## Turntable refinement (`pipeline/sfm/run_turntable_to_colmap.py`, added `44d02b7`, BA in `0b5b069`)

Post-processes an **existing COLMAP scene** (it needs `sparse/0/` as input, so run VGGT/COLMAP
first) by fitting an exact turntable rig — fixed rotation axis, constant angular step — and
re-solving poses on circular orbits.

```bash
python pipeline/sfm/run_turntable_to_colmap.py \
    --input_dir <existing_colmap_scene> --output_dir <dir> \
    [--use_masks] [--camera_regex 'camera\d+'] [--step_deg <float>] \
    [--max_image_size 2400] [--retriangulate {auto,tracks,sift}] \
    [--max_reproj <px>] [--rig_ba {auto,on,off}] [--rig_ba_iters 3]
```

Flow:
1. Group images into physical cameras by `--camera_regex` (default `camera\d+`), order within a
   group by the last integer in the filename (`order_key`).
2. `fit_axis_step()` — SVD-fit a plane to each group's camera centres for the axis; angular step
   from a least-squares slope of unwrapped angle vs. frame index, median across groups.
3. Step **sign is ambiguous**: both `+step` and `-step` are fitted and the one with lower centre
   error wins (skipped when `--step_deg` is given).
4. Retriangulation mode: `auto` picks `tracks` when the input mean track length >= 3.0, else
   falls back to masked SIFT (`sift`).
5. `tracks` mode: optional rig-constrained BA (`rig_ba`, resection–intersection rounds) refines
   axis/step/rig, then `apply_track_preserving()` re-triangulates existing tracks against the
   fixed poses with an adaptive reprojection threshold (default `2.5x` the median). Cameras are
   rewritten as SIMPLE_PINHOLE.
6. `sift` mode: masked SIFT + exhaustive matching, per-group shared focal (median), then batched
   DLT triangulation.

`--use_masks` auto-enables when `<input>/masks/` exists. `--rig_ba off` exists for ablations.
