#!/usr/bin/env python3
"""
Run the PGSR reconstruction pipeline: training followed by rendering and
TSDF mesh extraction.

Input: a COLMAP-format scene directory (images/ + sparse/0/).
Output: a triangle mesh (.ply) via TSDF fusion of rendered depth maps.

Note: PGSR expects sparse/ files directly (not inside sparse/0/). This
script copies the scene directory and flattens the sparse structure
automatically.

Usage:
    python pipeline/reconstruction/run_pgsr.py <scene_dir> <output_dir> \
        [--iterations 30000] [--max_depth 10.0] [--voxel_size 0.001] \
        [--num_cluster 1] [--opacity_cull_threshold 0.05] \
        [--max_abs_split_points 0] [--white_background]
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PGSR_DIR = REPO_ROOT / "src" / "pgsr"
TRAIN_SCRIPT = PGSR_DIR / "train.py"
RENDER_SCRIPT = PGSR_DIR / "render.py"


def run(cmd, cwd=None):
    """Run a command, streaming output to the terminal."""
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"  cwd: {cwd or os.getcwd()}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def prepare_scene(scene_dir: Path, output_dir: Path) -> Path:
    """Copy scene and flatten sparse/0/ into sparse/ for PGSR.

    PGSR reads COLMAP files directly from sparse/, not sparse/0/.
    This copies the scene directory and moves sparse/0/* up one level.

    Returns the path to the prepared scene directory.
    """
    pgsr_scene = output_dir / "scene"

    if pgsr_scene.exists():
        logger.info(f"Prepared scene already exists at {pgsr_scene}, reusing")
        return pgsr_scene

    logger.info(f"Copying scene from {scene_dir} to {pgsr_scene}")
    for subdir in ["images", "masks", "sparse"]:
        src = scene_dir / subdir
        dst = pgsr_scene / subdir
        shutil.copytree(src, dst)

    sparse_0 = pgsr_scene / "sparse" / "0"
    sparse = pgsr_scene / "sparse"
    if sparse_0.is_dir():
        logger.info("Flattening sparse/0/ -> sparse/")
        for item in sparse_0.iterdir():
            shutil.move(str(item), str(sparse / item.name))
        sparse_0.rmdir()
    else:
        logger.info("No sparse/0/ found; assuming sparse/ is already flat")

    return pgsr_scene


def main():
    parser = argparse.ArgumentParser(
        description="Run PGSR training and mesh extraction pipeline.",
    )
    parser.add_argument("scene_dir", type=Path, help="COLMAP scene directory (contains images/ and sparse/0/)")
    parser.add_argument("output_dir", type=Path, help="Model output directory")

    # Training parameters
    parser.add_argument("--iterations", type=int, default=30_000, help="Training iterations")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--max_abs_split_points", type=int, default=0, help="Max absolute split points (0 to disable)")
    parser.add_argument("--opacity_cull_threshold", type=float, default=0.05, help="Opacity threshold for pruning")
    parser.add_argument("--white_background", action="store_true", help="Use white background")

    # Fidelity knobs (forwarded to train.py)
    parser.add_argument("--lambda_dssim", type=float, default=0.2, help="SSIM loss weight")
    parser.add_argument("--single_view_weight", type=float, default=0.015,
                        help="Normal consistency weight (raise for sharper surfaces)")
    parser.add_argument("--multi_view_ncc_weight", type=float, default=0.15,
                        help="NCC patch-matching weight (cross-view photometric consistency)")
    parser.add_argument("--multi_view_geo_weight", type=float, default=0.03,
                        help="Multi-view geometric consistency weight")
    parser.add_argument("--multi_view_num", type=int, default=8,
                        help="Number of nearest views per frame for multi-view losses")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002,
                        help="Lower → more Gaussians (more detail, more memory)")
    parser.add_argument("--densify_until_iter", type=int, default=15_000,
                        help="Iteration to stop densifying; extend alongside --iterations")

    # Rendering / mesh extraction parameters
    parser.add_argument("--max_depth", type=float, default=10.0, help="Max depth for TSDF integration")
    parser.add_argument("--voxel_size", type=float, default=0.0005, help="TSDF voxel size")
    parser.add_argument("--num_cluster", type=int, default=1, help="Connected components to keep in mesh")
    parser.add_argument("--use_depth_filter", action="store_true",
                        help="Drop grazing-angle depths before TSDF fusion")
    parser.add_argument("--skip_mesh", action="store_true", help="Skip mesh extraction (render only)")

    args = parser.parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()

    logger.info("=" * 60)
    logger.info("PGSR Reconstruction Pipeline")
    logger.info(f"  Scene:  {scene_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    # Step 0: Prepare scene (copy + flatten sparse/)
    logger.info("Step 0: Preparing scene directory")
    pgsr_scene = prepare_scene(scene_dir, output_dir)

    # Step 1: Train PGSR
    logger.info("Step 1/2: Training PGSR")
    t0 = time.time()
    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "-s", str(pgsr_scene),
        "-m", str(output_dir),
        "--iterations", str(args.iterations),
        "--test_iterations", *[str(i) for i in args.test_iterations],
        "--save_iterations", *[str(i) for i in args.save_iterations],
        "--max_abs_split_points", str(args.max_abs_split_points),
        "--opacity_cull_threshold", str(args.opacity_cull_threshold),
        "--lambda_dssim", str(args.lambda_dssim),
        "--single_view_weight", str(args.single_view_weight),
        "--multi_view_ncc_weight", str(args.multi_view_ncc_weight),
        "--multi_view_geo_weight", str(args.multi_view_geo_weight),
        "--multi_view_num", str(args.multi_view_num),
        "--densify_grad_threshold", str(args.densify_grad_threshold),
        "--densify_until_iter", str(args.densify_until_iter),
    ]
    if args.white_background:
        train_cmd.append("--white_background")
    run(train_cmd, cwd=str(PGSR_DIR))
    train_time = time.time() - t0
    logger.info(f"PGSR training completed in {train_time:.1f}s")

    # Step 2: Render + mesh extraction
    logger.info("Step 2/2: Rendering and mesh extraction")
    t0 = time.time()
    render_cmd = [
        sys.executable, str(RENDER_SCRIPT),
        "-m", str(output_dir),
        "--max_depth", str(args.max_depth),
        "--voxel_size", str(args.voxel_size),
        "--num_cluster", str(args.num_cluster),
        "--skip_test",
    ]
    if args.use_depth_filter:
        render_cmd.append("--use_depth_filter")
    if args.skip_mesh:
        render_cmd.append("--skip_train")
    run(render_cmd, cwd=str(PGSR_DIR))
    render_time = time.time() - t0
    logger.info(f"Rendering + mesh extraction completed in {render_time:.1f}s")

    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info(f"  Train time:  {train_time:.1f}s")
    logger.info(f"  Render time: {render_time:.1f}s")
    logger.info(f"  Total:       {train_time + render_time:.1f}s")
    logger.info(f"  Scene:       {output_dir}")
    logger.info(f"  Mesh:        {output_dir / 'mesh' / 'tsdf_fusion_post.ply'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
