#!/usr/bin/env python3
"""
Run the 2DGS reconstruction pipeline: training followed by rendering and
TSDF mesh extraction.

Input: a COLMAP-format scene directory (images/ + sparse/0/).
Output: a triangle mesh (.ply) via TSDF fusion of rendered depth maps.

Usage:
    python pipeline/reconstruction/run_2dgs.py <scene_dir> <output_dir> \
        [--iterations 30000] [--voxel_size -1.0] [--depth_trunc -1.0] \
        [--num_cluster 50] [--unbounded] [--white_background]
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TWODGS_DIR = REPO_ROOT / "src" / "2dgs"
TRAIN_SCRIPT = TWODGS_DIR / "train.py"
RENDER_SCRIPT = TWODGS_DIR / "render.py"


def run(cmd, cwd=None):
    """Run a command, streaming output to the terminal."""
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"  cwd: {cwd or os.getcwd()}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run 2DGS training and mesh extraction pipeline.",
    )
    parser.add_argument("scene_dir", type=Path, help="COLMAP scene directory (contains images/ and sparse/0/)")
    parser.add_argument("output_dir", type=Path, help="Model output directory")

    # Training parameters
    parser.add_argument("--iterations", type=int, default=30_000, help="Training iterations")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--white_background", action="store_true", help="Use white background")

    # Fidelity knobs (forwarded to train.py)
    parser.add_argument("--lambda_dist", type=float, default=0.0,
                        help="Distortion loss weight (upstream default 0.0; 100 bounded / 1000 unbounded recommended)")
    parser.add_argument("--lambda_normal", type=float, default=0.05, help="Normal consistency loss weight")
    parser.add_argument("--depth_ratio", type=float, default=0.0,
                        help="Expected (0.0) vs median (1.0) depth blend; 1.0 for bounded scenes")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002,
                        help="Lower → more Gaussians (more detail, more memory)")
    parser.add_argument("--densify_until_iter", type=int, default=15_000,
                        help="Iteration to stop densifying; extend alongside --iterations")
    parser.add_argument("--opacity_cull", type=float, default=0.05, help="Opacity threshold for pruning")

    # Rendering / mesh extraction parameters
    parser.add_argument("--voxel_size", type=float, default=-1.0, help="TSDF voxel size (auto if negative)")
    parser.add_argument("--depth_trunc", type=float, default=-1.0, help="Max depth for TSDF (auto if negative)")
    parser.add_argument("--sdf_trunc", type=float, default=-1.0, help="SDF truncation (auto if negative)")
    parser.add_argument("--num_cluster", type=int, default=50, help="Connected components to keep in mesh")
    parser.add_argument("--unbounded", action="store_true", help="Use unbounded mesh extraction (marching cubes)")
    parser.add_argument("--mesh_res", type=int, default=8192, help="Resolution for unbounded mesh extraction")
    parser.add_argument("--skip_mesh", action="store_true", help="Skip mesh extraction (render only)")

    args = parser.parse_args()
    scene_dir = args.scene_dir.resolve()
    model_dir = args.output_dir.resolve()

    logger.info("=" * 60)
    logger.info("2DGS Reconstruction Pipeline")
    logger.info(f"  Scene:  {scene_dir}")
    logger.info(f"  Output: {model_dir}")
    logger.info("=" * 60)

    # Step 1: Train 2DGS
    logger.info("Step 1/2: Training 2DGS")
    t0 = time.time()
    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "-s", str(scene_dir),
        "-m", str(model_dir),
        "--iterations", str(args.iterations),
        "--test_iterations", *[str(i) for i in args.test_iterations],
        "--save_iterations", *[str(i) for i in args.save_iterations],
        "--lambda_dist", str(args.lambda_dist),
        "--lambda_normal", str(args.lambda_normal),
        "--depth_ratio", str(args.depth_ratio),
        "--densify_grad_threshold", str(args.densify_grad_threshold),
        "--densify_until_iter", str(args.densify_until_iter),
        "--opacity_cull", str(args.opacity_cull),
    ]
    if args.white_background:
        train_cmd.append("--white_background")
    run(train_cmd, cwd=str(TWODGS_DIR))
    train_time = time.time() - t0
    logger.info(f"2DGS training completed in {train_time:.1f}s")

    # Step 2: Render + mesh extraction
    logger.info("Step 2/2: Rendering and mesh extraction")
    t0 = time.time()
    render_cmd = [
        sys.executable, str(RENDER_SCRIPT),
        "-s", str(scene_dir),
        "-m", str(model_dir),
        "--voxel_size", str(args.voxel_size),
        "--depth_trunc", str(args.depth_trunc),
        "--sdf_trunc", str(args.sdf_trunc),
        "--num_cluster", str(args.num_cluster),
        "--mesh_res", str(args.mesh_res),
        "--skip_test",
    ]
    if args.unbounded:
        render_cmd.append("--unbounded")
    if args.skip_mesh:
        render_cmd.append("--skip_mesh")
    run(render_cmd, cwd=str(TWODGS_DIR))
    render_time = time.time() - t0
    logger.info(f"Rendering + mesh extraction completed in {render_time:.1f}s")

    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info(f"  Train time:  {train_time:.1f}s")
    logger.info(f"  Render time: {render_time:.1f}s")
    logger.info(f"  Total:       {train_time + render_time:.1f}s")
    logger.info(f"  Model:       {model_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
