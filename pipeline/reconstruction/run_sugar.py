#!/usr/bin/env python3
"""
Run the full SuGaR reconstruction pipeline: vanilla 3DGS training followed by
SuGaR coarse training, mesh extraction, refinement, and textured mesh export.

Input: a COLMAP-format scene directory (images/ + sparse/0/).
Output: a textured .obj mesh and supporting .ply files.

Usage:
    python pipeline/reconstruction/run_sugar.py <scene_dir> <output_dir> \
        [--gs_iterations 20000] [--iteration_to_load 7000] \
        [--regularization dn_consistency] [--high_poly] [--refinement_time long] \
        [--white_background]
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
SUGAR_DIR = REPO_ROOT / "src" / "sugar"
GS_TRAIN_SCRIPT = SUGAR_DIR / "gaussian_splatting" / "train.py"
SUGAR_TRAIN_SCRIPT = SUGAR_DIR / "train.py"


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
        description="Run vanilla 3DGS + SuGaR reconstruction pipeline.",
    )
    parser.add_argument("scene_dir", type=Path, help="COLMAP scene directory (contains images/ and sparse/0/)")
    parser.add_argument("output_dir", type=Path, help="Root output directory")

    # 3DGS parameters
    parser.add_argument("--gs_iterations", type=int, default=20_000, help="Vanilla 3DGS training iterations")
    parser.add_argument("--gs_densify_grad_threshold", type=float, default=0.0002,
                        help="Lower → denser Gaussian cloud (more detail, more memory)")
    parser.add_argument("--gs_densify_until_iter", type=int, default=15_000,
                        help="Iteration to stop densifying; extend alongside --gs_iterations")
    parser.add_argument("--gs_lambda_dssim", type=float, default=0.2, help="SSIM loss weight for 3DGS")
    parser.add_argument("--gs_sh_degree", type=int, default=3, help="Max spherical harmonics degree")

    # SuGaR parameters
    parser.add_argument("--iteration_to_load", type=int, default=7_000, help="3DGS iteration to load for SuGaR")
    parser.add_argument("--regularization", type=str, default="dn_consistency",
                        choices=["sdf", "density", "dn_consistency"],
                        help="Coarse SuGaR regularization type")
    parser.add_argument("--surface_level", type=float, default=0.3, help="Isosurface level for mesh extraction")
    parser.add_argument("--n_vertices", type=int, default=1_000_000, help="Target vertex count")
    parser.add_argument("--gaussians_per_triangle", type=int, default=1, help="Gaussians per mesh triangle")
    parser.add_argument("--refinement_iterations", type=int, default=15_000, help="Refinement training iterations")
    parser.add_argument("--low_poly", action="store_true", help="200k vertices, 6 gaussians/triangle")
    parser.add_argument("--high_poly", action="store_true", help="1M vertices, 1 gaussian/triangle")
    parser.add_argument("--refinement_time", type=str, default=None, choices=["short", "medium", "long"],
                        help="Preset refinement duration (2k/7k/15k iterations)")
    parser.add_argument("--square_size", type=int, default=8,
                        help="UV texture square size (larger → finer baked texture)")
    parser.add_argument("--postprocess_mesh", action="store_true",
                        help="Remove low-density border triangles (risky; can help single-sided objects)")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")

    args = parser.parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()

    gs_model_dir = output_dir / "gs_model"
    sugar_output_dir = output_dir / "sugar"

    logger.info("=" * 60)
    logger.info("SuGaR Reconstruction Pipeline")
    logger.info(f"  Scene:  {scene_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    # Step 1: Train vanilla 3DGS
    logger.info("Step 1/2: Training vanilla 3DGS")
    t0 = time.time()
    run(
        [
            sys.executable, str(GS_TRAIN_SCRIPT),
            "-s", str(scene_dir),
            "-m", str(gs_model_dir),
            "--iterations", str(args.gs_iterations),
            "--densify_grad_threshold", str(args.gs_densify_grad_threshold),
            "--densify_until_iter", str(args.gs_densify_until_iter),
            "--lambda_dssim", str(args.gs_lambda_dssim),
            "--sh_degree", str(args.gs_sh_degree),
        ],
        cwd=str(SUGAR_DIR),
    )
    gs_time = time.time() - t0
    logger.info(f"3DGS training completed in {gs_time:.1f}s")

    # Step 2: Train SuGaR (coarse + mesh + refine + texture)
    logger.info("Step 2/2: Training SuGaR (coarse -> mesh -> refine -> texture)")
    t0 = time.time()
    sugar_cmd = [
        sys.executable, str(SUGAR_TRAIN_SCRIPT),
        "-s", str(scene_dir),
        "-c", str(gs_model_dir),
        "-o", str(sugar_output_dir),
        "-i", str(args.iteration_to_load),
        "-r", args.regularization,
        "-l", str(args.surface_level),
        "-v", str(args.n_vertices),
        "-g", str(args.gaussians_per_triangle),
        "-f", str(args.refinement_iterations),
        "--square_size", str(args.square_size),
        "--eval", "False",
        "--gpu", str(args.gpu),
    ]
    if args.postprocess_mesh:
        sugar_cmd += ["--postprocess_mesh", "True"]
    if args.low_poly:
        sugar_cmd += ["--low_poly", "True"]
    if args.high_poly:
        sugar_cmd += ["--high_poly", "True"]
    if args.refinement_time:
        sugar_cmd += ["--refinement_time", args.refinement_time]
    if args.white_background:
        sugar_cmd += ["--white_background", "True"]

    run(sugar_cmd, cwd=str(SUGAR_DIR))
    sugar_time = time.time() - t0
    logger.info(f"SuGaR training completed in {sugar_time:.1f}s")

    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info(f"  3DGS time:  {gs_time:.1f}s")
    logger.info(f"  SuGaR time: {sugar_time:.1f}s")
    logger.info(f"  Total:      {gs_time + sugar_time:.1f}s")
    logger.info(f"  Output:     {sugar_output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
