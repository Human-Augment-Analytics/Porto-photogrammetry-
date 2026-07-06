#!/usr/bin/env python3
"""
Run the full Gaussian Wrapping ("ours" rasterizer) reconstruction pipeline:
training, pivot-based mesh extraction, and texture refinement.

Input: a COLMAP-format scene directory (images/ + sparse/0/).
Output: a textured mesh (.ply / .obj) and supporting artifacts under <output_dir>.

Usage:
    python pipeline/reconstruction/run_gw.py <scene_dir> <output_dir> \
        [--iterations 30000] [--sh_degree 3] [--max_gaussians 6000000] \
        [--n_pivots 2] [--std_factor 3.0] [--n_binary_steps 10] [--isosurface_value 0.0] \
        [--use_searched_pivots] [--use_smallest_axis_as_normal] \
        [--no-postprocess] [--no-filter_large_edges] \
        [--texture_n_iter 1000] [--texture_lr 0.0025]
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from argparse import BooleanOptionalAction
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
GW_DIR = REPO_ROOT / "src" / "gaussian_wrapping"
TRAIN_SCRIPT = GW_DIR / "train.py"
EXTRACT_SCRIPT = GW_DIR / "pivot_based_mesh_extraction.py"
TEXTURE_SCRIPT = GW_DIR / "texture_mesh.py"

DEFAULT_TRAIN_FEATURE_DC_LR = 0.0013
DEFAULT_TRAIN_FEATURE_REST_LR = 0.00011
DEFAULT_TRAIN_POSITION_LR_INIT = 0.00016
DEFAULT_TRAIN_POSITION_LR_FINAL = 0.0000016
DEFAULT_TRAIN_POSITION_LR_DELAY_MULT = 0.01
DEFAULT_TRAIN_POSITION_LR_MAX_STEPS = 30_000
DEFAULT_TRAIN_OPACITY_LR = 0.05
DEFAULT_TRAIN_SCALING_LR = 0.005
DEFAULT_TRAIN_ROTATION_LR = 0.001
DEFAULT_TRAIN_APPEARANCE_EMBEDDINGS_LR = 0.001
DEFAULT_TRAIN_APPEARANCE_NETWORK_LR = 0.001
DEFAULT_TRAIN_GAUSSIAN_FEATURES_LR = 0.05 / 2.0
DEFAULT_TRAIN_PGSR_APPEARANCE_LR = 0.001
DEFAULT_MAX_GAUSSIANS = 6_000_000

DEFAULT_EXTRACT_N_PIVOTS = 2
DEFAULT_EXTRACT_STD_FACTOR = 3.0
DEFAULT_EXTRACT_N_BINARY_STEPS = 10
DEFAULT_EXTRACT_ISOSURFACE_VALUE = 0.0

DEFAULT_TEXTURE_N_ITER = 1000
DEFAULT_TEXTURE_LAMBDA_DSSIM = 0.2
DEFAULT_TEXTURE_LR = 0.0025
DEFAULT_TEXTURE_SH_DEGREE = 0


def run(cmd, cwd=None):
    """Run a command, streaming output to the terminal."""
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"  cwd: {cwd or os.getcwd()}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def get_mesh_path(model_path, n_pivots, postprocess):
    mesh_name = f"mesh_ours_{n_pivots}pivots"
    if postprocess:
        mesh_name += "_post"
    mesh_name += ".ply"
    return os.path.join(str(model_path), mesh_name)


def get_textured_mesh_path(model_path, mesh_path, texture_n_iter):
    base = os.path.basename(mesh_path)
    mesh_name = base.split(".")[0]
    mesh_extension = base.split(".")[1]
    i_iter = texture_n_iter - 1
    return os.path.join(str(model_path), f"{mesh_name}_texture_refined_{i_iter}.{mesh_extension}")


def build_train_cmd(args, model_path, scene_dir, passthrough_args):
    return [
        sys.executable, str(TRAIN_SCRIPT),
        "--rasterizer", "ours",
        "-s", str(scene_dir),
        "-m", str(model_path),
        "--feature_dc_lr", str(args.feature_dc_lr),
        "--feature_rest_lr", str(args.feature_rest_lr),
        "--position_lr_init", str(args.position_lr_init),
        "--position_lr_final", str(args.position_lr_final),
        "--position_lr_delay_mult", str(args.position_lr_delay_mult),
        "--position_lr_max_steps", str(args.position_lr_max_steps),
        "--opacity_lr", str(args.opacity_lr),
        "--scaling_lr", str(args.scaling_lr),
        "--rotation_lr", str(args.rotation_lr),
        "--appearance_embeddings_lr", str(args.appearance_embeddings_lr),
        "--appearance_network_lr", str(args.appearance_network_lr),
        "--gaussian_features_lr", str(args.gaussian_features_lr),
        "--pgsr_appearance_lr", str(args.pgsr_appearance_lr),
        "--exposure_compensation",
        "--data_device", "cpu",
        "--iterations", str(args.iterations),
        "--sh_degree", str(args.sh_degree),
        "--N_max_gaussians", str(args.max_gaussians),
        "--densify_until_iter", str(args.densify_until_iter),
        "--densify_grad_threshold", str(args.densify_grad_threshold),
        "--lambda_depth_normal", str(args.lambda_depth_normal),
        "--multiview_factor", str(args.multiview_factor),
        *(["-r", str(args.resolution)] if args.resolution is not None else []),
        *passthrough_args,
    ]


def build_extract_cmd(args, model_path, scene_dir, extract_iteration):
    cmd = [
        sys.executable, str(EXTRACT_SCRIPT),
        "--sdf_mode", "ours",
        "--rasterizer", "ours",
        "--dtype", "int32",
        "-s", str(scene_dir),
        "-m", str(model_path),
        "--n_pivots", str(args.n_pivots),
        "--std_factor", str(args.std_factor),
        "--n_binary_steps", str(args.n_binary_steps),
        "--isosurface_value", str(args.isosurface_value),
        "--iteration", str(extract_iteration),
        "--use_valid_mask",
        "--data_device", "cpu",
    ]
    if args.use_searched_pivots:
        cmd.append("--use_searched_pivots")
    if args.use_smallest_axis_as_normal:
        cmd.append("--use_smallest_axis_as_normal")
    if args.postprocess:
        cmd.append("--postprocess")
    if args.filter_large_edges:
        cmd.append("--filter_large_edges")
    if args.resolution is not None:
        cmd += ["-r", str(args.resolution)]
    return cmd


def build_texture_cmd(args, model_path, scene_dir, mesh_path, extract_iteration):
    cmd = [
        sys.executable, str(TEXTURE_SCRIPT),
        "--rasterizer", "ours",
        "-s", str(scene_dir),
        "-m", str(model_path),
        "--mesh", mesh_path,
        "--iteration", str(extract_iteration),
        "--n_iter", str(args.texture_n_iter),
        "--lambda_dssim", str(args.texture_lambda_dssim),
        "--lr", str(args.texture_lr),
        "--sh_degree_for_texturing", str(args.texture_sh_degree),
    ]
    if args.resolution is not None:
        cmd += ["-r", str(args.resolution)]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Gaussian Wrapping (ours) training, pivot-based mesh extraction, "
            "and texture refinement end-to-end."
        ),
    )
    parser.add_argument("scene_dir", type=Path, help="COLMAP scene directory (contains images/ and sparse/0/)")
    parser.add_argument("output_dir", type=Path, help="Root output directory (used as the model_path for all stages)")
    parser.add_argument("-r", "--resolution", type=int, default=None, help="Image resolution override forwarded to all stages")

    # Training fidelity
    parser.add_argument("--iterations", type=int, default=30_000, help="Training iterations")
    parser.add_argument("--sh_degree", type=int, default=3, help="Max spherical harmonics degree")
    parser.add_argument("--max_gaussians", type=int, default=DEFAULT_MAX_GAUSSIANS, help="Maximum number of gaussians")
    parser.add_argument("--densify_until_iter", type=int, default=15_000, help="Iteration to stop densifying")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002,
                        help="Lower → denser Gaussian cloud (more detail, more memory)")
    parser.add_argument("--lambda_depth_normal", type=float, default=0.05, help="Depth-normal consistency loss weight")
    parser.add_argument("--multiview_factor", type=float, default=1.0, help="Multi-view loss factor")

    # Training learning rates
    parser.add_argument("--position_lr_init", type=float, default=DEFAULT_TRAIN_POSITION_LR_INIT)
    parser.add_argument("--position_lr_final", type=float, default=DEFAULT_TRAIN_POSITION_LR_FINAL)
    parser.add_argument("--position_lr_delay_mult", type=float, default=DEFAULT_TRAIN_POSITION_LR_DELAY_MULT)
    parser.add_argument("--position_lr_max_steps", type=int, default=DEFAULT_TRAIN_POSITION_LR_MAX_STEPS)
    parser.add_argument("--feature_dc_lr", type=float, default=DEFAULT_TRAIN_FEATURE_DC_LR)
    parser.add_argument("--feature_rest_lr", type=float, default=DEFAULT_TRAIN_FEATURE_REST_LR)
    parser.add_argument("--opacity_lr", type=float, default=DEFAULT_TRAIN_OPACITY_LR)
    parser.add_argument("--scaling_lr", type=float, default=DEFAULT_TRAIN_SCALING_LR)
    parser.add_argument("--rotation_lr", type=float, default=DEFAULT_TRAIN_ROTATION_LR)
    parser.add_argument("--appearance_embeddings_lr", type=float, default=DEFAULT_TRAIN_APPEARANCE_EMBEDDINGS_LR)
    parser.add_argument("--appearance_network_lr", type=float, default=DEFAULT_TRAIN_APPEARANCE_NETWORK_LR)
    parser.add_argument("--gaussian_features_lr", type=float, default=DEFAULT_TRAIN_GAUSSIAN_FEATURES_LR)
    parser.add_argument("--pgsr_appearance_lr", type=float, default=DEFAULT_TRAIN_PGSR_APPEARANCE_LR)

    # Mesh extraction fidelity
    parser.add_argument("--extract_iteration", type=int, default=None,
                        help="Checkpoint iteration to load for extraction and texture refinement. Defaults to --iterations.")
    parser.add_argument("--n_pivots", type=int, default=DEFAULT_EXTRACT_N_PIVOTS, help="Number of pivots for mesh extraction")
    parser.add_argument("--std_factor", type=float, default=DEFAULT_EXTRACT_STD_FACTOR,
                        help="Pivot offset scale relative to Gaussian extent during mesh extraction")
    parser.add_argument("--use_searched_pivots", action=BooleanOptionalAction, default=False,
                        help="Refine extraction pivots by searching along the normal direction")
    parser.add_argument("--use_smallest_axis_as_normal", action=BooleanOptionalAction, default=False,
                        help="Use the Gaussian's smallest axis as the extraction normal instead of learned normals")
    parser.add_argument("--n_binary_steps", type=int, default=DEFAULT_EXTRACT_N_BINARY_STEPS, help="Binary search refinement steps")
    parser.add_argument("--isosurface_value", type=float, default=DEFAULT_EXTRACT_ISOSURFACE_VALUE, help="Isosurface value")
    parser.add_argument("--postprocess", action=BooleanOptionalAction, default=True,
                        help="Postprocess the extracted mesh (default: on)")
    parser.add_argument("--filter_large_edges", action=BooleanOptionalAction, default=True,
                        help="Filter triangles with large edges (default: on)")

    # Texture refinement fidelity
    parser.add_argument("--texture_n_iter", type=int, default=DEFAULT_TEXTURE_N_ITER, help="Texture refinement iterations")
    parser.add_argument("--texture_lambda_dssim", type=float, default=DEFAULT_TEXTURE_LAMBDA_DSSIM, help="SSIM loss weight for texture refinement")
    parser.add_argument("--texture_lr", type=float, default=DEFAULT_TEXTURE_LR, help="Learning rate for texture refinement")
    parser.add_argument("--texture_sh_degree", type=int, default=DEFAULT_TEXTURE_SH_DEGREE, help="SH degree used while baking the texture")

    args, passthrough_args = parser.parse_known_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    extract_iteration = args.extract_iteration or args.iterations
    mesh_path = get_mesh_path(output_dir, args.n_pivots, args.postprocess)
    textured_mesh_path = get_textured_mesh_path(output_dir, mesh_path, args.texture_n_iter)

    logger.info("=" * 60)
    logger.info("Gaussian Wrapping Reconstruction Pipeline")
    logger.info(f"  Scene:  {scene_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Extract iteration:  {extract_iteration}")
    logger.info(f"  Extracted mesh:     {mesh_path}")
    logger.info(f"  Textured mesh:      {textured_mesh_path}")
    if passthrough_args:
        logger.info(f"  Forwarding extra train.py args: {' '.join(passthrough_args)}")
    logger.info("=" * 60)

    # Step 1: Training
    logger.info("Step 1/3: Training Gaussian Wrapping (ours)")
    t0 = time.time()
    run(build_train_cmd(args, output_dir, scene_dir, passthrough_args))
    train_time = time.time() - t0
    logger.info(f"Training completed in {train_time:.1f}s")

    # Step 2: Pivot-based mesh extraction
    logger.info("Step 2/3: Extracting mesh (pivot-based)")
    t0 = time.time()
    run(build_extract_cmd(args, output_dir, scene_dir, extract_iteration))
    extract_time = time.time() - t0
    logger.info(f"Mesh extraction completed in {extract_time:.1f}s")

    # Step 3: Texture refinement
    logger.info("Step 3/3: Refining texture")
    t0 = time.time()
    run(build_texture_cmd(args, output_dir, scene_dir, mesh_path, extract_iteration))
    texture_time = time.time() - t0
    logger.info(f"Texture refinement completed in {texture_time:.1f}s")

    total_time = train_time + extract_time + texture_time
    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info(f"  Train time:    {train_time:.1f}s")
    logger.info(f"  Extract time:  {extract_time:.1f}s")
    logger.info(f"  Texture time:  {texture_time:.1f}s")
    logger.info(f"  Total:         {total_time:.1f}s")
    logger.info(f"  Output:        {output_dir}")
    logger.info(f"  Extracted mesh: {mesh_path}")
    logger.info(f"  Textured mesh:  {textured_mesh_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
