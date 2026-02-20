#!/usr/bin/env python3
"""
End-to-end VGGT -> SuGaR pipeline.

Orchestrates:
  1. VGGT inference (camera poses + 3D points)
  2. COLMAP format export (bypasses COLMAP SfM)
  3. 3D Gaussian Splatting training
  4. SuGaR mesh extraction + texture export

Usage:
  # Full pipeline from images
  python run_sugar_pipeline.py /path/to/data --quality medium

  # Skip VGGT (reuse existing predictions)
  python run_sugar_pipeline.py /path/to/data --predictions_pkl output/vggt_predictions.pkl --skip_vggt

  # Skip to SuGaR (reuse existing 3DGS checkpoint)
  python run_sugar_pipeline.py /path/to/data --skip_vggt --skip_3dgs --gs_checkpoint output/gs_model
"""

import argparse
import os
import sys
import subprocess
import time
import json

QUALITY_PRESETS = {
    "draft": {
        "gs_iterations": 7_000,
        "sugar_regularization": "dn_consistency",
        "sugar_high_poly": True,
        "sugar_refinement_time": "short",
        "sugar_export_obj": True,
        "point_conf_threshold": 60.0,
        "max_points3d": 100_000,
    },
    "medium": {
        "gs_iterations": 15_000,
        "sugar_regularization": "dn_consistency",
        "sugar_high_poly": True,
        "sugar_refinement_time": "medium",
        "sugar_export_obj": True,
        "point_conf_threshold": 70.0,
        "max_points3d": 200_000,
    },
    "high": {
        "gs_iterations": 30_000,
        "sugar_regularization": "dn_consistency",
        "sugar_high_poly": True,
        "sugar_refinement_time": "long",
        "sugar_export_obj": True,
        "point_conf_threshold": 80.0,
        "max_points3d": 500_000,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end VGGT to SuGaR pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir", help="Input directory with images/ subdirectory"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (defaults to input_dir/sugar_output)",
    )
    parser.add_argument(
        "--quality",
        choices=["draft", "medium", "high"],
        default="medium",
        help="Quality preset",
    )
    parser.add_argument(
        "--predictions_pkl",
        default=None,
        help="Skip VGGT inference, use existing predictions .pkl",
    )
    parser.add_argument(
        "--sugar_repo",
        default=None,
        help="Path to SuGaR repository (auto-detected from project root)",
    )
    parser.add_argument(
        "--gs_repo",
        default=None,
        help="Path to gaussian-splatting repository (auto-detected from project root)",
    )
    parser.add_argument(
        "--gs_checkpoint",
        default=None,
        help="Path to existing 3DGS checkpoint (skips 3DGS training)",
    )
    parser.add_argument("--skip_vggt", action="store_true", help="Skip VGGT inference")
    parser.add_argument("--skip_3dgs", action="store_true", help="Skip 3DGS training")
    parser.add_argument(
        "--skip_sugar", action="store_true", help="Skip SuGaR extraction"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU device ID"
    )
    return parser.parse_args()


def find_repo(name, project_root):
    """Search for a repository in common locations relative to the project."""
    candidates = [
        os.path.join(project_root, name),
        os.path.join(project_root, "..", name),
        os.path.join(project_root, "src", name),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    return None


def run_vggt_inference(input_dir, output_dir, preset):
    """Stage 1: Run VGGT and export COLMAP format."""
    print("\n" + "=" * 60)
    print("STAGE 1: VGGT Inference + COLMAP Export")
    print("=" * 60)

    pipeline_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = [
        sys.executable,
        os.path.join(pipeline_dir, "run_pipeline.py"),
        input_dir,
        "--output_dir", output_dir,
        "--save_for_sugar",
        "--skip_glb",
        "--max_points3d", str(preset["max_points3d"]),
        "--point_conf_threshold", str(preset["point_conf_threshold"]),
        "--colmap_output_dir", os.path.join(output_dir, "colmap"),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return os.path.join(output_dir, "colmap")


def run_vggt_from_pkl(predictions_pkl, output_dir, images_dir, preset):
    """Stage 1 (alt): Convert existing predictions to COLMAP format."""
    print("\n" + "=" * 60)
    print("STAGE 1: COLMAP Export from Existing Predictions")
    print("=" * 60)

    # Import directly to avoid subprocess overhead
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, pipeline_dir)
    from vggt_to_colmap import convert_vggt_to_colmap

    colmap_dir = os.path.join(output_dir, "colmap")
    convert_vggt_to_colmap(
        predictions_path=predictions_pkl,
        output_dir=colmap_dir,
        images_source_dir=images_dir,
        conf_threshold_percentile=preset["point_conf_threshold"],
        max_points=preset["max_points3d"],
    )
    return colmap_dir


def run_gaussian_splatting(colmap_dir, gs_output_dir, gs_repo, iterations, gpu_id):
    """Stage 2: Train 3D Gaussian Splatting."""
    print("\n" + "=" * 60)
    print("STAGE 2: 3D Gaussian Splatting Training")
    print("=" * 60)

    train_script = os.path.join(gs_repo, "train.py")
    if not os.path.exists(train_script):
        raise FileNotFoundError(
            f"3DGS train.py not found at {train_script}. "
            f"Run scripts/setup_sugar_b200.sh first."
        )

    cmd = [
        sys.executable,
        train_script,
        "-s", colmap_dir,
        "-m", gs_output_dir,
        "--iterations", str(iterations),
    ]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
    return gs_output_dir


def run_sugar_extraction(colmap_dir, gs_checkpoint, sugar_repo, sugar_output_dir, preset, gpu_id):
    """Stage 3: Extract mesh with SuGaR."""
    print("\n" + "=" * 60)
    print("STAGE 3: SuGaR Mesh Extraction")
    print("=" * 60)

    train_script = os.path.join(sugar_repo, "train_full_pipeline.py")
    if not os.path.exists(train_script):
        # Fallback: some SuGaR versions use different entry points
        train_script = os.path.join(sugar_repo, "train.py")

    if not os.path.exists(train_script):
        raise FileNotFoundError(
            f"SuGaR training script not found in {sugar_repo}. "
            f"Run scripts/setup_sugar_b200.sh first."
        )

    cmd = [
        sys.executable,
        train_script,
        "-s", colmap_dir,
        "-c", gs_checkpoint,
        "-o", sugar_output_dir,
        "-r", preset["sugar_regularization"],
        "--refinement_time", preset["sugar_refinement_time"],
    ]

    if preset["sugar_high_poly"]:
        cmd.extend(["--high_poly", "True"])

    if preset["sugar_export_obj"]:
        cmd.extend(["--export_obj", "True"])

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
    return sugar_output_dir


def main():
    args = parse_args()
    preset = QUALITY_PRESETS[args.quality]
    output_dir = args.output_dir or os.path.join(args.input_dir, "sugar_output")
    os.makedirs(output_dir, exist_ok=True)

    # Find project root (2 levels up from this script)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Auto-detect repos
    gs_repo = args.gs_repo or find_repo("gaussian-splatting", project_root)
    sugar_repo = args.sugar_repo or find_repo("SuGaR", project_root)

    pipeline_start = time.time()
    print(f"VGGT -> SuGaR Pipeline")
    print(f"  Quality:    {args.quality}")
    print(f"  Input:      {args.input_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  GPU:        {args.gpu_id}")
    if gs_repo:
        print(f"  3DGS repo:  {gs_repo}")
    if sugar_repo:
        print(f"  SuGaR repo: {sugar_repo}")

    # Save config
    config = {
        "quality": args.quality,
        "preset": preset,
        "input_dir": os.path.abspath(args.input_dir),
        "output_dir": os.path.abspath(output_dir),
        "gpu_id": args.gpu_id,
    }
    with open(os.path.join(output_dir, "pipeline_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- Stage 1: VGGT -> COLMAP ---
    colmap_dir = os.path.join(output_dir, "colmap")
    if not args.skip_vggt:
        if args.predictions_pkl:
            images_dir = os.path.join(args.input_dir, "images")
            colmap_dir = run_vggt_from_pkl(
                args.predictions_pkl, output_dir, images_dir, preset
            )
        else:
            colmap_dir = run_vggt_inference(args.input_dir, output_dir, preset)
    else:
        if not os.path.exists(os.path.join(colmap_dir, "sparse", "0", "cameras.bin")):
            print(f"Warning: COLMAP data not found at {colmap_dir}")

    # --- Stage 2: 3DGS Training ---
    gs_output_dir = args.gs_checkpoint or os.path.join(output_dir, "gs_model")
    if not args.skip_3dgs and not args.gs_checkpoint:
        if not gs_repo:
            print("ERROR: gaussian-splatting repo not found.")
            print("Run: scripts/setup_sugar_b200.sh")
            sys.exit(1)
        gs_output_dir = run_gaussian_splatting(
            colmap_dir, gs_output_dir, gs_repo, preset["gs_iterations"], args.gpu_id
        )

    # --- Stage 3: SuGaR Mesh Extraction ---
    sugar_output_dir = os.path.join(output_dir, "sugar_mesh")
    if not args.skip_sugar:
        if not sugar_repo:
            print("ERROR: SuGaR repo not found.")
            print("Run: scripts/setup_sugar_b200.sh")
            sys.exit(1)
        run_sugar_extraction(
            colmap_dir, gs_output_dir, sugar_repo, sugar_output_dir, preset, args.gpu_id
        )

    # --- Summary ---
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  COLMAP:     {colmap_dir}")
    print(f"  3DGS:       {gs_output_dir}")
    print(f"  SuGaR mesh: {sugar_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
