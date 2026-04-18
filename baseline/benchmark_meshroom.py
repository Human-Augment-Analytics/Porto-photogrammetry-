#!/usr/bin/env python3
"""
Run the Meshroom (AliceVision) photogrammetry baseline via meshroom_batch.

Input: a directory of source images.
Output: Meshroom reconstruction artifacts in the output directory and an
optional save file capturing the pipeline graph.

Usage:
    python baseline/benchmark_meshroom.py <input_images> <output_dir> \
        [--pipeline photogrammetry] [--save_file <path>] \
        [--meshroom_root <path>]
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
        description="Run the Meshroom batch photogrammetry pipeline.",
    )
    parser.add_argument("input_images", type=Path, help="Directory of input images")
    parser.add_argument("output_dir", type=Path, help="Meshroom output directory")
    parser.add_argument("--save_file", type=Path, default=None, help="Path to save the pipeline graph (.mg)")
    parser.add_argument(
        "--meshroom_root",
        type=Path,
        default=None,
        help="Path to MESHROOM_ROOT (defaults to $MESHROOM_ROOT env var)",
    )

    args = parser.parse_args()
    input_images = args.input_images.resolve()
    output_dir = args.output_dir.resolve()

    meshroom_root = args.meshroom_root or os.environ.get("MESHROOM_ROOT")
    if not meshroom_root:
        logger.error("MESHROOM_ROOT is not set. Pass --meshroom_root or export MESHROOM_ROOT.")
        sys.exit(1)
    meshroom_batch = Path(meshroom_root) / "bin" / "meshroom_batch"
    if not meshroom_batch.exists():
        logger.error(f"meshroom_batch not found at {meshroom_batch}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Meshroom Baseline Pipeline")
    logger.info(f"  Input:    {input_images}")
    logger.info(f"  Output:   {output_dir}")
    logger.info(f"  Pipeline: {args.pipeline}")
    logger.info("=" * 60)

    cmd = [
        sys.executable, str(meshroom_batch),
        "-i", str(input_images),
        "-o", str(output_dir),
        "-p", "photogrammetry",
    ]
    if args.save_file is not None:
        cmd += ["-s", str(args.save_file.resolve())]

    t0 = time.time()
    run(cmd)
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Output:     {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
