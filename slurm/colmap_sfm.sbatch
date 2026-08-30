#!/bin/bash
# Mask-aware COLMAP SfM. Usage: sbatch slurm/colmap_sfm.sbatch <scene_dir> <output_dir>
# Uses pycolmap from the conda env; no colmap module is loaded.
#SBATCH --job-name=colmap-sfm
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=srizvi63@gatech.edu

source "$(dirname "$0")/common.sh"

if [ $# -lt 2 ]; then
    echo "usage: sbatch $0 <scene_dir> <output_dir> [extra flags]" >&2
    exit 1
fi
SCENE="$1"; OUT="$2"; shift 2

python pipeline/sfm/run_masked_colmap.py \
    --input_dir  "$SCENE" \
    --output_dir "$OUT" "$@"
