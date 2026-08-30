#!/bin/bash
# Copy this file, edit the command block at the bottom, then: sbatch slurm/<yours>.sbatch
#SBATCH --job-name=augenblick
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
##SBATCH --partition=hpg-b200          # alternative; pair with GPU=b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=128gb
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=srizvi63@gatech.edu

source "$(dirname "$0")/common.sh"

SCENE="data/CHANGE_ME"
OUT="output/CHANGE_ME"

# ================= EDIT BELOW: your command =================
python pipeline/sfm/run_vggt_to_colmap.py \
    --input_dir  "$SCENE" \
    --output_dir "$OUT"
# ============================================================
