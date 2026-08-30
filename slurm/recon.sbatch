#!/bin/bash
# Gaussian reconstruction. Usage: BACKEND=2dgs sbatch slurm/recon.sbatch <scene_dir> <output_dir>
# BACKEND selects pipeline/reconstruction/run_<backend>.py (2dgs|sugar|pgsr|gw).
#SBATCH --job-name=recon
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

BACKEND="${BACKEND:-2dgs}"
case "$BACKEND" in
    2dgs|sugar|pgsr|gw) ;;
    *) echo "ERROR: unknown BACKEND='$BACKEND' (valid: 2dgs, sugar, pgsr, gw)" >&2; exit 2 ;;
esac

if [ $# -lt 2 ]; then
    echo "usage: BACKEND=$BACKEND sbatch $0 <scene_dir> <output_dir> [extra flags]" >&2
    exit 1
fi
SCENE="$1"; OUT="$2"; shift 2

# All four reconstruction wrappers take two positionals.
python "pipeline/reconstruction/run_${BACKEND}.py" "$SCENE" "$OUT" "$@"
