# Cluster execution (SLURM / HiPerGator)

Batch job scripts live in `slurm/`. They replace the manual
`salloc -N1 -t8:00:00 --cpus-per-task 32 --ntasks-per-node=1 --partition=hpg-rtx6000 --gpus=1`
workflow.

## Site facts

- Slurm 25.11.6. Account `arthur.porto` (an `arthur.porto-phenomics` association also exists).
- Partitions in use: `hpg-rtx6000` (45 nodes, `Gres=gpu:rtx_pro_6000:8`, MaxTime 14 days) and
  `hpg-b200` (60 nodes, `Gres=gpu:b200:8`). Others exist: `hpg-default`, `hpg-milan`,
  `hpg-turin`, `bigmem`, `hpg-dev` (12 h).
- Prebuilt conda envs at `/blue/arthur.porto/srizvi63.gatech/conda/`: `augenblick_rtx_6000`,
  `augenblick_b200`, `gaussian_wrapping`, `meshroom`. These are *not* the `augenblick` env named
  in the README quick start.

## Files

| File | Role |
|------|------|
| `slurm/common.sh` | Sourced by every script: GPU switch, module loads, conda activate, banner |
| `slurm/template.sbatch` | Copy-and-edit starting point |
| `slurm/vggt_sfm.sbatch` | VGGT -> COLMAP (`--input_dir`/`--output_dir`, `--use_ba`) |
| `slurm/colmap_sfm.sbatch` | Masked COLMAP SfM (`--input_dir`/`--output_dir`) |
| `slurm/recon.sbatch` | `BACKEND=2dgs\|sugar\|pgsr\|gw`, two positionals |

## GPU switch

`GPU=rtx6000` (default) or `GPU=b200` selects env, CUDA module, and `TORCH_CUDA_ARCH_LIST`:

| GPU | Env | CUDA module | Arch |
|-----|-----|-------------|------|
| `rtx6000` | `augenblick_rtx_6000` | `cuda/13.0.2` | 12.0 |
| `b200` | `augenblick_b200` | `cuda/12.8` | 10.0 |

Arch strings mirror `GPU_ARCH` in `scripts/setup_rtx_pro_6000.sh` / `setup_b200.sh`. Note
`scripts/auto_setup.sh` maps compute cap 8.9 to `setup_l40s.sh` and comments it "L40S / RTX 6000
Ada" — that is a *different* card from the Blackwell RTX Pro 6000 on `hpg-rtx6000`. `GPU` does
not change the partition; pass `--partition` too.

## Deliberate choices

- **No `module load colmap/3.11`, no `export -f colmap`.** Every SfM path goes through the
  `pycolmap` Python API from the conda env and never shells out, so no COLMAP binary is needed.
- `module purge` first, since a batch shell inherits no `~/.bashrc`.
- `conda activate` requires `source "$(conda info --base)/etc/profile.d/conda.sh"` first in a
  non-interactive shell.
- `--mem=128gb` set explicitly (the interactive `salloc` let it default).
- Logs to `slurm/logs/%x-%j.{out,err}`, gitignored.
