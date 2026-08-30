# SLURM job scripts

Batch equivalents of the manual `salloc` workflow, so runs survive a dropped SSH session.

## Submit

```bash
sbatch slurm/vggt_sfm.sbatch    data/my_scene output/my_scene_sfm
sbatch slurm/colmap_sfm.sbatch  data/my_scene output/my_scene_sfm
BACKEND=2dgs sbatch slurm/recon.sbatch output/my_scene_sfm output/my_scene_2dgs
```

`recon.sbatch` accepts `BACKEND=2dgs|sugar|pgsr|gw`. Extra flags after the two directories are
forwarded to the underlying script.

For anything not covered, copy `template.sbatch` and edit its command block.

## Picking a GPU

`GPU=rtx6000` (default) or `GPU=b200` selects the conda env, CUDA module, and arch string in
`common.sh`. It does **not** change the partition — override that too:

```bash
GPU=b200 sbatch --partition=hpg-b200 slurm/vggt_sfm.sbatch data/my_scene output/my_scene_sfm
```

VGGT wants >= 80 GB VRAM on large scenes, so `hpg-b200` is often the right call there. The
reconstruction backends run fine on `hpg-rtx6000`.

## Logs and monitoring

Logs land in `slurm/logs/<job-name>-<job-id>.{out,err}` (gitignored). Each starts with a banner
naming the node, GPU target, conda env, and repo commit.

```bash
squeue -u $USER
scancel <jobid>
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Gotchas

- **`~/.bashrc` is not sourced in a batch job.** `common.sh` loads the modules explicitly; do not
  assume your interactive environment carries over.
- **COLMAP is intentionally not module-loaded.** `run_masked_colmap.py` drives everything through
  the `pycolmap` Python API from the conda env. Only the shell entry point `run_colmap.sh` needs a
  real binary, and it takes `--colmap <path>` if you ever need it.
- **Never run `scripts/setup_*.sh` from two concurrent jobs against one checkout** — they race on
  the same `build/` dirs and silently reuse stale artifacts. Use a separate checkout per parallel
  build. Training jobs sharing a checkout are fine.
