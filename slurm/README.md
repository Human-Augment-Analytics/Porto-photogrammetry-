# SLURM job scripts

Batch equivalents of the manual `salloc` workflow, so runs survive a dropped SSH session.

## Submit

```bash
# One array task per scene found under DATA_ROOT; add --array=N for a single scene.
sbatch slurm/vggt_sfm.sbatch
sbatch slurm/colmap_sfm.sbatch
BACKEND=2dgs SFM=vggt sbatch slurm/recon.sbatch
```

`recon.sbatch` accepts `BACKEND=2dgs|sugar|pgsr|gw` and `SFM=<name>` to pick which SfM result
feeds it. Scenes are discovered under `DATA_ROOT` and sorted, so an array index maps to the same
scene across submissions. Extra flags are forwarded to the `augenblick` CLI.

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
- **COLMAP is intentionally not module-loaded.** Every SfM path drives the `pycolmap` Python API
  from the conda env, so no COLMAP binary is needed anywhere.
- **The jobs call the bare `augenblick` console script**, so the package must be installed
  (`pip install -e . --no-deps --no-build-isolation`) into each per-GPU conda env.
- **Never run `scripts/setup_*.sh` from two concurrent jobs against one checkout** — they race on
  the same `build/` dirs and silently reuse stale artifacts. Use a separate checkout per parallel
  build. Training jobs sharing a checkout are fine.
