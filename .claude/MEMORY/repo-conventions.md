# Repo Conventions and Documentation Rules

## Documentation layout rule

`.claude/CLAUDE.md` stays **succinct**: orientation, entry points, hard-won gotchas that change
what you type, and pointers into `.claude/MEMORY/`. Anything long — full flag lists, per-backend
internals, parameter tables, algorithm walk-throughs — lives in a topic file under
`.claude/MEMORY/` and is linked from CLAUDE.md's index table.

When adding documentation: put the detail in the right `MEMORY/` file (or add a new one and
index it in CLAUDE.md); add to CLAUDE.md itself only if it changes how someone invokes or
navigates the repo.

## Repository layout

```
pipeline/       Canonical entry points (preparation/, sfm/, reconstruction/)
baseline/       Meshroom wrapper
scripts/        Per-GPU installers (auto_setup.sh + setup_{l40s,a100,h100,b200,common}.sh)
src/            Backends: vggt/, sugar/, 2dgs/, pgsr/, gaussian_wrapping/, light_glue/, pytorch3d/
assets/         README result grids
data/, output/  Local scene data and run outputs (not for commit)
```

## Legacy / non-canonical code

- **`src/pipeline/`** — superseded first-generation scripts (`run_vggt.py`,
  `vggt_to_colmap.py`, `new_vggt_converter.py`, `quick_fix.py`, `run_sugar_pipeline.py`,
  `vggt_to_ingp.py`, `vggt_to_neus2.py`, `example_usage.py`, `conversion_utils.py`). Use
  `pipeline/` for new experiments. `vggt_to_neus2.py` / `vggt_to_ingp.py` are the only place
  the NeuS2 / instant-ngp exports live.
- **`src/utils/visual_util.py`** — standalone visualisation helper.
- Untracked scratch in the repo root (`commands.md`, `install_commands.md`, `notebooks/`) is
  personal working material, not part of the documented pipeline.

## Naming and ID conventions

- COLMAP IDs (image / camera / point3D) are **1-indexed**; VGGT batch index → COLMAP ID has a
  `+1` offset throughout.
- COLMAP mask files must be named `<image_name>.png` (`foo.jpg.png`, not `foo.png`) — hence the
  `masks_colmap/` symlink dirs in `run_masked_colmap.py` and `run_turntable_to_colmap.py`.
- `masks/` convention elsewhere in the repo: binary PNG named after the image *stem*, white =
  foreground.
- Turntable camera grouping keys off `--camera_regex` (default `camera\d+`) and orders frames by
  the **last** integer in the filename.

## Commit / branch conventions

Branches are `<author>/<topic>` (e.g. `syed/spring-26-experiments`, `ihor/turntable`) and land on
`main` via PR merges. Commit subjects are imperative and short ("Add masked colmap, Improved
turntable prior").
