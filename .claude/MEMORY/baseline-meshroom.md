# `baseline/` — Baseline Wrappers

Thin wrappers around third-party photogrammetry tools used as qualitative comparisons. They
match the logging style of `pipeline/reconstruction/run_2dgs.py` (banner, `subprocess.run`,
total-time summary).

## Meshroom (AliceVision)

```bash
python baseline/benchmark_meshroom.py <input_images> <output_dir> \
    [--save_file <path.mg>] [--meshroom_root <path>]
```

Resolves `meshroom_batch` from `$MESHROOM_ROOT` (or `--meshroom_root`), invokes the hardcoded
`photogrammetry` pipeline template, prepends `$MESHROOM_ROOT` to `PYTHONPATH` so its Python
modules resolve, and logs total runtime. Masks are supported via
`--paramOverrides FeatureExtraction:masksFolder=<path>` (see `ff0198b`); texture output is
forced to `.png` (`afdc84b`), and fidelity knobs were added in `87fcdb5`.

Installation and env-var setup: `meshroom-setup.md` (repo root) — it includes a dedicated
`meshroom` conda env for the batch CLI.

RealityScan is the other (commercial) qualitative baseline, run outside this repo.

## Reference runtimes

138 images at 6240x4160 with masks, NVIDIA A100 PCIe 40 GB (end-to-end):
COLMAP+SuGaR ~80 min · COLMAP+2DGS ~40 min · COLMAP+PGSR ~70 min · COLMAP+GW ~65 min ·
Meshroom ~60 min.

SfM only, same scene on a B200 80 GB: COLMAP ~10 min · VGGT ~3 min · VGGT+BA ~15 min.
