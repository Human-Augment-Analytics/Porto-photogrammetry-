# Environment, Install, and GPU Notes

## Install (canonical)

There is **no `environment.yml`** any more (it was removed; older docs referencing it are stale).
Create the env yourself, then run the per-GPU installer:

```bash
conda create --name augenblick python=3.10
conda activate augenblick
git submodule update --init --recursive
bash scripts/auto_setup.sh          # detects GPU via nvidia-smi, dispatches to a wrapper
```

`scripts/` layout (added in `0ed266e`, which also deleted the old sbatch/`setup_sugar_b200.sh` scripts in `b1fccc1`):

| Script | GPU | sm | CUDA module | Torch |
|--------|-----|----|-------------|-------|
| `auto_setup.sh` | detect + dispatch | — | — | — |
| `setup_l40s.sh` | L40S / RTX 6000 Ada | 8.9 | `cuda/12.1.1` | 2.3.1 / cu121 |
| `setup_a100.sh` | A100 | 8.0 | `cuda/12.1.1` | 2.3.1 / cu121 |
| `setup_h100.sh` | H100 / H200 | 9.0 | `cuda/12.1.1` | 2.3.1 / cu121 |
| `setup_b200.sh` | B200 (original target) | 10.0 | `cuda/12.8` | 2.9.1 / cu130 |

Wrappers only export `GPU_LABEL / GPU_ARCH / CUDA_MODULE / TORCH_SPEC / TORCH_INDEX_URL` and
`exec` into `scripts/setup_common.sh`, which does all the work (7 stages + an import
verification block). `auto_setup.sh` exits 2 with a copy-this-wrapper hint on an unknown
compute capability.

### `setup_common.sh` knobs

- `BACKENDS` (default `"sugar 2dgs pgsr gw"`) — subset of CUDA rasterizers to build.
- `SKIP_TETRA=1` — skip the fragile CGAL `tetra_triangulation` build (GW pivot extraction only).
- `PYTORCH3D_WHEEL=<url>` — install a prebuilt pytorch3d instead of the source build.

### Gotchas encoded in `setup_common.sh`

- **numpy must stay `<2`**: a `PIP_CONSTRAINT` temp file pins it globally, because numpy 2.x
  breaks the pinned scipy/scikit-learn C-ABI and any transitive dep can pull it in.
- **Stale `build/` dirs**: each rasterizer build does `rm -rf <pkg>/build <pkg>/*.egg-info`
  first — leftover objects from another GPU arch are silently reused and ignore
  `TORCH_CUDA_ARCH_LIST`.
- **No concurrent setups against one checkout** — they race on those `build/` dirs. Use a
  separate checkout per parallel build.
- `TORCH_CUDA_ARCH_LIST` is set to the wrapper's `GPU_ARCH`; import checks only validate the
  arch of the node the script ran on.
- nvdiffrast and tetra_triangulation failures are `WARN`-only (non-fatal).

### Manual setup

The manual pip sequence the scripts wrap is in [README.md](../../README.md) ("Manual setup").
VGGT is installed editable from `src/libs/vggt` (the package root with `pyproject.toml`; the
importable package is the inner `src/libs/vggt/vggt/`). An old editable install made from `src/`
must be redone from `src/libs/vggt`.

## Submodules

`.gitmodules` lists exactly three: `src/libs/light_glue`, `src/libs/pytorch3d`,
`src/libs/gaussian_wrapping/submodules/Depth-Anything-V2`.
**`src/libs/sugar` is not a submodule** (older docs claimed it was); it is vendored in-tree, as are
`src/libs/2dgs`, `src/libs/pgsr`, `src/libs/gaussian_wrapping`, `src/libs/vggt`.

## GPU notes

- Mixed precision: bfloat16 on Ampere+ (SM >= 8.0), float16 otherwise.
- Blackwell (B200, SM >= 10.0): `torch.compile(mode="max-autotune")` applied automatically.
- VGGT wants >= 80 GB VRAM for large scenes; COLMAP and the reconstruction backends run on
  32–40 GB (validated on an A100 PCIe 40 GB).
- Crash dumps (`core.colmap-*.ufhpc.*`) in the repo root are HPC artifacts — safe to delete.
