# Plan: `src/augenblick` — an extensible SfM + reconstruction package

## How to use this document

This is an implementation spec, not a sketch. Follow the phases **in order**. Each phase lists
files to create, exact signatures, and a verification command that must pass before moving on.

**Absolute rules for the implementing agent:**

1. **Do not rewrite algorithms.** Every numeric routine (VGGT inference and rescaling, turntable
   rig fitting, pycolmap calls) is **moved verbatim**, body unchanged. You are reorganising code
   into classes, not improving it. If you find a bug, leave it and note it — do not fix it in the
   same change.
2. **Do not invent parameters.** Every flag, default value, and help string is copied from the
   source script. This document lists them; where it says "copy from `<file>`", open that file and
   copy exactly. Never guess a default.
3. **No GPU is available on the login node.** You cannot run VGGT or any reconstruction backend.
   Verification is limited to imports, unit tests, `--help`, and argv construction. Never claim a
   backend was tested end-to-end.
4. **Never delete a source script until its phase's verification passes.**
5. If something in this plan contradicts the code, **the code wins** — stop and report the
   discrepancy rather than guessing.

## Goal

Replace six standalone scripts under `pipeline/sfm/` and `pipeline/reconstruction/` with a package
at `src/augenblick/`, built on two abstract base classes (`SfMMethod`, `ReconstructionMethod`) and
a registry. Adding a method should mean writing one subclass, not copying a 150-line wrapper.

The research question — how SfM initialisations interact with mesh extractors — is an N x M
matrix. Today that matrix is a shell-scripting problem; a registry makes it a loop.

## Source inventory (verified against the tree)

Exactly seven first-party files are in scope. `pipeline/sfm/run_colmap.sh` was deleted in commit
`ba6d228` and is **out of scope** — it is not ported, not restored, and not mentioned again.

| Source file | Lines | Becomes | Kind |
|---|---|---|---|
| `pipeline/sfm/run_vggt_to_colmap.py` | 420 | `sfm/vggt.py` | in-process (torch) |
| `pipeline/sfm/run_masked_colmap.py` | 84 | `sfm/colmap.py` | in-process (pycolmap) |
| `pipeline/sfm/run_turntable_to_colmap.py` | 495 | `sfm/turntable.py` | in-process (pycolmap+scipy) |
| `pipeline/reconstruction/run_2dgs.py` | 140 | `reconstruction/twodgs.py` | subprocess |
| `pipeline/reconstruction/run_sugar.py` | 153 | `reconstruction/sugar.py` | subprocess |
| `pipeline/reconstruction/run_pgsr.py` | 188 | `reconstruction/pgsr.py` | subprocess |
| `pipeline/reconstruction/run_gw.py` | 279 | `reconstruction/gw.py` | subprocess |

`src/pipeline/` (legacy) and `src/libs/` (vendored backends) are **not touched**.

### Facts that constrain the design

- **`src/` is not an importable package.** No `__init__.py` outside vendored `src/libs/`, and no
  `setup.py` / `pyproject.toml` — only `requirements.txt`. Scripts work today only because they
  are run as `python pipeline/.../run_x.py` from the repo root. Phase 0 solves this.
- **The four reconstruction wrappers are near-identical**: argparse -> build argv ->
  `subprocess.run(cmd, cwd=BACKEND_DIR)` -> log timings. Their `run()` helper is byte-identical
  in all four. This is the largest duplication and the clearest win.
- **The three SfM scripts are genuinely heterogeneous** and must not be forced into one shape.
  VGGT is a torch model; masked-COLMAP is pycolmap; turntable is ~500 lines of rig-fitting math
  that *refines an existing scene* rather than creating one.
- **Per-backend divergences that must be preserved, not smoothed away:**
  - `run_pgsr.py` copies the scene and flattens `sparse/0/` -> `sparse/`.
  - `run_gw.py` runs with **no `cwd`**, uses `parse_known_args()` to forward unknown flags to the
    **training step only**, and uses `BooleanOptionalAction` (`--no-postprocess`).
  - `run_sugar.py` runs a nested vanilla-3DGS train first, and passes booleans as `--flag True`
    **string** arguments, not `store_true`.
  - Mesh output paths differ per backend; GW computes its filename from `n_pivots` +
    `postprocess` + `texture_n_iter`.
- **Backward compatibility is explicitly NOT required.** The old scripts are deleted and the SLURM
  jobs migrated in Phase 5. The CLI may be redesigned freely.

## Target layout

```
src/augenblick/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── errors.py             # AugenblickError, SceneError, BackendError, MethodNotFound
│   ├── scene.py              # Scene: the COLMAP scene contract
│   ├── config.py             # dataclass <-> argparse bridge
│   ├── registry.py           # register_sfm / register_reconstruction + lookup
│   ├── process.py            # the one run() subprocess helper
│   ├── timing.py             # StageTimer: banner + per-stage + total logging
│   └── method.py             # Method ABC + StageResult
├── sfm/
│   ├── __init__.py           # imports vggt, colmap, turntable so registration fires
│   ├── base.py               # SfMMethod, SceneRefiner
│   ├── vggt.py │ colmap.py │ turntable.py
├── reconstruction/
│   ├── __init__.py           # imports all four backends
│   ├── base.py               # ReconstructionMethod, SubprocessBackend, Stage
│   ├── twodgs.py │ sugar.py │ pgsr.py │ gw.py
└── cli/
    ├── __init__.py
    └── main.py
tests/
├── test_scene.py │ test_config.py │ test_registry.py │ test_recon_argv.py
```

---

## Phase 0 — make `src/` importable

**Create `pyproject.toml` at the repo root:**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "augenblick"
version = "0.1.0"
requires-python = ">=3.10"
# Dependencies are deliberately EMPTY. See warning below.

[tool.setuptools.packages.find]
where = ["src"]
include = ["augenblick*"]
```

> **Critical:** the `[project]` table must declare **no `dependencies`**. Per
> `.claude/MEMORY/environment-and-gpu.md`, numpy/scipy pins live in `constraints/numpy{1,2}.txt`
> and must match the GPU's torch wheel; a dependency list here would let pip resolve a
> wrong-generation numpy, which breaks **at runtime, not at install**. `scripts/auto_setup.sh`
> remains the only installer.

Create every `__init__.py` in the layout above (empty for now except where noted later).

Install: `pip install -e . --no-deps --no-build-isolation`

> `--no-deps` and `--no-build-isolation` are mandatory: they stop pip touching the carefully
> pinned environment.

**Fallback if the editable install misbehaves on the cluster:** skip `pyproject.toml` and have
callers add `src/` to `sys.path`. Prefer the install; use the fallback only if the install
demonstrably perturbs the env, and record which was used.

**Verify:** `python -c "import augenblick; print(augenblick.__file__)"`

---

## Phase 1 — core

### `core/errors.py`

```python
class AugenblickError(Exception):
    """Base for all first-party pipeline errors."""

class SceneError(AugenblickError):
    """A scene directory is missing something a stage requires."""

class BackendError(AugenblickError):
    """A backend subprocess exited non-zero."""
    def __init__(self, message: str, returncode: int):
        super().__init__(message)
        self.returncode = returncode

class MethodNotFound(AugenblickError):
    """A method name was not present in the registry."""
```

`BackendError` carries `returncode` so the CLI can exit with the backend's own code instead of a
generic `1`. This replaces the current `sys.exit(returncode)` inside the helper, which makes the
wrappers unusable as a library.

### `core/scene.py`

```python
@dataclass(frozen=True)
class Scene:
    """A COLMAP-format scene: images/ plus optional masks/ and sparse/0/."""
    root: Path

    @property
    def images_dir(self) -> Path:      # root/"images"
    @property
    def masks_dir(self) -> Path:       # root/"masks"
    @property
    def sparse_dir(self) -> Path:      # root/"sparse"/"0"
    def has_masks(self) -> bool:       # masks_dir.is_dir()
    def has_reconstruction(self) -> bool:
        # sparse_dir.is_dir() and any file in it
    def require_images(self) -> None:  # raise SceneError if missing or empty
    def require_reconstruction(self) -> None:  # raise SceneError if missing or empty
    def link_colmap_masks(self, dest: Path) -> Path | None:
```

`has_reconstruction` must check the directory is **non-empty**, mirroring the shell guard at
`slurm/recon.sbatch:70` (`-z "$(ls -A ...)"`). An empty `sparse/0/` means SfM failed.

`link_colmap_masks` centralises the symlink trick duplicated in `run_masked_colmap.py:30-37` and
`run_turntable_to_colmap.py:428-437`. **Port the existing `.jpg.png` behaviour exactly:**

```python
link = dest / f"{m.rsplit('.', 1)[0]}.jpg.png"
```

> The hardcoded `.jpg` is a known latent bug for non-`.jpg` scenes. **Do not fix it here.** All
> current scenes are `.jpg` (`prepare_uf_dataset.py` normalises to `.jpg`). Preserve behaviour;
> record the bug in the Phase 7 MEMORY file as separate follow-up work.

Guard every `os.symlink` with an existence check, as the sources do — these scripts are re-run on
existing output dirs and must stay idempotent.

### `core/config.py`

Each method declares a frozen dataclass; help text rides in `field(metadata={"help": ...})`.
One function generates the parser:

```python
def add_dataclass_arguments(parser: argparse.ArgumentParser, config_cls: type) -> None:
    """Add one CLI argument per dataclass field, inferring the argparse spec from its type."""
```

Mapping rules — implement exactly these, via `dataclasses.fields()` and `typing.get_type_hints()`:

| Field type | Default | argparse |
|---|---|---|
| `bool` | `False` | `action="store_true"` |
| `bool` | `True` | `action=BooleanOptionalAction` (gives `--no-<name>`) |
| `int` / `float` / `str` | any | `type=<t>, default=<default>` |
| `Optional[T]` | `None` | `type=T, default=None` |
| `Literal["a","b"]` | any | `choices=["a","b"], type=str` |
| `list[int]` | via `default_factory` | `nargs="+", type=int` |
| `Path` | any | `type=Path` |

- Flag name is `--<field_name>`, unless `metadata["cli_name"]` overrides it.
- The `bool` split is load-bearing: it reproduces GW's `--no-postprocess` (default `True`) and
  everyone else's `store_true` (default `False`) **from the defaults alone**.
- Use `dataclasses.MISSING` to detect `default_factory`; mutable defaults must use it.

Also provide the inverse:

```python
def config_from_namespace(config_cls: type, ns: argparse.Namespace):
    """Build a config instance from parsed args, ignoring unrelated namespace entries."""
```

This dataclass-first design is what makes each method usable as a library —
`VGGTSfM(VGGTConfig(use_ba=True)).run(scene, out)` — which argparse-only scripts cannot do, and
which a future sweep driver needs.

### `core/registry.py`

```python
SFM_REGISTRY: dict[str, type] = {}
RECONSTRUCTION_REGISTRY: dict[str, type] = {}

def register_sfm(cls): ...          # keys on cls.name; raise ValueError on duplicate
def register_reconstruction(cls): ...
def get_sfm(name: str): ...         # raise MethodNotFound listing sorted available names
def get_reconstruction(name: str): ...
```

Decorators return `cls` unchanged. `MethodNotFound` messages must list available names — that
error is the discoverability surface when a SLURM job passes a bad backend string.

### `core/process.py`

```python
def run(cmd: list[str], cwd: str | Path | None = None) -> None:
    """Run a backend command, streaming its output, raising BackendError on failure."""
```

Body copied from any wrapper's `run()` (all four are identical), with one change: replace
`sys.exit(result.returncode)` with `raise BackendError(...)`. Keep both log lines
(`Running: ...` and `  cwd: ...`) so existing SLURM logs stay recognisable.

### `core/timing.py`

```python
class StageTimer:
    """Log a banner, per-stage headers, and a per-stage + total timing summary."""
    def __init__(self, title: str, total_stages: int, header: dict[str, object]): ...
    @contextmanager
    def stage(self, name: str): ...   # logs "Step i/N: <name>", records elapsed
    def summary(self, footer: dict[str, object]) -> None: ...
```

Reproduce the existing output exactly: `"=" * 60` rules, `Step i/N: ...`, per-stage
`"<name> completed in %.1fs"`, then the summary block with total.

### `core/method.py`

```python
@dataclass
class StageResult:
    """What a completed stage produced, for logging and for chaining stages."""
    output_dir: Path
    elapsed: float
    details: dict[str, object] = field(default_factory=dict)

class Method(ABC):
    """Base for any pipeline stage that transforms a scene directory."""
    name: ClassVar[str]
    config_cls: ClassVar[type]

    def __init__(self, config): self.config = config
    @classmethod
    def from_namespace(cls, ns): return cls(config_from_namespace(cls.config_cls, ns))
    def validate(self, scene: Scene) -> None: scene.require_images()
    @abstractmethod
    def run(self, scene: Scene, output_dir: Path) -> StageResult: ...
```

**Verify Phase 1:** `pytest tests/test_scene.py tests/test_config.py tests/test_registry.py`

Write these tests now; all are CPU-only. Cover: `Scene` path properties; `require_*` raising
`SceneError` on missing **and** on empty `sparse/0/`; `link_colmap_masks` producing `foo.jpg.png`
from `foo.png` and being idempotent; each config→argparse mapping row above, especially both
`bool` branches; registry duplicate-name rejection and `MethodNotFound` listing names.

---

## Phase 2 — reconstruction backends

### `reconstruction/base.py`

```python
@dataclass
class Stage:
    """One backend invocation: a label and the argv to run for it."""
    name: str
    cmd: list[str]

class ReconstructionMethod(Method):
    """Consumes a COLMAP scene, produces a mesh."""
    def validate(self, scene):
        scene.require_images()
        scene.require_reconstruction()
    @abstractmethod
    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]: ...
    @abstractmethod
    def mesh_path(self, output_dir: Path) -> Path: ...

class SubprocessBackend(ReconstructionMethod):
    backend_dir: ClassVar[Path]
    use_cwd: ClassVar[bool] = True
    title: ClassVar[str]

    def prepare(self, scene: Scene, output_dir: Path) -> Scene:
        """Hook for backends needing a modified scene; default returns it unchanged."""
        return scene

    def run(self, scene, output_dir) -> StageResult:
        # validate -> prepare -> for each stage: timer.stage() + process.run(cmd, cwd)
        # cwd = self.backend_dir if self.use_cwd else None
```

`mesh_path()` turns the per-backend output table in `MEMORY/pipeline-reconstruction.md` into code,
so a driver can locate a result without knowing which backend produced it.

Backend dirs resolve from the package: `REPO_ROOT = Path(__file__).resolve().parents[3]` — verify
this yields the repo root from `src/augenblick/reconstruction/base.py` and assert it in a test,
since the sources used `parents[2]` from a different depth.

### The four backends

For each: a `<Name>Config` dataclass and a `<Name>Backend(SubprocessBackend)`.

**Copy every field name, type, default, and help string verbatim from the source script's
`add_argument` calls.** They are not reproduced here precisely so you copy rather than retype.

| Backend | class / name | backend_dir | use_cwd | Stages |
|---|---|---|---|---|
| 2DGS | `TwoDGSBackend` / `"2dgs"` | `src/libs/2dgs` | `True` | train, render |
| SuGaR | `SugarBackend` / `"sugar"` | `src/libs/sugar` | `True` | gs_train, sugar_train |
| PGSR | `PgsrBackend` / `"pgsr"` | `src/libs/pgsr` | `True` | train, render |
| GW | `GWBackend` / `"gw"` | `src/libs/gaussian_wrapping` | **`False`** | train, extract, texture |

Per-backend requirements:

- **2DGS** (`run_2dgs.py`): mesh at `<output>/train/ours_<iterations>/fuse_post.ply`. Render stage
  always passes `--skip_test`; `--unbounded` and `--skip_mesh` are appended only when set.
- **SuGaR** (`run_sugar.py`): two stages. Stage 1 is `gaussian_splatting/train.py` into
  `<output>/gs_model`; stage 2 is `train.py` into `<output>/sugar`. **Booleans here are `--flag
  True` string pairs**, not `store_true` — see `run_sugar.py:128-136`. `--eval False` and `--gpu`
  are always passed. `mesh_path` → `<output>/sugar/refined_mesh/<scene_name>/`.
- **PGSR** (`run_pgsr.py`): override `prepare()` with `prepare_scene()` from `run_pgsr.py:47-78`,
  copied verbatim — copies `images`/`masks`/`sparse`, moves `sparse/0/*` up, `rmdir`s `sparse/0`,
  and **returns early if the prepared scene already exists**. Keep that reuse check. Note the
  render stage takes only `-m`, no `-s`; and `--skip_mesh` maps to `--skip_train` on render (yes,
  really — `run_pgsr.py:171-172`). Mesh at `<output>/mesh/tsdf_fusion_post.ply`.
- **GW** (`run_gw.py`): `use_cwd = False`. Add this comment above it, since it looks like a bug:

  ```python
  # GW's scripts import from their own directory, which Python adds to sys.path only for the
  # script's own path; setting cwd would not help and upstream expects absolute invocation.
  ```

  Port `build_train_cmd` / `build_extract_cmd` / `build_texture_cmd` verbatim, plus
  `get_mesh_path` and `get_textured_mesh_path` as methods. Keep all `DEFAULT_*` constants.
  `extract_iteration` defaults to `iterations` when unset. `--resolution` (`-r`) forwards to all
  three stages. Passthrough: constructor takes `passthrough: list[str] = []` appended to the
  **train stage only**.

### `reconstruction/__init__.py`

```python
from augenblick.reconstruction import gw, pgsr, sugar, twodgs  # noqa: F401
```

The import is what fires registration; the `noqa` comment must stay or linters will strip it.

**Verify Phase 2:** `pytest tests/test_recon_argv.py`

The outer CLI is being redesigned freely, but the **argv handed to each backend script** is
dictated by upstream `train.py` / `render.py` signatures and must not drift — a wrong flag there
surfaces only as a bad reconstruction hours later. For each backend, construct the config with
all-default values plus one representative non-default, and assert `stages()` produces exactly the
argv the current script builds for the same inputs. Derive expected argv **by reading the source
script**, not by running it (no GPU). Assert on the full list, not a subset.

---

## Phase 3 — SfM methods

### `sfm/base.py`

```python
class SfMMethod(Method):
    """Consumes a scene with images/, produces sparse/0/ in the output directory."""
    def validate(self, scene): scene.require_images()
    @abstractmethod
    def run(self, scene, output_dir) -> SfMResult: ...

class SceneRefiner(SfMMethod):
    """An SfM step that refines an existing reconstruction rather than creating one."""
    def validate(self, scene):
        scene.require_images()
        scene.require_reconstruction()
```

`SceneRefiner` exists so the type system encodes what CLAUDE.md currently states only in prose:
the turntable method is a post-SfM step. `SfMResult` extends `StageResult` with `scene: Scene`,
`num_images: int`, `num_points: int`.

### `sfm/colmap.py` — port as-is

`ColmapSfM(SfMMethod)`, `name = "colmap"`. Move `main()` from `run_masked_colmap.py` into `run()`
**unchanged in behaviour**. Preserve exactly:

- `max_image_size=2400`, `camera_model="SIMPLE_PINHOLE"` defaults.
- `pycolmap.CameraMode.PER_IMAGE`, `eo.num_threads = 8`.
- `images/` and `masks/` **symlinked** into the output (not copied).
- Deleting a pre-existing `database.db`.
- Best-model selection by `num_reg_images()`, written to `sparse/0`.
- The failure path: currently `print("COLMAP_FAIL...")` then `raise SystemExit(2)`. Convert to
  `raise SceneError("COLMAP_FAIL: no model reconstructed")`; the CLI maps `SceneError` to **exit
  code 2**, preserving what SLURM sees.
- Keep the `COLMAP_DONE`/`COLMAP_FAIL` tokens in messages — log scrapers may match them.

Replace the inline mask-symlink block with `scene.link_colmap_masks(out_dir / "masks_colmap")`.
Swap bare `print(..., flush=True)` for the module logger.

### `sfm/vggt.py`

`VGGTSfM(SfMMethod)`, `name = "vggt"`. Config fields copied from `parse_args()` in
`run_vggt_to_colmap.py:50-72` (note `fine_tracking` defaults to **`True`**).

- Keep the upstream attribution header comment at the top of the new file.
- Move `run_VGGT()` and `rename_colmap_recons_and_rescale_camera()` as module-level functions,
  **bodies unchanged**.
- Keep the torch/CUDA setup, the seeding block, and `with torch.no_grad():` around the run.
- Keep both branches (BA and no-BA) exactly, including the no-BA constants
  (`max_points_for_colmap = 100000`, forced `PINHOLE`, `shared_camera = False`).
- Keep the image/mask copy-out loop and `points.ply` export.
- Drop the WIP docstring at the end of the source file.
- Keep the module-level `torch.backends.cudnn` settings — but place them inside the module, not at
  package import time, so importing `augenblick` never touches CUDA. **Import torch and `vggt.*`
  lazily inside `run()`**, so `augenblick.sfm` is importable on a GPU-less login node. This is
  required for Phase 6 verification to work at all.

### `sfm/turntable.py`

`TurntableRefiner(SceneRefiner)`, `name = "turntable"`.

Move these module-level helpers **verbatim, bodies untouched**: `group_key`, `order_key`, `Rt`,
`centers`, `Rf`, `fit_axis_step`, `fit_rig_poses`, `_batch_dlt`, `apply_track_preserving`,
`rig_poses`, `_triangulate_batched`, `rig_ba`. This is ~300 lines of rig-fitting math; do not
touch it. Keep `from scipy.optimize import least_squares` inside `rig_ba` where it already is.

`run()` is `main()` from `run_turntable_to_colmap.py:320-491`, restructured only as:

```python
def run(self, scene, output_dir):
    # ... shared setup: load reconstruction, group images, fit axis/step, pick sign ...
    if mode == "tracks":
        return self._run_tracks(...)
    return self._run_sift(...)
```

Preserve exactly: `use_masks = config.use_masks or scene.has_masks()` (masks auto-enable);
step-sign disambiguation trying `[step, -step]` and keeping lower centre error; `auto` mode
picking `tracks` when mean input track length `>= 3.0`, else `sift`; `rig_ba` skipped only when
`--rig_ba off`; the SIMPLE_PINHOLE camera rewrite in tracks mode; and the pycolmap>=4 calls
`add_camera_with_trivial_rig` / `add_image_with_trivial_frame` **with their existing comments**.

**Verify Phase 3:** `python -c "import augenblick.sfm, augenblick.reconstruction"` on the login
node — this proves the lazy torch import works. Then `augenblick sfm --list` and
`augenblick recon --list` after Phase 4.

---

## Phase 4 — CLI

`cli/main.py`, exposed as console script `augenblick` via `[project.scripts]` in `pyproject.toml`.

```
augenblick sfm   <method> --scene <dir> --output <dir> [method flags]
augenblick recon <method> --scene <dir> --output <dir> [method flags]
augenblick sfm --list
augenblick recon --list
```

Uniform `--scene` / `--output` replaces today's split between positionals (reconstruction) and
`--input_dir` / `--output_dir` (SfM). Build subparsers by iterating the registry and calling
`add_dataclass_arguments(sub, cls.config_cls)`, so a newly registered method appears in `--help`
with no CLI edit.

GW alone needs `parse_known_args()`, its extras passed as `passthrough`. Gate this on a class-level
`accepts_passthrough: ClassVar[bool] = False` that `GWBackend` sets `True` — do **not** apply
passthrough globally, or typos in other backends will be silently forwarded instead of rejected.

Exit codes: `SceneError` → **2** (matches the current COLMAP path and the SLURM guard);
`BackendError` → its `returncode`; `MethodNotFound` → **2**. Log the message, no traceback.

**Verify Phase 4:** `augenblick --help`, `augenblick sfm --list`, `augenblick recon --list`, and
`augenblick recon gw --help` (checks `--no-postprocess` renders correctly).

---

## Phase 5 — delete old scripts, migrate SLURM

Only after Phases 0–4 verify. Delete all seven source scripts and both `__pycache__` dirs; if
`pipeline/sfm/` and `pipeline/reconstruction/` are then empty, remove them
(`pipeline/preparation/` stays).

Update the SLURM jobs — read each fully before editing, and keep every surrounding guard:

| File | Change |
|---|---|
| `slurm/recon.sbatch:78` | `python "pipeline/reconstruction/run_${BACKEND}.py" "$SCENE" "$OUT" "$@"` → `augenblick recon "$BACKEND" --scene "$SCENE" --output "$OUT" "$@"` |
| `slurm/colmap_sfm.sbatch:64` | → `augenblick sfm colmap --scene "$SCENE" --output "$OUT" "$@"` |
| `slurm/vggt_sfm.sbatch:58`, `slurm/vggt_ba_sfm.sbatch:58` | → `augenblick sfm vggt --scene ... --output ... "$@"` (keep `--use_ba` in the BA variant) |
| `slurm/template.sbatch:33` | same VGGT form |

`recon.sbatch`'s `run_${BACKEND}.py` interpolation becomes a registry lookup that fails with a
listed set of valid names instead of a missing-file error.

> `slurm/common.sh` and `slurm/vggt_sfm.sbatch` have **uncommitted local modifications**. Do not
> revert them; read the working-tree version and edit on top.

Deleting and migrating in one commit is what keeps the tree honest — leaving the old scripts
behind produces two divergent paths and no way to tell which produced a result.

---

## Phase 6 — SLURM smoke run (mandatory gate)

Nothing in Phases 0-5 exercises a GPU. This phase is the **first and only** end-to-end proof that
the port works. It is a required gate, not an optional check: **do not report the port as complete
until Phase 6 passes.** Until then, state plainly that backends are unverified.

### Scene selection

The scene list is built by `find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name prepared`,
sorted, so the array index is stable. Verified contents of
`/blue/arthur.porto/data/datasets/photogrammetry/main`:

| Index | Scene | Images | Masks |
|---|---|---|---|
| 0 | `TH24-21_Birdsnest` | 159 | 0 |
| 1 | `UF_birds_59449-2` | 365 | 365 |
| 2 | **`UF_birds_ivory2`** | **184** | **184** |
| 3 | `UF_herp_148667` | 229 | 229 |
| 4 | `UF_Herp_3998` | 276 | 276 |
| 5 | `UF_mammals_36342_skull` | 276 | 276 |

Use **`--array=2` (`UF_birds_ivory2`)**: the smallest scene that still has masks, so the masked
code paths are exercised. Index 0 has no masks and would silently skip them. Re-run the `find`
command before submitting and confirm index 2 is still `UF_birds_ivory2` — a new scene directory
would shift every index.

### Step 6.1 — VGGT SfM

```bash
cd <repo root>
sbatch --array=2 slurm/vggt_sfm.sbatch
```

Wait for completion (`squeue -u $USER`). Then check, in order:

1. `sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS` shows `COMPLETED`.
2. `slurm/logs/vggt-sfm-<jobid>_2.err` contains no traceback.
3. The log's banner shows the expected conda env and a visible GPU.
4. The output scene exists and is non-empty:
   ```bash
   ls -la /blue/arthur.porto/srizvi63.gatech/results/UF_birds_ivory2/all/vggt/sparse/0/
   ```
   Expect `cameras.bin`, `images.bin`, `points3D.bin`, `points.ply`, all non-zero.
5. Registered image count is sane (should be near 184, not 0 or 1):
   ```bash
   python -c "import pycolmap; r=pycolmap.Reconstruction('<out>/sparse/0'); print(r.num_reg_images(), r.num_points3D())"
   ```

**If this fails, stop.** Do not proceed to 6.2 and do not start fixing forward blindly — report
the log excerpt.

### Step 6.2 — 2DGS reconstruction

2DGS is the right smoke backend: two stages, no scene rewriting, and the shortest runtime of the
four. Use a reduced iteration count so the gate is minutes, not hours:

```bash
BACKEND=2dgs SFM=vggt sbatch --array=2 slurm/recon.sbatch --iterations 2000
```

> `--iterations 2000` is a **smoke value only** — far below the 30000 default. It proves plumbing,
> not reconstruction quality. Never quote a mesh from this run as a result.

Check:

1. `sacct` shows `COMPLETED`; `slurm/logs/recon-<jobid>_2.err` has no traceback.
2. Both stages appear in the log — `Step 1/2` (train) and `Step 2/2` (render), each with a
   `completed in ...s` line, then the summary block with a total.
3. The mesh exists and is non-empty, at the path `mesh_path()` predicts:
   ```bash
   ls -la <out>/train/ours_2000/fuse_post.ply
   ```
   **This is the key assertion of the whole phase**: it proves the CLI, the config bridge, the
   registry, `Scene` validation, argv construction, and `mesh_path()` all agree with what the
   backend actually did.

### Step 6.3 — record the result

Append to the Phase 7 MEMORY file: the two job IDs, the scene, the smoke `--iterations` value, and
the observed stage timings. This is the provenance record showing the port was validated, and the
timings become the baseline for spotting a future regression.

### What Phase 6 does and does not prove

- **Proves:** the package imports under the batch env; the CLI parses; the registry resolves;
  `Scene` validation passes on real data; argv reaches the backends correctly; VGGT and 2DGS run
  to completion; `mesh_path()` matches reality.
- **Does not prove:** SuGaR, PGSR, or GW work; that any output is *good*; that non-default flags
  behave. PGSR's `prepare()` scene-flattening and GW's no-`cwd` invocation are the two highest-risk
  unexercised paths — flag both as untested when reporting.

### Optional extension

If time allows, one further run per remaining backend closes the biggest gaps — highest value
first, since these exercise the most unusual code paths:

```bash
BACKEND=pgsr SFM=vggt sbatch --array=2 slurm/recon.sbatch --iterations 2000   # exercises prepare()
BACKEND=gw   SFM=vggt sbatch --array=2 slurm/recon.sbatch --iterations 2000   # exercises no-cwd + passthrough
```

Do not run all four concurrently against one checkout if any would trigger a backend rebuild —
per `slurm/README.md`, concurrent `scripts/setup_*.sh` runs race on shared `build/` dirs. Training
jobs sharing a checkout are fine.

---

## Phase 7 — documentation

Per `.claude/MEMORY/repo-conventions.md`, detail goes in `MEMORY/`; CLAUDE.md changes only where
invocation or navigation changes.

1. **Create `.claude/MEMORY/augenblick-package.md`**: the two ABCs and their contracts, the
   registry, the config/argparse bridge, `Scene`, exit codes, and a worked "adding a new backend"
   recipe. Note the known `.jpg`-hardcoded mask-symlink bug as follow-up work.
2. **Index it** in CLAUDE.md's MEMORY table.
3. **Update CLAUDE.md**: new invocation in Quick start; `src/` is no longer only third-party
   backends; **delete the two stale `run_colmap.sh` gotchas** (lines 62-65) and the stage-table entry (line 39). The `run_pgsr.py` flattening and `run_gw.py` no-`cwd` gotchas are now
   class properties — repoint them at the new MEMORY file.
4. **Update `MEMORY/pipeline-sfm.md`** (remove its `run_colmap.sh` section, lines 45-62),
   **`MEMORY/scene-format.md`** (its line in the data-flow diagram), **`MEMORY/cluster-slurm.md`**
   (line 44's `run_colmap.sh` mention), and **`MEMORY/pipeline-reconstruction.md`** (new commands).
5. **Update `README.md`**: the tree at line ~135 and the usage block at line ~236 both still show
   `run_colmap.sh`; update all invocations to the new CLI.
6. **Update `slurm/README.md`**: the submit examples at the top, and the gotcha at line ~44 whose
   second half ("Only the shell entry point `run_colmap.sh` needs a real binary...") is now stale —
   every SfM path goes through `pycolmap`, so no COLMAP binary is ever needed.

## Conventions

Per `repo-conventions.md`, with `pipeline/preparation/prepare_uf_dataset.py` as the reference:

- Comments are **one line, at most 20 words**, stating *purpose*, never restating the code. No
  `# Create the parser` above `parser = ...`. Keep comments carrying non-obvious intent — the GW
  no-`cwd` rationale, the `<image>.jpg.png` quirk, and the pycolmap>=4 rig notes are exactly the
  ones worth writing.
- Docstrings may run several lines with `Args:` / `Returns:`, describing purpose rather than
  walking through behaviour. No worked examples. Every ABC and abstract method gets one — those
  docstrings *are* the extension contract.
- Type-annotate all public signatures. The config bridge reads annotations at runtime, so these
  are load-bearing, not decoration.
- Module logger per module: `logger = logging.getLogger(__name__)`. Configure logging **only** in
  `cli/main.py` — never `basicConfig` at import time in a library module.

## Risks

- **Editable install perturbing the cluster env** (Phase 0), given the numpy-generation fragility
  in `MEMORY/environment-and-gpu.md`. Mitigated by empty dependencies + `--no-deps`, and by the
  `sys.path` fallback.
- **Backend argv regressions.** The outer CLI is reshaped freely, but a wrong flag reaching
  `train.py` shows up only as a bad reconstruction hours later. The Phase 2 argv fixtures exist
  solely to guard this.
- **Nothing is end-to-end verifiable on the login node.** No GPU: Phases 0-5 are checked by
  imports, unit tests, and `--help` only. **Phase 6 is the mandatory gate** — the port is not
  complete until it passes, and backends must be reported as unverified until then.
- **In-flight SLURM batches** break the moment Phase 5 lands, since the old scripts disappear.
  Land it when the queue is drained; the failure is loud and immediate, not silent.
- **Over-abstraction.** The three SfM methods share little beyond "produce `sparse/0/`". Keep the
  ABC thin and honest; forcing the turntable refiner and the VGGT network into one template method
  would be worse than the status quo.
