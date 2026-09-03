# Plan: Move vendored/submodule backends into `src/libs/`

## Context

`src/` currently mixes two very different kinds of content at the same level:

- **First-party code** the team writes and edits: `src/pipeline/` (legacy first-generation
  pipeline) and `src/utils/`.
- **Third-party backends** that are vendored copies or git submodules and are never hand-edited
  as part of normal work: `2dgs`, `pgsr`, `sugar`, `gaussian_wrapping`, `pytorch3d`,
  `light_glue`, `vggt`.

The second group is ~3800 tracked files and dominates every `src/` listing, grep, and IDE tree.
Moving those seven directories under `src/libs/` makes the first-party/third-party boundary
explicit, so `src/pipeline` and `src/utils` are visible again and tooling can exclude
`src/libs/` wholesale.

**Intended outcome:** a pure relocation. No file contents inside the seven moved directories
change, no packages are renamed, and every entry point (`scripts/auto_setup.sh`, the four
`pipeline/reconstruction/run_*.py` wrappers, the SfM scripts) behaves exactly as before. Git
history is preserved via `git mv`, and the three git submodules stay correctly registered at
their new paths.

### Facts established during exploration (do not re-derive)

- Tracked-file counts: `src/2dgs` 732, `src/pgsr` 728, `src/sugar` 783,
  `src/gaussian_wrapping` 1541, `src/vggt` 48, plus 1 gitlink each for `src/pytorch3d` and
  `src/light_glue`.
- **Three git submodules** are registered (`git submodule status` confirms all three are
  initialized and checked out):
  - `src/light_glue` → `https://github.com/cvg/LightGlue.git`
  - `src/pytorch3d` → `https://github.com/facebookresearch/pytorch3d.git`
  - `src/gaussian_wrapping/submodules/Depth-Anything-V2` →
    `https://github.com/DepthAnything/Depth-Anything-V2.git`
- **Depth-Anything-V2 is nested inside `src/gaussian_wrapping`**, which itself moves. Its
  submodule path, its `.git` gitdir pointer, and its `.git/modules` location all shift by one
  directory level. This is the single most error-prone part of the whole change — handle it
  exactly as written in Step 3.
- `git version 2.52.0` — modern enough that `git mv` on a submodule path updates `.gitmodules`,
  the gitlink, the submodule's `.git` file, and `.git/modules/**/config`'s `core.worktree`
  automatically. **The plan still verifies each of these explicitly**, because the nested
  Depth-Anything-V2 case is the one git handles least reliably.
- `.gitignore` contains only `src/*/__pycache__/` as a `src/`-relative rule. After the move,
  backend `__pycache__` dirs live at `src/libs/*/__pycache__/`, which that rule no longer
  matches — this needs a new rule (Step 5).
- Only **four Python files** contain path constants that break:
  `pipeline/reconstruction/run_{2dgs,sugar,pgsr,gw}.py`, each with a single
  `REPO_ROOT / "src" / "<name>"` line. `REPO_ROOT = Path(__file__).resolve().parents[2]` is
  unaffected (those files do not move).
- `pipeline/sfm/run_vggt_to_colmap.py` imports `from vggt.models.vggt import VGGT` etc. These
  are **package imports resolved through the editable install**, not path references. They do
  not change; the editable install is simply re-pointed at the new source location (Step 6).
- `src/vggt/pyproject.toml` declares `name = "vggt"` with
  `[tool.setuptools.packages.find] where = ["."]`. Package identity is path-independent, so the
  move does not require any packaging edit.
- `run_gw.py`'s `GW_DIR` points at `src/gaussian_wrapping/` directly (which is where `train.py`,
  `pivot_based_mesh_extraction.py`, and `texture_mesh.py` live). Some prose in
  `gaussian-wrapping-inclusion.md` describes a nested `gaussian_wrapping/gaussian_wrapping/`
  layout that **does not exist on disk** — do not "fix" `run_gw.py` to match that prose.
- `src/pipeline/*.py` (legacy) contains many `vggt`/`sugar` *identifier* matches
  (`run_vggt_inference`, `sugar_output_dir`, `--skip_sugar`, …) but **no `src/<backend>` path
  strings**. It requires no changes. Do not touch `src/pipeline/` or `src/utils/`.
- The `augenblick` conda env is not present on this login node, so the editable-install repair
  in Step 6 cannot be run or verified here — it must be run on a compute node.

---

## Scope

**Move (7 directories):**

| From | To |
|------|-----|
| `src/2dgs` | `src/libs/2dgs` |
| `src/pgsr` | `src/libs/pgsr` |
| `src/sugar` | `src/libs/sugar` |
| `src/gaussian_wrapping` | `src/libs/gaussian_wrapping` |
| `src/pytorch3d` | `src/libs/pytorch3d` (submodule) |
| `src/light_glue` | `src/libs/light_glue` (submodule) |
| `src/vggt` | `src/libs/vggt` |

**Do not move:** `src/pipeline/`, `src/utils/`, `src/__pycache__/`.

**Do not edit any file inside the seven moved directories.** Their internal imports are all
relative or package-based and survive relocation untouched.

---

## Step 0 — Pre-flight

Run from the repo root (`/home/srizvi63.gatech/porto-photogrammetry`). Every command in this
plan assumes that cwd.

```bash
cd /home/srizvi63.gatech/porto-photogrammetry
git status
git rev-parse --abbrev-ref HEAD          # expect: syed/fall-26-experiments
git submodule status                     # expect 3 lines, none prefixed '-' or 'U'
```

The working tree has **pre-existing uncommitted changes** that are unrelated to this task:

```
 M src/gaussian_wrapping/submodules/diff-gaussian-rasterization-ms/conv.cu
 M src/gaussian_wrapping/submodules/diff-gaussian-rasterization-ms/cuda_rasterizer/forward.cu
 M src/gaussian_wrapping/submodules/diff-gaussian-rasterization-ms/rasterize_points.cu
 M src/gaussian_wrapping/submodules/fused-ssim/ssim.cu
?? .codex, commands.md, gaussian-wrapping-inclusion.md, gw_train_depth_mask.py,
?? pr-description.md, src/gaussian_wrapping/install.py
```

`git mv` **carries modified tracked files along correctly** — the four `M` files move with
`src/gaussian_wrapping` and stay modified at their new paths. That is fine and expected.

The **untracked** `src/gaussian_wrapping/install.py` is different: `git mv` on the parent
directory moves the whole directory on disk (including untracked files) but git only records
the tracked ones. Verify after the move that `src/libs/gaussian_wrapping/install.py` exists on
disk and still shows as untracked. It will.

Record a baseline to compare against at the end:

```bash
git ls-files | wc -l > /tmp/baseline_tracked_count.txt
cat /tmp/baseline_tracked_count.txt
```

**Safety net.** Create a rollback tag before touching anything:

```bash
git tag pre-libs-move-backup
```

If anything goes irrecoverably wrong mid-way:
`git reset --hard pre-libs-move-backup && git submodule update --init --recursive`

---

## Step 1 — Create the destination directory

```bash
mkdir -p src/libs
```

Do **not** add a `src/libs/__init__.py`. None of these are imported as a `src.libs.*` package —
they are reached by filesystem path (the `run_*.py` wrappers) or by installed package name
(`vggt`, `lightglue`, `pytorch3d`). Adding an `__init__.py` would be inert at best.

---

## Step 2 — `git mv` the four vendored (non-submodule) directories

These four are plain tracked directories. Move them first, before the submodules, so that any
problem surfaces on the simple cases.

```bash
git mv src/2dgs             src/libs/2dgs
git mv src/pgsr             src/libs/pgsr
git mv src/sugar            src/libs/sugar
git mv src/vggt             src/libs/vggt
```

Verify:

```bash
ls src/libs/                 # 2dgs pgsr sugar vggt
git status --short | head -5 # renames staged as R
git ls-files src/libs/2dgs | wc -l   # 732
git ls-files src/libs/pgsr | wc -l   # 728
git ls-files src/libs/sugar | wc -l  # 783
git ls-files src/libs/vggt | wc -l   # 48
```

If any count differs from the expected value above, stop and investigate before continuing.

---

## Step 3 — Move `gaussian_wrapping` (contains a nested submodule)

This is the delicate one. `src/gaussian_wrapping/submodules/Depth-Anything-V2` is a submodule
whose recorded path, `.git` gitdir pointer, and `.git/modules` location all live *underneath*
the directory being moved.

### 3a. Perform the move

```bash
git mv src/gaussian_wrapping src/libs/gaussian_wrapping
```

### 3b. Verify what git updated automatically

```bash
grep -n "Depth-Anything-V2" .gitmodules
cat src/libs/gaussian_wrapping/submodules/Depth-Anything-V2/.git
ls .git/modules/src/                 # and .git/modules/src/libs/ if it exists
```

Expected end state — check all three, and repair any that git did not handle:

1. **`.gitmodules`** — the stanza must read:
   ```
   [submodule "src/libs/gaussian_wrapping/submodules/Depth-Anything-V2"]
       path = src/libs/gaussian_wrapping/submodules/Depth-Anything-V2
       url = https://github.com/DepthAnything/Depth-Anything-V2.git
   ```
   Both the section **name** and the `path` value must be updated. (Step 4 handles the two
   top-level submodules; if git already rewrote this stanza, leave it alone.)

2. **The submodule's `.git` file.** It was
   `gitdir: ../../../../.git/modules/src/gaussian_wrapping/submodules/Depth-Anything-V2`.
   The new path is one level deeper, so it needs **five** `../` and a `libs/` component:
   ```
   gitdir: ../../../../../.git/modules/src/libs/gaussian_wrapping/submodules/Depth-Anything-V2
   ```
   Repair only if wrong:
   ```bash
   printf 'gitdir: ../../../../../.git/modules/src/libs/gaussian_wrapping/submodules/Depth-Anything-V2\n' \
     > src/libs/gaussian_wrapping/submodules/Depth-Anything-V2/.git
   ```

3. **The git-dir location under `.git/modules/`.** If git did not relocate it, move it by hand:
   ```bash
   mkdir -p .git/modules/src/libs/gaussian_wrapping/submodules
   mv .git/modules/src/gaussian_wrapping/submodules/Depth-Anything-V2 \
      .git/modules/src/libs/gaussian_wrapping/submodules/Depth-Anything-V2
   rmdir -p .git/modules/src/gaussian_wrapping/submodules 2>/dev/null || true
   ```
   Then set `core.worktree` in that module's config to point back at the new worktree
   (relative to the module dir, which is now 7 levels deep):
   ```bash
   git config --file .git/modules/src/libs/gaussian_wrapping/submodules/Depth-Anything-V2/config \
     core.worktree ../../../../../../../src/libs/gaussian_wrapping/submodules/Depth-Anything-V2
   ```

   > The `core.worktree` value must be a path from
   > `.git/modules/src/libs/gaussian_wrapping/submodules/Depth-Anything-V2/` back to the repo
   > root, then down to the worktree. That module dir is 7 components below the root
   > (`.git`, `modules`, `src`, `libs`, `gaussian_wrapping`, `submodules`, `Depth-Anything-V2`),
   > hence seven `../`. **Do not guess this** — confirm with the verification command in 3c
   > rather than by counting a second time.

### 3c. Verify the nested submodule is healthy

```bash
git -C src/libs/gaussian_wrapping/submodules/Depth-Anything-V2 rev-parse HEAD
# expect: a561b849ebae10a6f5ef49e26c83cbbcd36c71bf

git -C src/libs/gaussian_wrapping/submodules/Depth-Anything-V2 status --short
# expect: clean (no output), and NO "fatal:" error

git submodule status
# expect 3 lines; none prefixed with '-' (uninitialized) or 'U' (conflict)
```

If `rev-parse` errors with `fatal: not a git repository`, one of the three items in 3b is still
wrong. Fix it before proceeding — do not continue with a broken submodule.

### 3d. Confirm the untracked file and modified CUDA files came along

```bash
ls src/libs/gaussian_wrapping/install.py            # exists
git status --short | grep -E "diff-gaussian-rasterization-ms|fused-ssim"
# the four modified .cu files should now show at src/libs/... paths
```

---

## Step 4 — Move the two top-level submodules

```bash
git mv src/pytorch3d  src/libs/pytorch3d
git mv src/light_glue src/libs/light_glue
```

Verify all four aspects for each, exactly as in Step 3b:

```bash
cat .gitmodules
cat src/libs/pytorch3d/.git
cat src/libs/light_glue/.git
grep -n worktree .git/modules/src/libs/pytorch3d/config .git/modules/src/libs/light_glue/config
```

Expected `.gitmodules` (complete file, all three stanzas):

```
[submodule "src/libs/light_glue"]
	path = src/libs/light_glue
	url = https://github.com/cvg/LightGlue.git
[submodule "src/libs/pytorch3d"]
	path = src/libs/pytorch3d
	url = https://github.com/facebookresearch/pytorch3d.git
[submodule "src/libs/gaussian_wrapping/submodules/Depth-Anything-V2"]
	path = src/libs/gaussian_wrapping/submodules/Depth-Anything-V2
	url = https://github.com/DepthAnything/Depth-Anything-V2.git
```

Note the section names in `[submodule "..."]` must be updated too, not just the `path` lines. If
git left the old names, rewrite the file to match the block above verbatim, then
`git add .gitmodules`.

Expected `.git` files (these are one level deeper than before, so **four** `../` plus `libs/`):

```
src/libs/pytorch3d/.git   → gitdir: ../../../.git/modules/src/libs/pytorch3d
src/libs/light_glue/.git  → gitdir: ../../../.git/modules/src/libs/light_glue
```

Expected `core.worktree` (was `../../../../src/<name>`, now one level deeper):

```
.git/modules/src/libs/pytorch3d/config   → worktree = ../../../../../src/libs/pytorch3d
.git/modules/src/libs/light_glue/config  → worktree = ../../../../../src/libs/light_glue
```

Repair any that are wrong, using the same pattern as Step 3b, then verify:

```bash
git -C src/libs/pytorch3d  rev-parse HEAD   # 61cc79aa340412c33407771bc97236ccd9ee1548
git -C src/libs/light_glue rev-parse HEAD   # eb42fee2d71449efb0aa5c10549752b5d75384d8
git submodule status                        # 3 clean lines
git submodule foreach --recursive 'echo OK $sm_path'   # must not error
```

---

## Step 5 — Update `.gitignore`

The existing rule `src/*/__pycache__/` (line 8) matched `src/2dgs/__pycache__/` etc. After the
move those live at `src/libs/2dgs/__pycache__/`, one level deeper. Add a matching rule directly
below line 8 — **keep the existing line**, since `src/pipeline/` and `src/utils/` still need it:

```
src/*/__pycache__/
src/libs/*/__pycache__/
```

Verify:

```bash
grep -n "__pycache__" .gitignore
```

---

## Step 6 — Update the four `pipeline/reconstruction/run_*.py` path constants

These are the only code changes in the whole task. Each file has exactly one line to edit.

| File | Line | Old | New |
|------|------|-----|-----|
| `pipeline/reconstruction/run_2dgs.py` | 26 | `TWODGS_DIR = REPO_ROOT / "src" / "2dgs"` | `TWODGS_DIR = REPO_ROOT / "src" / "libs" / "2dgs"` |
| `pipeline/reconstruction/run_sugar.py` | 27 | `SUGAR_DIR = REPO_ROOT / "src" / "sugar"` | `SUGAR_DIR = REPO_ROOT / "src" / "libs" / "sugar"` |
| `pipeline/reconstruction/run_pgsr.py` | 32 | `PGSR_DIR = REPO_ROOT / "src" / "pgsr"` | `PGSR_DIR = REPO_ROOT / "src" / "libs" / "pgsr"` |
| `pipeline/reconstruction/run_gw.py` | 30 | `GW_DIR = REPO_ROOT / "src" / "gaussian_wrapping"` | `GW_DIR = REPO_ROOT / "src" / "libs" / "gaussian_wrapping"` |

Apply with `sed`:

```bash
sed -i 's|REPO_ROOT / "src" / "2dgs"|REPO_ROOT / "src" / "libs" / "2dgs"|' \
  pipeline/reconstruction/run_2dgs.py
sed -i 's|REPO_ROOT / "src" / "sugar"|REPO_ROOT / "src" / "libs" / "sugar"|' \
  pipeline/reconstruction/run_sugar.py
sed -i 's|REPO_ROOT / "src" / "pgsr"|REPO_ROOT / "src" / "libs" / "pgsr"|' \
  pipeline/reconstruction/run_pgsr.py
sed -i 's|REPO_ROOT / "src" / "gaussian_wrapping"|REPO_ROOT / "src" / "libs" / "gaussian_wrapping"|' \
  pipeline/reconstruction/run_gw.py
```

**Do not change** `REPO_ROOT = Path(__file__).resolve().parents[2]` in any of these files. Those
scripts stay at `pipeline/reconstruction/`, so `parents[2]` still resolves to the repo root.

Verify every derived script path now resolves on disk:

```bash
python - <<'PY'
from pathlib import Path
R = Path("/home/srizvi63.gatech/porto-photogrammetry")
expect = [
    R/"src/libs/2dgs/train.py",
    R/"src/libs/2dgs/render.py",
    R/"src/libs/sugar/gaussian_splatting/train.py",
    R/"src/libs/sugar/train.py",
    R/"src/libs/pgsr/train.py",
    R/"src/libs/pgsr/render.py",
    R/"src/libs/gaussian_wrapping/train.py",
    R/"src/libs/gaussian_wrapping/pivot_based_mesh_extraction.py",
    R/"src/libs/gaussian_wrapping/texture_mesh.py",
]
bad = [p for p in expect if not p.exists()]
print("MISSING:", bad or "none - all good")
PY
```

`pipeline/sfm/run_vggt_to_colmap.py` needs **no change**: its `from vggt.* import ...` lines
resolve through the installed `vggt` package, not through a filesystem path.

---

## Step 7 — Update `scripts/setup_common.sh`

Seven path occurrences across lines 51–126. All are `src/<backend>/...` prefixes that gain
`libs/`. A targeted `sed` handles all of them in one pass without touching the many
`BACKENDS`-name matches (`sugar`, `2dgs`, `pgsr`, `gw`) that are *not* paths:

```bash
sed -i -E 's|(^|[[:space:]"])src/(2dgs|pgsr|sugar|gaussian_wrapping|pytorch3d|light_glue|vggt)|\1src/libs/\2|g' \
  scripts/setup_common.sh
```

> If that alternation is awkward to escape in the executing shell, edit the seven sites by hand
> instead — the exact expected result is listed below.

Expected result, line by line:

| Line | After edit |
|------|-----------|
| 51 | `git submodule update --init src/libs/light_glue src/libs/pytorch3d` |
| 63 | `$PIP install -e src/libs/vggt       --no-build-isolation` |
| 64 | `$PIP install -e src/libs/light_glue --no-build-isolation` |
| 72 | `    $PIP install -e src/libs/pytorch3d --no-build-isolation` |
| 94 | `    build src/libs/sugar/gaussian_splatting/submodules/diff-gaussian-rasterization` |
| 95 | `    build src/libs/sugar/gaussian_splatting/submodules/simple-knn` |
| 99 | `    build src/libs/2dgs/submodules/diff-surfel-rasterization ;;` |
| 102 | `    build src/libs/pgsr/submodules/diff-plane-rasterization ;;` |
| 105–108 | `    build src/libs/gaussian_wrapping/submodules/{diff-gaussian-rasterization-gw,diff-gaussian-rasterization-ms,fused-ssim,warp-patch-ncc}` |
| 126 | `        TETRA="src/libs/gaussian_wrapping/submodules/tetra_triangulation"` |

**Leave alone** in this file:

- Line 12 `BACKENDS="${BACKENDS:-sugar 2dgs pgsr gw}"` — backend *names*, not paths.
- Line 4 comment listing the same names.
- Lines 93/98/101/104 `case " $BACKENDS " in *" sugar "*)` — matching on names.
- Lines 66–70 `pytorch3d` as a *pip package name* (`$PIP install ... pytorch3d -f "$PYTORCH3D_WHEEL"`).
- Line 146 `"vggt": "vggt", "pytorch3d": "pytorch3d"` in the import-verification dict — these are
  **module names**, not paths.

Verify no stale path survived and no name got corrupted:

```bash
grep -nE "src/(2dgs|pgsr|sugar|gaussian_wrapping|pytorch3d|light_glue|vggt)" scripts/setup_common.sh
# expect: NO output

grep -nE "src/libs/" scripts/setup_common.sh   # expect 11 lines
grep -n 'BACKENDS="${BACKENDS' scripts/setup_common.sh   # unchanged
bash -n scripts/setup_common.sh                          # syntax OK, no output
```

The per-GPU wrappers (`setup_a100.sh`, `setup_b200.sh`, `setup_h100.sh`, `setup_l40s.sh`,
`setup_rtx_pro_6000.sh`, `auto_setup.sh`) contain **no `src/` paths** — confirmed by grep. They
only export env vars and dispatch to `setup_common.sh`. Do not edit them.

---

## Step 8 — Update `README.md`

Two regions.

### 8a. Manual install block (lines 85–105)

Add `libs/` to each path. Expected result:

```
    src/libs/sugar/gaussian_splatting/submodules/diff-gaussian-rasterization \
    src/libs/sugar/gaussian_splatting/submodules/simple-knn \
    src/libs/light_glue \
    src/libs/pytorch3d \
    src/libs/2dgs/submodules/diff-surfel-rasterization \
    src/libs/pgsr/submodules/diff-plane-rasterization \
    src/libs/gaussian_wrapping/submodules/diff-gaussian-rasterization-gw \
    src/libs/gaussian_wrapping/submodules/diff-gaussian-rasterization-ms \
    src/libs/gaussian_wrapping/submodules/fused-ssim \
    src/libs/gaussian_wrapping/submodules/warp-patch-ncc \
```

Line 98: `python -m pip install -e src/libs/vggt --no-build-isolation`

Line 105: `cd src/libs/gaussian_wrapping/submodules/tetra_triangulation`

### 8b. Repo-structure tree (around lines 133–162)

Re-indent the seven backend entries one level under a new `libs/` node. Replace the block that
currently begins `├── src/` and runs through `│   └── pytorch3d/ ...` with:

```
├── src/
│   ├── libs/                  # Third-party backends (vendored + submodules)
│   │   ├── vggt/              # VGGT model (Meta)
│   │   │   └── vggt/          #   Importable Python package
│   │   │       ├── models/    #     VGGT, Aggregator
│   │   │       ├── heads/     #     Camera, depth, point, track heads
│   │   │       ├── layers/    #     Attention, RoPE, patch embedding
│   │   │       ├── utils/     #     Loading, pose encoding, geometry
│   │   │       └── dependency/#     COLMAP conversion, tracking
│   │   ├── sugar/             # SuGaR (vendored)
│   │   │   ├── gaussian_splatting/ # Embedded vanilla 3DGS
│   │   │   ├── sugar_trainers/     # Coarse + refined training
│   │   │   ├── sugar_extractors/   # Mesh extraction
│   │   │   └── sugar_scene/        # SuGaR model definition
│   │   ├── 2dgs/              # 2D Gaussian Splatting
│   │   │   ├── gaussian_renderer/  # Surfel rasterizer
│   │   │   ├── scene/              # Scene + Gaussian model
│   │   │   └── utils/              # Mesh extraction (TSDF + marching cubes)
│   │   ├── pgsr/              # PGSR
│   │   │   ├── gaussian_renderer/  # Plane rasterizer
│   │   │   ├── scene/              # Scene + Gaussian + AppModel
│   │   │   └── utils/              # Loss functions, graphics
│   │   ├── gaussian_wrapping/ # Gaussian Wrapping (Blobs to Spokes)
│   │   │   ├── gaussian_renderer/  # ours/radegs/sof rasterizers
│   │   │   ├── extraction/         # Pivot sampling + mesh extraction
│   │   │   ├── regularization/     # Normal-field, multiview, MILo, SDF
│   │   │   ├── scene/              # Scene + GaussianModel + Mesh
│   │   │   ├── scripts/            # End-to-end driver scripts
│   │   │   └── submodules/         # CUDA rasterizers + tetra triangulation
│   │   ├── light_glue/        # LightGlue (submodule)
│   │   └── pytorch3d/         # PyTorch3D (submodule)
│   ├── pipeline/              # Legacy first-generation pipeline
│   └── utils/
├── scripts/                   # Per-GPU environment installers
```

Note this also corrects the stale `sugar/  # SuGaR (submodule)` annotation — `src/sugar` is
vendored in-tree, not a submodule (`.gitmodules` lists only three, and `sugar` is not among
them). It also surfaces `src/pipeline/` and `src/utils/`, which the old tree omitted.

Verify:

```bash
grep -nE "src/(2dgs|pgsr|sugar|gaussian_wrapping|pytorch3d|light_glue|vggt)" README.md
# expect: NO output
```

---

## Step 9 — Update `.claude/CLAUDE.md`

Two edits.

**9a.** Lines 72–73, the submodule gotcha bullet:

```markdown
- Only `src/libs/light_glue`, `src/libs/pytorch3d`, and
  `src/libs/gaussian_wrapping/submodules/Depth-Anything-V2` are git submodules — `src/libs/sugar`
  and the other backends are vendored in-tree.
```

**9b.** Add a new bullet to the "Gotchas worth knowing before you type" list, immediately after
the existing `src/pipeline/` legacy bullet, recording the new layout:

```markdown
- All third-party backends live under `src/libs/` (`2dgs`, `pgsr`, `sugar`, `gaussian_wrapping`,
  `vggt`, `light_glue`, `pytorch3d`). First-party code is `pipeline/` plus `src/pipeline/`
  (legacy) and `src/utils/`.
```

Keep both edits terse — `.claude/CLAUDE.md` explicitly states it is orientation-only and must
stay short. Do not expand it further.

---

## Step 10 — Update `.claude/MEMORY/*.md`

Six files carry `src/<backend>` paths. These are documentation; update paths only, leave all
surrounding prose and structure intact.

| File | Sites |
|------|-------|
| `backend-2dgs.md` | line 1 heading `` `src/2dgs/` ``; lines 9, 29 `cd src/2dgs` |
| `backend-pgsr.md` | line 1 heading; lines 9, 30 `cd src/pgsr` |
| `backend-sugar.md` | line 1 heading; lines 6, 8 prose paths; line 59 `cd src/sugar` |
| `backend-vggt.md` | lines 1, 5, 7, 26, 49 |
| `backend-gaussian-wrapping.md` | lines 1, 9 |
| `environment-and-gpu.md` | lines 52–54 (editable-install prose), 58–61 (submodule list) |
| `pipeline-reconstruction.md` | line 46 (`src/gaussian_wrapping/`) |

A single mechanical pass covers every site:

```bash
sed -i -E 's|src/(2dgs|pgsr|sugar|gaussian_wrapping|pytorch3d|light_glue|vggt)|src/libs/\1|g' \
  .claude/CLAUDE.md .claude/MEMORY/*.md
```

Run this **instead of** hand-editing 9a/10 if preferred, then apply the 9b addition manually.
Guard against double-application: if a path already reads `src/libs/...`, the regex will not
match it again (it anchors on `src/` immediately followed by a backend name), so the sed is
idempotent.

Verify:

```bash
grep -rnE "src/(2dgs|pgsr|sugar|gaussian_wrapping|pytorch3d|light_glue|vggt)" .claude/
# expect: NO output
```

**Do not** update `.claude/MEMORY/backend-sugar.md`'s heading to claim sugar is a submodule; and
in `environment-and-gpu.md` keep the existing correct statement that `src/libs/sugar` is
vendored, not a submodule.

---

## Step 11 — Root-level markdown notes (lower priority, do them anyway)

Three untracked scratch documents reference old paths. They are not wired into any tooling, but
leaving them stale is misleading:

- `commands.md` — lines 26, 30 (`src/sugar/...`).
- `gaussian-wrapping-inclusion.md` — ~60 markdown links, all `src/gaussian_wrapping/...`.
- `pr-description.md` — lines 27, 91.

```bash
sed -i -E 's|src/(2dgs|pgsr|sugar|gaussian_wrapping|pytorch3d|light_glue|vggt)|src/libs/\1|g' \
  commands.md gaussian-wrapping-inclusion.md pr-description.md
```

These files are untracked, so this does not affect the commit. `gaussian-wrapping-inclusion.md`
also describes a nested `gaussian_wrapping/gaussian_wrapping/` directory that does not exist on
disk — that pre-existing inaccuracy is **out of scope**; only rewrite the `src/` prefix.

---

## Step 12 — Repair the editable installs (must run on a GPU/compute node)

`vggt`, `lightglue`, and `pytorch3d` were pip-installed with `-e` pointing at the **old** source
paths. An editable install records an absolute path to the source tree; after the move those
paths no longer exist, so `import vggt` will fail (or silently resolve to a stale location) until
the installs are redone.

This **cannot be done or verified on the login node** — the `augenblick` conda env is not present
here. On a compute node with the env active:

```bash
conda activate augenblick
cd /home/srizvi63.gatech/porto-photogrammetry

python -m pip uninstall -y vggt lightglue pytorch3d

python -m pip install -e src/libs/vggt       --no-build-isolation
python -m pip install -e src/libs/light_glue --no-build-isolation
python -m pip install -e src/libs/pytorch3d  --no-build-isolation   # slow source build
```

> If `pytorch3d` was originally installed from a prebuilt wheel (`PYTORCH3D_WHEEL` set), it is a
> **non-editable** install and is unaffected by the move — skip its uninstall/reinstall. Check
> first with `python -c "import pytorch3d; print(pytorch3d.__file__)"`: a path under
> `site-packages/` means wheel (leave it); a path under `src/` means editable (redo it).

The compiled CUDA rasterizers (`diff_gaussian_rasterization`, `simple_knn`,
`diff_surfel_rasterization`, `diff_plane_rasterization`, the four GW ones) were installed
**non-editable** by `setup_common.sh`'s `build()` helper (`$PIP install "$1"`, no `-e`), so their
compiled artifacts already live in `site-packages` and are **unaffected by the move**. Do not
rebuild them.

`tetra_triangulation` **is** installed editable (`pip install -e .` at
`setup_common.sh:132`), so it does need repointing:

```bash
python -c "import tetranerf" 2>&1 | head -1   # if it fails:
cd src/libs/gaussian_wrapping/submodules/tetra_triangulation && \
  python -m pip install -e . --no-build-isolation && cd -
```

Verify on the compute node:

```bash
python -c "import vggt, lightglue; print(vggt.__file__); print(lightglue.__file__)"
# both paths must contain /src/libs/
```

---

## Step 13 — Full verification sweep

### 13a. No stale paths anywhere

```bash
cd /home/srizvi63.gatech/porto-photogrammetry
grep -rn --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=build \
  --exclude-dir=src/libs \
  -E "src/(2dgs|pgsr|sugar|gaussian_wrapping|pytorch3d|light_glue|vggt)" . \
  | grep -v "src/libs/"
```

Expected: **no output**. Any hit inside `src/libs/**` itself is a third-party file's own internal
documentation and is out of scope — ignore those.

### 13b. Git state is coherent

```bash
git ls-files | wc -l
diff <(cat /tmp/baseline_tracked_count.txt) <(git ls-files | wc -l) && echo "COUNT MATCHES"

git status --short | grep -c "^R"     # ~3800 renames staged
git submodule status                  # 3 clean lines, no '-' or 'U' prefix
git submodule foreach --recursive 'git rev-parse HEAD' && echo "SUBMODULES OK"
```

The tracked-file count **must be identical** to the baseline. A drop means files were lost in a
`git mv`; a rise means something untracked got added.

### 13c. Python entry points still parse and resolve

```bash
python -m py_compile pipeline/reconstruction/run_2dgs.py \
                     pipeline/reconstruction/run_sugar.py \
                     pipeline/reconstruction/run_pgsr.py \
                     pipeline/reconstruction/run_gw.py && echo "COMPILE OK"
```

Then re-run the path-resolution snippet from Step 6 — all nine script paths must exist.

Each wrapper's `--help` is a good end-to-end smoke test but **requires the conda env** (they
import torch transitively in some cases). On a compute node:

```bash
python pipeline/reconstruction/run_2dgs.py  --help
python pipeline/reconstruction/run_sugar.py --help
python pipeline/reconstruction/run_pgsr.py  --help
python pipeline/reconstruction/run_gw.py    --help
```

### 13d. Shell scripts parse

```bash
for f in scripts/*.sh; do bash -n "$f" || echo "SYNTAX FAIL: $f"; done
echo "SHELL OK"
```

### 13e. Fresh-clone sanity for submodules

The strongest check that `.gitmodules` is correct — do this **after committing** (Step 14),
in a scratch directory:

```bash
git clone --recurse-submodules /home/srizvi63.gatech/porto-photogrammetry \
  /tmp/libs-move-clonetest
ls /tmp/libs-move-clonetest/src/libs/
ls /tmp/libs-move-clonetest/src/libs/pytorch3d/ | head -3
ls /tmp/libs-move-clonetest/src/libs/light_glue/ | head -3
ls /tmp/libs-move-clonetest/src/libs/gaussian_wrapping/submodules/Depth-Anything-V2/ | head -3
rm -rf /tmp/libs-move-clonetest
```

All three submodule directories must be **non-empty**. An empty one means that submodule's
`.gitmodules` stanza is wrong.

### 13f. End-to-end functional check (compute node, optional but recommended)

Run one short reconstruction against an existing SfM scene to confirm subprocess invocation
still works through the moved paths — 2DGS is the quickest:

```bash
python pipeline/reconstruction/run_2dgs.py <existing-sfm-scene> /tmp/libs-move-smoke \
  --iterations 1000
```

Confirm it gets past process launch (the point of failure a bad path would cause is immediate).

---

## Step 14 — Commit

Stage the rename plus the reference updates. **Do not** stage the four pre-existing modified
`.cu` files or the untracked scratch markdown — they are unrelated to this change:

```bash
git add -A src/libs .gitmodules .gitignore \
  pipeline/reconstruction/run_2dgs.py \
  pipeline/reconstruction/run_sugar.py \
  pipeline/reconstruction/run_pgsr.py \
  pipeline/reconstruction/run_gw.py \
  scripts/setup_common.sh README.md .claude/

git status --short | grep -vE "^R" | head -20   # review what is staged beyond renames
```

If the four modified `.cu` files got staged as part of the `src/libs` rename, unstage just their
content change while keeping the rename:

```bash
git restore --staged --worktree=false \
  src/libs/gaussian_wrapping/submodules/diff-gaussian-rasterization-ms/conv.cu 2>/dev/null || true
```

> In practice it is simpler and acceptable to let the rename carry those files as-is: the rename
> itself must be staged, and git records the content modification alongside it. If separating
> them proves fiddly, **leave them together and note it in the commit message** rather than
> risking losing the CUDA edits.

Commit:

```bash
git commit -m "$(cat <<'EOF'
Move third-party backends under src/libs/

Relocate the seven vendored/submodule backends (2dgs, pgsr, sugar,
gaussian_wrapping, vggt, light_glue, pytorch3d) from src/ into src/libs/
so first-party code (src/pipeline, src/utils) is no longer buried among
~3800 third-party files.

Pure relocation via git mv; no file contents inside the moved trees change.
Updates .gitmodules (including the nested Depth-Anything-V2 stanza),
.gitignore, the four pipeline/reconstruction/run_*.py path constants,
scripts/setup_common.sh, README.md, and .claude/ docs.

Editable installs (vggt, lightglue, pytorch3d, tetra_triangulation) must be
reinstalled from their new paths on each compute environment; the compiled
non-editable CUDA rasterizers are unaffected.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

Then run the fresh-clone check (13e) against the committed state.

Once everything verifies, drop the safety tag:

```bash
git tag -d pre-libs-move-backup
```

---

## Summary of every file changed outside the moved trees

| File | Change |
|------|--------|
| `.gitmodules` | 3 stanzas: section names + `path` values gain `libs/` |
| `.gitignore` | add `src/libs/*/__pycache__/` |
| `pipeline/reconstruction/run_2dgs.py` | 1 line (`TWODGS_DIR`) |
| `pipeline/reconstruction/run_sugar.py` | 1 line (`SUGAR_DIR`) |
| `pipeline/reconstruction/run_pgsr.py` | 1 line (`PGSR_DIR`) |
| `pipeline/reconstruction/run_gw.py` | 1 line (`GW_DIR`) |
| `scripts/setup_common.sh` | 11 path sites (lines 51, 63, 64, 72, 94, 95, 99, 102, 105–108, 126) |
| `README.md` | install block (85–105) + repo tree (133–162) |
| `.claude/CLAUDE.md` | submodule bullet + one new layout bullet |
| `.claude/MEMORY/*.md` | 6 files, path-only updates |
| `commands.md`, `gaussian-wrapping-inclusion.md`, `pr-description.md` | untracked notes, path-only |
| `.git/modules/**/config`, `src/libs/*/.git` | plumbing, verified/repaired in Steps 3–4 |

**Explicitly unchanged:** `pipeline/sfm/*` (package imports, not paths), `src/pipeline/*`,
`src/utils/*`, `baseline/*`, `tests/*`, `requirements.txt`, `src/libs/vggt/pyproject.toml`, and
the per-GPU `scripts/setup_*.sh` wrappers.

---

## Risk register

| Risk | Mitigation |
|------|-----------|
| Nested Depth-Anything-V2 submodule breaks (most likely failure) | Step 3b checks all three plumbing artifacts explicitly; 3c fails loudly; 13e fresh-clone is the final proof |
| Editable installs point at dead paths → `import vggt` fails | Step 12, run on a compute node; cannot be verified from the login node |
| `sed` in Step 7 corrupts `BACKENDS` names or the import-check dict | Step 7's verify greps confirm `BACKENDS` line intact and `bash -n` passes |
| Pre-existing modified `.cu` files lost during `git mv` | Step 0 documents them; Step 3d re-checks them at the new paths |
| Files silently dropped in a large `git mv` | Step 0 baseline count vs Step 13b exact comparison |
| Half-finished move leaves repo unusable | `pre-libs-move-backup` tag from Step 0 restores everything |
