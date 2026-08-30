#!/bin/bash
# Shared installer behind the per-GPU wrappers (setup_l40s/a100/b200.sh, auto_setup.sh).
# Wrappers must export: GPU_LABEL, GPU_ARCH, CUDA_MODULE, TORCH_SPEC, TORCH_INDEX_URL.
# Optional: BACKENDS (default "sugar 2dgs pgsr gw"), SKIP_TETRA=1, PYTORCH3D_WHEEL=<url>.
set -euo pipefail

: "${GPU_LABEL:?set by wrapper}"
: "${GPU_ARCH:?set by wrapper}"
: "${CUDA_MODULE:?set by wrapper}"
: "${TORCH_SPEC:?set by wrapper}"
: "${TORCH_INDEX_URL:?set by wrapper}"
BACKENDS="${BACKENDS:-sugar 2dgs pgsr gw}"
SKIP_TETRA="${SKIP_TETRA:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

banner() { printf '\n========== %s ==========\n' "$*"; }
have()   { command -v "$1" >/dev/null 2>&1; }

banner "Augenblick setup — $GPU_LABEL (sm_$GPU_ARCH)"
echo "Repo:        $REPO_ROOT"
echo "CUDA module: $CUDA_MODULE"
echo "Torch:       $TORCH_SPEC  ($TORCH_INDEX_URL)"
echo "Backends:    $BACKENDS"

# --- CUDA toolkit + arch flags ------------------------------------------------
module load "$CUDA_MODULE" 2>/dev/null || echo "WARN: 'module load $CUDA_MODULE' failed (no Lmod?); ensure nvcc is on PATH."
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
if ! have nvcc; then echo "ERROR: nvcc not found after loading $CUDA_MODULE."; exit 1; fi
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
echo "nvcc:        $(nvcc --version | grep -oP 'release \K[0-9.]+')   CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# --- Python env sanity --------------------------------------------------------
if ! have python; then echo "ERROR: no 'python' on PATH. Create & activate a py3.10 conda env first."; exit 1; fi
PYV="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "python:      $(python --version) @ $(command -v python)"
[ "$PYV" = "3.10" ] || echo "WARN: Python $PYV (repo targets 3.10)."
PIP="python -m pip"

# numpy 2.x breaks the repo's pinned scipy/scikit-learn (C-ABI mismatch). Pin <2
# globally so no transitive dep (torch, fvcore, ...) silently pulls numpy 2.x.
NUMPY_CONSTRAINT="$(mktemp)"; echo "numpy<2" > "$NUMPY_CONSTRAINT"
export PIP_CONSTRAINT="$NUMPY_CONSTRAINT"
trap 'rm -f "$NUMPY_CONSTRAINT"' EXIT

# --- Submodules (light_glue + pytorch3d are git submodules) -------------------
banner "Submodules"
git submodule update --init src/libs/light_glue src/libs/pytorch3d

# --- 1. PyTorch (arch-specific wheel index) -----------------------------------
banner "1/7 PyTorch"
$PIP install $TORCH_SPEC --index-url "$TORCH_INDEX_URL"

# --- 2. PyPI dependencies (repo requirements.txt) -----------------------------
banner "2/7 requirements.txt"
$PIP install -r requirements.txt

# --- 3. Editable source packages: VGGT + LightGlue ----------------------------
banner "3/7 VGGT + LightGlue (editable)"
$PIP install -e src/libs/vggt       --no-build-isolation
$PIP install -e src/libs/light_glue --no-build-isolation

# --- 4. pytorch3d (source build, arch-agnostic; or prebuilt wheel) ------------
banner "4/7 pytorch3d"
if [ -n "${PYTORCH3D_WHEEL:-}" ]; then
    $PIP install fvcore iopath
    $PIP install --no-index --no-cache-dir pytorch3d -f "$PYTORCH3D_WHEEL"
else
    $PIP install -e src/libs/pytorch3d --no-build-isolation
fi

# --- 5. nvdiffrast (SuGaR texture export + GW mesh rasterisation) -------------
banner "5/7 nvdiffrast"
$PIP install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation || \
    echo "WARN: nvdiffrast install failed (texture export will fall back)."

# --- 6. Per-backend CUDA rasterizers (compiled at sm_$GPU_ARCH) ---------------
banner "6/7 Backend CUDA rasterizers"
# Clean any stale build/ first: leftover object files from an earlier build
# (e.g. a different GPU arch) are reused and silently ignore TORCH_CUDA_ARCH_LIST.
# NOTE: do not run two setups against the SAME checkout concurrently — they race
# on these build/ dirs. Use a separate checkout per parallel build.
build() {
    echo "-- building $1"
    rm -rf "$1/build" "$1"/*.egg-info
    $PIP install "$1" --no-cache-dir --no-build-isolation
}

# 3DGS base rasterizers — needed by SuGaR (and the vanilla 3DGS step).
case " $BACKENDS " in *" sugar "*)
    build src/libs/sugar/gaussian_splatting/submodules/diff-gaussian-rasterization
    build src/libs/sugar/gaussian_splatting/submodules/simple-knn
    ;;
esac
case " $BACKENDS " in *" 2dgs "*)
    build src/libs/2dgs/submodules/diff-surfel-rasterization ;;
esac
case " $BACKENDS " in *" pgsr "*)
    build src/libs/pgsr/submodules/diff-plane-rasterization ;;
esac
case " $BACKENDS " in *" gw "*)
    build src/libs/gaussian_wrapping/submodules/diff-gaussian-rasterization-gw
    build src/libs/gaussian_wrapping/submodules/diff-gaussian-rasterization-ms
    build src/libs/gaussian_wrapping/submodules/fused-ssim
    build src/libs/gaussian_wrapping/submodules/warp-patch-ncc
    ;;
esac

# --- 7. Gaussian Wrapping: CGAL / tetra_triangulation (fragile, optional) -----
case " $BACKENDS " in *" gw "*)
    if [ "$SKIP_TETRA" = "1" ]; then
        echo "Skipping tetra_triangulation (SKIP_TETRA=1)."
    else
        banner "7/7 tetra_triangulation (CGAL)"
        CONDA="${CONDA_EXE:-conda}"
        if have "$CONDA"; then
            "$CONDA" install -y cmake || true
            "$CONDA" install -y -c conda-forge gmp cgal || true
        else
            echo "WARN: conda not found; assuming cmake/gmp/cgal already present."
        fi
        export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
        TETRA="src/libs/gaussian_wrapping/submodules/tetra_triangulation"
        ( cd "$TETRA"
          cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
                  -DCGAL_DIR="${CONDA_PREFIX:-/usr}/lib/cmake/CGAL" \
                  -DTorch_DIR="$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share/cmake/Torch'))")" \
          && make \
          && python -m pip install -e . --no-build-isolation
        ) || echo "WARN: tetra_triangulation build failed (GW pivot extraction unavailable; other backends fine)."
    fi
    ;;
esac

# --- Verify -------------------------------------------------------------------
banner "Verifying imports"
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| dev cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "no-GPU-here")
checks = {
    "numpy": "numpy", "scipy.spatial": "scipy", "sklearn": "scikit-learn",
    "pycolmap": "pycolmap", "open3d": "open3d", "vggt": "vggt", "pytorch3d": "pytorch3d",
    "diff_gaussian_rasterization": "diff_gaussian_rasterization (SuGaR/3DGS)",
    "simple_knn._C": "simple_knn (SuGaR/3DGS)",
    "diff_surfel_rasterization": "diff_surfel_rasterization (2DGS)",
    "diff_plane_rasterization": "diff_plane_rasterization (PGSR)",
}
import numpy
if int(numpy.__version__.split(".")[0]) >= 2:
    print("WARN: numpy", numpy.__version__, "(>=2 breaks pinned scipy/sklearn ABI)")
ok, miss = [], []
for mod, label in checks.items():
    try: importlib.import_module(mod); ok.append(label)
    except Exception as e: miss.append(f"{label}: {type(e).__name__}")
print("OK   :", ", ".join(ok) or "none")
print("MISS :", " | ".join(miss) or "none (all good)")
PY

banner "Done — $GPU_LABEL environment ready"
echo "Note: GPU import checks only validate the arch of the node you ran on."
echo "If you compiled fat binaries for multiple arches, smoke-test the others via a short sbatch job."
