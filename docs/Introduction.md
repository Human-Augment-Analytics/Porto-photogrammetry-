# Augenblick: Introduction

Augenblick is a photogrammetry pipeline that reconstructs 3D geometry from
multi-view photographs. It chains a feed-forward vision transformer
(**VGGT** — Visual Geometry Grounded Transformer) with downstream neural
surface-reconstruction backends (**NeuS2**, **3D Gaussian Splatting**, and
**SuGaR**) to go from a set of unposed images to textured meshes with minimal
manual intervention.

## Motivation

Classical photogrammetry relies on hand-crafted feature detectors (SIFT, ORB)
and incremental bundle adjustment (COLMAP) to recover camera poses and sparse
point clouds before dense reconstruction. This works well for textured,
Lambertian scenes but is slow, brittle on low-texture or repetitive surfaces,
and requires careful parameter tuning. Augenblick replaces the traditional
front-end with VGGT that jointly predicts camera poses, dense depth maps,
3D world points, and point tracks in one forward pass, and then feeds these
predictions into modern neural renderers.

## High-Level Pipeline

```
              Input images
                  │
                  ▼
┌──────────────────────────────────────────────┐
│  VGGT Inference  (src/vggt/, src/pipeline/)  │
│  ─ camera poses  (extrinsics + intrinsics)   │
│  ─ dense depth maps                          │
│  ─ 3D world-point cloud with confidence      │
│  ─ point tracks (optional)                   │
└─────────────────┬────────────────────────────┘
                  │
          ┌───────┴────────┐
          ▼                ▼
      COLMAP export    NeuS2 export
      (binary fmt)     (transform.json)
          │                │
          ▼                ▼
      3D Gaussian      NeuS2 neural
      Splatting        surface recon
          │
          ▼
      SuGaR mesh
      extraction
```

Three export paths are available from a single VGGT run:

| Path | Export format | Downstream tool | Output |
|------|-------------|-----------------|--------|
| **GLB preview** | `.glb` scene | — | Quick 3D point-cloud + cameras for visual inspection |
| **COLMAP → 3DGS → SuGaR** | COLMAP binary (`cameras.bin`, `images.bin`, `points3D.bin`) | Gaussian Splatting, then SuGaR | High-quality textured mesh (`.obj` / `.ply`) |
| **NeuS2** | `transform.json` + images | NeuS2 | Watertight neural implicit mesh |

## Core Model: VGGT

VGGT (Wang et al., 2025) is a ~1 billion-parameter transformer that takes an
*unordered* set of images and produces dense 3D predictions *without* any
explicit feature matching or RANSAC.

### Architecture

| Component | Role | Key detail |
|-----------|------|------------|
| **Patch embedding** | Tokenize each image | Uses a frozen DINOv2 ViT (large or giant2) encoder |
| **Alternating-attention aggregator** | Fuse information within and across frames | Alternates *frame blocks* (per-image self-attention) and *global blocks* (cross-image attention over all patches). Uses 2-D Rotary Position Embeddings (RoPE). Includes learnable camera tokens and register tokens. |
| **Camera head** | Predict camera pose per frame | Iterative refinement (4 rounds) with Adaptive Layer Norm (AdaLN). Outputs a 9-D encoding: translation (3) + quaternion (4) + field-of-view (2). |
| **DPT depth/point heads** | Predict dense depth and 3D world points | Dense Prediction Transformer (DPT) architecture with multi-scale feature fusion from intermediate aggregator layers. Outputs per-pixel depth, world-point XYZ, and confidence. |
| **Track head** | Track points across frames (optional) | Correlation-based tracker derived from VGGSfM, with coarse-to-fine iterative refinement. |

The model is loaded from Hugging Face Hub (`facebook/VGGT-1B`) and runs
inference in mixed precision (bfloat16 / float16). The pipeline includes
`torch.compile` optimizations targeting NVIDIA Blackwell (B200, sm_100) GPUs.

### Outputs

Given *S* input images of size *H × W*, VGGT returns:

- `pose_enc` — camera pose encoding, shape `[B, S, 9]`
- `depth` — per-pixel depth maps, shape `[B, S, H, W, 1]`
- `world_points` — per-pixel 3D coordinates, shape `[B, S, H, W, 3]`
- `world_points_conf` — per-pixel confidence, shape `[B, S, H, W]`
- `images` — preprocessed input images

These are serialized as a pickle file and consumed by the format converters.

## Downstream Reconstruction Backends

### NeuS2

NeuS2 (Wang et al., 2023) learns a neural signed-distance function from posed
multi-view images and extracts watertight meshes via marching cubes. The
converter (`src/pipeline/vggt_to_neus2_converter.py`) transforms VGGT
predictions into `transform.json` with:

- Camera-to-world 4×4 matrices (OpenCV → NeRF coordinate convention, with the
  `from_na` 180° x-rotation applied)
- 4×4 intrinsic matrices
- Scene scale and offset computed from the point-cloud bounding box

### 3D Gaussian Splatting + SuGaR

The COLMAP export path (`src/pipeline/vggt_to_colmap.py`) writes standard
COLMAP binary files so that 3D Gaussian Splatting can be trained directly.
The converter:

- Filters world points by confidence (percentile threshold, default 70th)
- Removes outliers (1st–99th percentile per axis)
- Subsamples spatially to cap point count (default 200 000)
- Writes `cameras.bin`, `images.bin`, `points3D.bin`

After 3DGS training, **SuGaR** (Surface-Aligned Gaussian Splatting) extracts a
high-quality polygonal mesh from the Gaussian field. The full three-stage
pipeline is orchestrated by `src/pipeline/run_sugar_pipeline.py` (Python) or
`scripts/run_vggt_sugar.sbatch` (SLURM).

Quality presets for the SuGaR path:

| Preset | 3DGS iterations | SuGaR density threshold |
|--------|---------------:|------------------------:|
| draft  | 7 000 | 1.0 |
| medium | 15 000 | 0.5 |
| high   | 30 000 | 0.3 |

## Supporting Utilities

| Module | Purpose |
|--------|---------|
| `src/visual_util.py` | Convert predictions to a `.glb` trimesh scene; sky segmentation via an ONNX model to mask out sky points |
| `src/pipeline/conversion_utils.py` | Validate `transform.json` structure, visualize camera poses, check image paths |
| `src/vggt/utils/load_fn.py` | Load and preprocess images (resize to 518 px, normalize) |
| `src/vggt/utils/geometry.py` | Depth-map unprojection, SE(3) inverse, coordinate transforms |
| `src/vggt/utils/pose_enc.py` | Encode/decode between pose vectors and extrinsic/intrinsic matrices |
| `scripts/setup_sugar_b200.sh` | One-time setup: clone and compile 3DGS + SuGaR for Blackwell GPUs |
| `scripts/run_vggt_sugar.sbatch` | SLURM batch job running the full three-stage pipeline on a B200 node |

## Repository Layout

```
├── src/
│   ├── vggt/                   # VGGT model implementation
│   │   ├── models/             #   main model (vggt.py) + aggregator
│   │   ├── heads/              #   camera, depth/point (DPT), track heads
│   │   ├── layers/             #   ViT, attention, RoPE, MLP, etc.
│   │   ├── utils/              #   image loading, geometry, pose encoding
│   │   └── dependency/         #   VGGSfM tracker, keypoint extraction
│   ├── pipeline/               # Pipeline orchestration and format converters
│   │   ├── run_pipeline.py     #   main CLI entry point for VGGT inference
│   │   ├── run_sugar_pipeline.py  # end-to-end VGGT → 3DGS → SuGaR
│   │   ├── vggt_to_colmap.py   #   VGGT → COLMAP binary export
│   │   └── vggt_to_neus2_converter.py  # VGGT → NeuS2 export
│   └── visual_util.py          # GLB export and sky segmentation
├── scripts/
│   ├── setup_sugar_b200.sh     # Build dependencies for Blackwell GPUs
│   └── run_vggt_sugar.sbatch   # SLURM job for the full pipeline
├── docs/                       # Documentation
├── environment.yml             # Conda environment (Python 3.10, PyTorch 2.6, CUDA 12.8)
├── LICENSE.txt                 # CC BY-NC 4.0
└── README.md
```

## Key Dependencies

| Package | Role |
|---------|------|
| PyTorch 2.6 + CUDA 12.8 | Deep-learning runtime |
| torchvision | Image transforms |
| Hugging Face Hub | Model weights download (`facebook/VGGT-1B`) |
| DINOv2 (via ViT) | Frozen patch-embedding backbone inside VGGT |
| einops | Tensor reshaping |
| NumPy, SciPy, OpenCV | Numerical and image processing |
| trimesh, Open3D | 3D mesh / point-cloud handling |
| ONNX Runtime | Sky-segmentation inference |
| 3D Gaussian Splatting | Gaussian-field training (external repo) |
| SuGaR | Mesh extraction from Gaussian fields (external repo) |
| NeuS2 | Neural implicit surface reconstruction (external repo / submodule) |

## References

- **VGGT**: Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., &
  Novotny, D. (2025). *Visual Geometry Grounded Transformer.* arXiv:2503.11651.
- **NeuS2**: Wang, Y., Han, Q., Habermann, M., Daniilidis, K., Theobalt, C., &
  Liu, L. (2023). *NeuS2: Fast Learning of Neural Implicit Surfaces for
  Multi-view Reconstruction.* ICCV 2023.
- **3D Gaussian Splatting**: Kerbl, B., Kopanas, G., Leimkühler, T., &
  Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field
  Rendering.* SIGGRAPH 2023.
- **SuGaR**: Guédon, A., & Lepetit, V. (2024). *SuGaR: Surface-Aligned
  Gaussian Splatting for Efficient 3D Mesh Reconstruction.* CVPR 2024.
- **DINOv2**: Oquab, M., et al. (2024). *DINOv2: Learning Robust Visual
  Features without Supervision.* TMLR 2024.
- **DPT**: Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). *Vision
  Transformers for Dense Prediction.* ICCV 2021.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
Non-commercial use only; attribution required.
