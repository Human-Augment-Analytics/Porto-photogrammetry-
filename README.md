# Augenblick

A two-stage photogrammetry pipeline for high-fidelity 3D object reconstruction from multi-view images. The pipeline pairs a Structure-from-Motion (SfM) initialiser with a Gaussian-based mesh extractor.

## Overview

Modern photogrammetry pipelines are two-stage: an SfM method first estimates camera parameters and a sparse point cloud, which then initialises a dense surface reconstruction step. This project evaluates how different SfM initialisations interact with Gaussian-primitive-based mesh extraction methods.

**Stage 1 – Structure-from-Motion:**

| Method | Description |
|--------|-------------|
| COLMAP | Classical incremental SfM with SIFT features and bundle adjustment |
| VGGT | Feed-forward transformer that regresses camera parameters and a point map in a single pass |
| VGGT + BA | VGGT output refined by VGGSfM tracking and bundle adjustment |

**Stage 2 – Gaussian Mesh Extraction:**

| Method | Description |
|--------|-------------|
| SuGaR | Surface-regularised 3D Gaussians with Poisson mesh extraction and UV texturing |
| 2DGS | Flat 2D Gaussian surfel disks with TSDF depth fusion |
| PGSR | Planar-consistent Gaussians with multi-view geometric/photometric losses and TSDF meshing |

All combinations are benchmarked on runtime and compared qualitatively against RealityScan (commercial) and Meshroom (open-source) baselines.

## Installation

### Requirements

- Linux (tested on Ubuntu / WSL2)
- Python 3.10
- CUDA-capable GPU (12 GB+ VRAM recommended)
- CUDA 12.8

### Setup

```bash
git clone --recursive <repository-url> porto-photogrammetry
cd porto-photogrammetry

# Create conda environment
conda create --name augenblick python=3.10
conda activate augenblick

# Initialise submodules (SuGaR, LightGlue, PyTorch3D)
git submodule update --init --recursive

# Install Python dependencies
python -m pip install -r requirements.txt

# Install PyTorch with CUDA (adjust URL for your CUDA version)
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130

# Install nvdiffrast (required by SuGaR)
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# Install CUDA submodules and local packages
python -m pip install \
    src/sugar/gaussian_splatting/submodules/diff-gaussian-rasterization \
    src/sugar/gaussian_splatting/submodules/simple-knn \
    src/light_glue \
    src/pytorch3d \
    src/2dgs/submodules/diff-surfel-rasterization \
    src/pgsr/submodules/diff-plane-rasterization \
    --no-build-isolation

# Install VGGT as editable package
python -m pip install -e src/vggt --no-build-isolation
```

VGGT model weights (~4 GB) are downloaded automatically from `facebook/VGGT-1B` on HuggingFace on first run.

## Project Structure

```
augenblick/
├── pipeline/                  # Canonical entry points
│   ├── preparation/           #   Dataset preparation scripts
│   │   └── prepare_uf_dataset.py
│   ├── sfm/                   #   Structure-from-Motion scripts
│   │   ├── run_vggt_to_colmap.py
│   │   └── run_colmap.sh
│   └── reconstruction/        #   Surface reconstruction scripts
│       ├── run_sugar.py
│       ├── run_2dgs.py
│       └── run_pgsr.py
├── src/
│   ├── vggt/                  # VGGT model (Meta)
│   │   └── vggt/              #   Importable Python package
│   │       ├── models/        #     VGGT, Aggregator
│   │       ├── heads/         #     Camera, depth, point, track heads
│   │       ├── layers/        #     Attention, RoPE, patch embedding
│   │       ├── utils/         #     Loading, pose encoding, geometry
│   │       └── dependency/    #     COLMAP conversion, tracking
│   ├── sugar/                 # SuGaR (submodule)
│   │   ├── gaussian_splatting/#   Embedded vanilla 3DGS
│   │   ├── sugar_trainers/    #   Coarse + refined training
│   │   ├── sugar_extractors/  #   Mesh extraction
│   │   └── sugar_scene/       #   SuGaR model definition
│   ├── 2dgs/                  # 2D Gaussian Splatting
│   │   ├── gaussian_renderer/ #   Surfel rasterizer
│   │   ├── scene/             #   Scene + Gaussian model
│   │   └── utils/             #   Mesh extraction (TSDF + marching cubes)
│   ├── pgsr/                  # PGSR
│   │   ├── gaussian_renderer/ #   Plane rasterizer
│   │   ├── scene/             #   Scene + Gaussian + AppModel
│   │   └── utils/             #   Loss functions, graphics
│   ├── light_glue/            # LightGlue (submodule)
│   └── pytorch3d/             # PyTorch3D (submodule)
└── CLAUDE.md                  # Detailed codebase documentation
```

## Usage

All reconstruction backends consume a common COLMAP-format scene directory:

```
<scene>/
├── images/         # Source images
├── masks/          # Optional: binary PNGs (white = foreground)
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

### Step 0: Data Preparation

If your source data has mixed images and masks in a flat directory:

```bash
python pipeline/preparation/prepare_uf_dataset.py /path/to/raw/data \
    --out /path/to/organized/ --mode copy
```

### Step 1: Structure-from-Motion

Choose one SfM method to produce the COLMAP scene:

```bash
# VGGT with bundle adjustment
python pipeline/sfm/run_vggt_to_colmap.py \
    --input_dir /path/to/scene/ \
    --output_dir /output/vggt_ba/ \
    --use_ba --shared_camera \
    --max_reproj_error 32 --max_query_pts 1048576 --query_frame_num 8

# VGGT (no BA)
python pipeline/sfm/run_vggt_to_colmap.py \
    --input_dir /path/to/scene/ \
    --output_dir /output/vggt_mask/ \

# Classical COLMAP
bash pipeline/sfm/run_colmap.sh \
    --image_path /path/to/scene/images/ \
    --output_path /output/colmap/
```

### Step 2: Surface Reconstruction

Replace `<sfm>` below with the SfM output directory (e.g., `/output/vggt_ba/`).

#### SuGaR

```bash
python pipeline/reconstruction/run_sugar.py <sfm> /output/sugar/ \
    --gs_iterations 20000 --iteration_to_load 7000 \
    --regularization dn_consistency --high_poly --refinement_time long \
    --white_background
```

#### 2DGS

```bash
python pipeline/reconstruction/run_2dgs.py <sfm> /output/2dgs/
```

#### PGSR

```bash
python pipeline/reconstruction/run_pgsr.py <sfm> /output/pgsr/ \
    --max_abs_split_points 0 --opacity_cull_threshold 0.05 \
    --max_depth 10.0 --voxel_size 0.001
```

### Output

| Method | Output files | Location |
|--------|-------------|----------|
| SuGaR | Textured mesh (`.obj`), point cloud (`.ply`) | `<output>/refined_mesh/<scene>/` |
| 2DGS | Triangle mesh (`.ply`) | `<model>/train/ours_<iter>/fuse_post.ply` |
| PGSR | Triangle mesh (`.ply`) | `<model>/mesh/tsdf_fusion_post.ply` |

## Input Requirements

- **Format**: JPG or PNG
- **Resolution**: 1024x768 minimum, higher recommended
- **Overlap**: 60-80% between adjacent views
- **Lighting**: Consistent across all views
- **Focus**: Sharp, minimal motion blur

### Masks

Optional foreground masks can be provided as binary PNG files in a `masks/` directory alongside `images/`. Mask filenames should match image stems (e.g., `image001.png` for `image001.jpg`). White pixels indicate foreground; black pixels indicate background.

## Baselines

### Meshroom

Meshroom (AliceVision) is used as the open-source baseline. See [meshroom-setup.md](meshroom-setup.md) for installation instructions.

Once installed and the environment variables are loaded, run a batch reconstruction with:

```bash
python "$MESHROOM_ROOT/bin/meshroom_batch" \
    -i <path-to-input-images> \
    -o <path-to-output-folder> \
    -p photogrammetry \
    -s <path-to-save-file>
```

## Citations

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

@article{guedon2023sugar,
  title={SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering},
  author={Gu{\'e}don, Antoine and Lepetit, Vincent},
  journal={CVPR},
  year={2024}
}

@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}

@article{chen2024pgsr,
  title={PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction},
  author={Chen, Danpeng and Li, Hai and Ye, Weicai and Wang, Yifan and Xie, Weijian and Zhai, Shangjin and Wang, Nan and Liu, Haomin and Bao, Hujun and Zhang, Guofeng},
  journal={arXiv preprint arXiv:2406.06521},
  year={2024}
}

@inproceedings{schoenberger2016sfm,
    author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
    title={Structure-from-Motion Revisited},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2016},
}

@inproceedings{schoenberger2016mvs,
    author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
    title={Pixelwise View Selection for Unstructured Multi-View Stereo},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2016},
}

@inproceedings{schoenberger2016vote,
    author={Sch\"{o}nberger, Johannes Lutz and Price, True and Sattler, Torsten and Frahm, Jan-Michael and Pollefeys, Marc},
    title={A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval},
    booktitle={Asian Conference on Computer Vision (ACCV)},
    year={2016},
}

@inproceedings{alicevision2021,
  title={{A}liceVision {M}eshroom: An open-source {3D} reconstruction pipeline},
  author={Carsten Griwodz and Simone Gasparini and Lilian Calvet and Pierre Gurdjos and Fabien Castan and Benoit Maujean and Gregoire De Lillo and Yann Lanthony},
  booktitle={Proceedings of the 12th ACM Multimedia Systems Conference - {MMSys '21}},
  doi = {10.1145/3458305.3478443},
  publisher = {ACM Press},
  year = {2021}
}
```

## Acknowledgements

This project was developed as part of a research project at Georgia Institute of Technology under the supervision of Dr. Arthur Porto. Original pipeline contributors: Clinton Kunhardt, James Hennessey, Xin Lin, and Syed Fahad Rizvi, with support provided by Charles Clark, Caleb Wheeler, and Bree Wang.

## Support

For questions or issues, contact: srizvi63@gatech.edu
