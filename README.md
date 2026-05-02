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
| Gaussian Wrapping | Stochastic oriented Gaussians with pivot-based marching-tetrahedra extraction and texture refinement |

All combinations are benchmarked on runtime and compared qualitatively against RealityScan (commercial) and Meshroom (open-source) baselines.

## Installation

### Requirements

- Linux (tested on RHEL 9)
- Python 3.10
- CUDA-capable GPU (80 GB+ VRAM recommended)
- CUDA 12.8

**Note**: VGGT requires a GPU with at least 80 GB of VRAM for large scenes. COLMAP and the reconstruction methods, however, can run on more modest hardware (32-40 GB), with tests conducted on a single NVIDIA A100 PCIe 40 GB.

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
python -m pip install torch==2.9.1 torchvision==0.24.1 \
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
    src/gaussian_wrapping/submodules/diff-gaussian-rasterization-gw \
    src/gaussian_wrapping/submodules/diff-gaussian-rasterization-ms \
    src/gaussian_wrapping/submodules/fused-ssim \
    src/gaussian_wrapping/submodules/warp-patch-ncc \
    --no-build-isolation

# Install VGGT as editable package
python -m pip install -e src/vggt --no-build-isolation

# Build and install Tetra-NeRF triangulation module
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
conda install -y cmake
conda install -y conda-forge::gmp
conda install -y conda-forge::cgal
cd src/gaussian_wrapping/submodules/tetra_triangulation
cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCGAL_DIR=$CONDA_PREFIX/lib/cmake/CGAL \
        -DTorch_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share/cmake/Torch'))") \
        -DCMAKE_IGNORE_PATH="$ALICEVISION_ROOT;$ALICEVISION_ROOT/bin;$ALICEVISION_ROOT/lib"
make
pip install -e .
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
│       ├── run_pgsr.py
│       └── run_gw.py
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
│   ├── gaussian_wrapping/     # Gaussian Wrapping (Blobs to Spokes)
│   │   ├── gaussian_renderer/ #   ours/radegs/sof rasterizers
│   │   ├── extraction/        #   Pivot sampling + mesh extraction
│   │   ├── regularization/    #   Normal-field, multiview, MILo, SDF
│   │   ├── scene/             #   Scene + GaussianModel + Mesh
│   │   ├── scripts/           #   End-to-end driver scripts
│   │   └── submodules/        #   CUDA rasterizers + tetra triangulation
│   ├── light_glue/            # LightGlue (submodule)
│   └── pytorch3d/             # PyTorch3D (submodule)
└── CLAUDE.md                  # Detailed codebase documentation
```

## Input Requirements

- **Format**: JPG or PNG
- **Resolution**: 1024x768 minimum, higher recommended
- **Overlap**: 60-80% between adjacent views
- **Lighting**: Consistent across all views
- **Focus**: Sharp, minimal motion blur

### Masks

Optional foreground masks can be provided as binary PNG files in a `masks/` directory alongside `images/`. Mask filenames should match image stems (e.g., `image001.png` for `image001.jpg`). White pixels indicate foreground; black pixels indicate background.

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
    --output_dir /output/vggt/ \
    --conf_thres_value 1.0

# Classical COLMAP
bash pipeline/sfm/run_colmap.sh \
    --input_dir /path/to/scene/ \
    --output_dir /output/colmap/
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

#### Gaussian Wrapping

```bash
python pipeline/reconstruction/run_gw.py <sfm> /output/gw/ \
    --iterations 30000 --sh_degree 3 --max_gaussians 6000000 \
    --n_pivots 2 --std_factor 3.0 --n_binary_steps 10 --isosurface_value 0.0
```

The wrapper runs three sequential stages: training (`train.py --rasterizer ours`),
pivot-based marching-tetrahedra mesh extraction (`pivot_based_mesh_extraction.py`),
and texture refinement (`texture_mesh.py`). Pass `-r 2` to downsample images for
metrics-comparable runs; pass `--isosurface_value 0.2` if the mesh is missing fine
details.

### Output

| Method | Output files | Location |
|--------|-------------|----------|
| SuGaR | Textured mesh (`.obj`), point cloud (`.ply`) | `<output>/refined_mesh/<scene>/` |
| 2DGS | Triangle mesh (`.ply`) | `<model>/train/ours_<iter>/fuse_post.ply` |
| PGSR | Triangle mesh (`.ply`) | `<model>/mesh/tsdf_fusion_post.ply` |
| Gaussian Wrapping | Extracted mesh, post-processed mesh, textured mesh (`.ply`) | `<output>/mesh_ours_2pivots{,_post,_post_texture_refined_<iter>}.ply` |

## Baselines

### Meshroom

Meshroom (AliceVision) is used as the open-source baseline. See [meshroom-setup.md](meshroom-setup.md) for installation instructions.

Once installed and `MESHROOM_ROOT` is set, run the wrapper script to execute a batch reconstruction and log runtime:

```bash
python baseline/benchmark_meshroom.py /path/to/scene/ /output/meshroom/ \
    --save_file /output/meshroom/graph.mg
```

The wrapper resolves `meshroom_batch` from `$MESHROOM_ROOT` (or `--meshroom_root`), runs the `photogrammetry` pipeline by default, and prints total runtime on completion. To invoke `meshroom_batch` directly instead:

```bash
python "$MESHROOM_ROOT/bin/meshroom_batch" \
    -i <path-to-input-images> \
    -o <path-to-output-folder> \
    -p photogrammetry \
    -s <path-to-save-file>
    --paramOverrides FeatureExtraction:masksFolder=<path-to-masks>
```

## Runtime Comparison
### End-to-end
The following table compares the total runtime of each method on a sample scene with 138 high-resolution images (6240x4160) and masks, run on an NVIDIA A100 PCIe 40 GB GPU.

| Method                        | Total Runtime     |
|-------------------------------|-------------------|
| COLMAP + SuGaR                | ~80 mins          |
| COLMAP + 2DGS                 | ~40 mins          |
| COLMAP + PGSR                 | ~70 mins          |
| COLMAP + Gaussian Wrapping    | ~65 mins          |
| Meshroom                      | ~60 mins          |

### Structure-from-Motion
The following table compares the runtime of the for each SfM method on the same scene as above,
run on an NVIDIA B200 80 GB GPU.
| Method            | Runtime       |
|-------------------|---------------|
| COLMAP            | ~10 mins      |
| VGGT              | ~3 mins       |
| VGGT + BA         | ~15 mins      |

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

@misc{gomez2026blobsspokeshighfidelitysurface,
      title={From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians}, 
      author={Diego Gomez and Antoine Guédon and Nissim Maruani and Bingchen Gong and Maks Ovsjanikov},
      year={2026},
      eprint={2604.07337},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.07337}, 
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

This project was developed as part of a research project at Georgia Institute of Technology under the supervision of Dr. Arthur Porto. Original pipeline contributors: Clinton Kunhardt, James Hennessey, Xin Lin, and Syed Fahad Rizvi, with support provided by Charles Clark, Caleb Wheeler, Bree Wang, and Riyam Zaman.

## Support

For questions or issues, contact: srizvi63@gatech.edu
