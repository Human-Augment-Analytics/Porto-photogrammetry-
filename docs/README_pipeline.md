# VGG-T to NeuS2 Converter

A comprehensive pipeline to convert VGG-T (Visual Geometry Grounded Transformer) predictions into NeuS2-compatible format for high-quality neural surface reconstruction.

## Overview

This converter bridges the gap between VGG-T's rich 3D understanding and NeuS2's powerful neural surface reconstruction capabilities, enabling you to leverage VGG-T's point clouds and camera estimates for detailed 3D reconstruction.

## Features

- ✅ **Complete Format Conversion**: Transforms VGG-T predictions to NeuS2's `transform.json` format
- ✅ **Coordinate System Handling**: Properly handles coordinate transformations between VGG-T and NeuS2
- ✅ **Camera Parameter Conversion**: Converts quaternion + translation + FOV to transformation matrices
- ✅ **Scene Normalization**: Automatically computes optimal scale and offset parameters
- ✅ **Validation Tools**: Comprehensive validation and debugging utilities
- ✅ **Visualization**: Camera pose visualization for verification
- ✅ **Flexible Input Support**: Handles various VGG-T output formats

## Installation

```bash
# Required dependencies
pip install numpy scipy matplotlib opencv-python pillow

# Optional for visualization
pip install matplotlib
```

## Quick Start

### 1. Basic Conversion

```python
from vggt_to_neus2_converter import VGGTToNeuS2Converter, load_vggt_predictions

# Load VGG-T predictions
predictions = load_vggt_predictions("path/to/vggt_predictions.pkl")

# Convert to NeuS2 format
converter = VGGTToNeuS2Converter()
transform_json_path = converter.convert_vggt_predictions(
    predictions=predictions,
    output_dir="./neus2_data",
    images_dir="./original_images"  # Optional
)

print(f"Ready for NeuS2! Transform file: {transform_json_path}")
```

### 2. Command Line Usage

```bash
# Convert predictions
python vggt_to_neus2_converter.py \
    path/to/predictions.pkl \
    ./neus2_output \
    --images_dir ./original_images

# Validate conversion
python conversion_utils.py validate ./neus2_output/transform.json

# Visualize camera poses
python conversion_utils.py visualize ./neus2_output/transform.json --output poses.png
```

### 3. Train with NeuS2

```bash
# Use the generated transform.json for NeuS2 training
python scripts/run.py \
    --scene ./neus2_output/transform.json \
    --name my_reconstruction \
    --network dtu.json \
    --n_steps 15000
```

## Input Format

### VGG-T Predictions Dictionary

The converter expects a dictionary with the following structure:

```python
predictions = {
    'world_points': np.ndarray,      # (S, H, W, 3) - 3D world coordinates
    'world_points_conf': np.ndarray, # (S, H, W) - confidence scores
    'images': np.ndarray,            # (S, H, W, 3) - input images
    'extrinsic': np.ndarray,         # (S, 3, 4) or (S, 4, 4) - camera extrinsics
    
    # Optional - if available:
    'camera_params': np.ndarray,     # (S, 9) - [qx,qy,qz,qw,tx,ty,tz,fx,fy]
    'intrinsic': np.ndarray,         # (S, 3, 3) or (S, 4, 4) - intrinsic matrices
}
```

Where:
- `S` = number of frames/cameras
- `H` = image height
- `W` = image width

## Output Format

### NeuS2 Transform.json Structure

```json
{
    "w": 640,
    "h": 480,
    "aabb_scale": 1.0,
    "scale": 0.5,
    "offset": [0.5, 0.5, 0.5],
    "from_na": true,
    "frames": [
        {
            "file_path": "images/000000.png",
            "transform_matrix": [
                [/* 4x4 camera-to-world matrix */]
            ],
            "intrinsic_matrix": [
                [/* 4x4 intrinsic matrix */]
            ]
        },
        // ... more frames
    ]
}
```

## Key Technical Details

### Coordinate System Transformation

VGG-T uses the first camera as the world reference frame, while NeuS2 expects a specific coordinate system with `from_na=true` (180° rotation around x-axis). The converter handles this transformation automatically.

### Camera Parameter Conversion

The converter supports multiple input formats:

1. **Quaternion + Translation + FOV**: `[qx, qy, qz, qw, tx, ty, tz, fx, fy]`
2. **Extrinsic Matrices**: Direct 3×4 or 4×4 transformation matrices
3. **Separate Intrinsics**: 3×3 or 4×4 intrinsic matrices

### Scene Normalization

The converter automatically computes optimal `scale` and `offset` parameters by:
1. Analyzing the point cloud distribution
2. Filtering outliers using percentile bounds
3. Computing scene bounding box
4. Normalizing to unit scale

## Validation and Debugging

### Validation Tools

```python
from conversion_utils import ConversionValidator

validator = ConversionValidator()

# Validate transform.json
results = validator.validate_transform_json("./neus2_data/transform.json")
print(f"Valid: {results['valid']}")

# Check if images exist
image_check = validator.check_images_exist("./neus2_data/transform.json")
print(f"All images exist: {image_check['all_exist']}")

# Visualize camera poses
validator.visualize_camera_poses("./neus2_data/transform.json", "poses.png")
```

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| "Missing intrinsic matrices" | Ensure your VGG-T predictions include `camera_params` or `intrinsic` |
| "Invalid coordinate system" | Check that `from_na=true` in output transform.json |
| "Images not found" | Provide correct `images_dir` path or ensure images are in predictions |
| "Scale/offset issues" | Verify your point cloud has reasonable bounds |

## Advanced Configuration

### Custom Coordinate Transformations

```python
# Custom coordinate transformation
converter = VGGTToNeuS2Converter()
converter.coordinate_transform = custom_transform_matrix

# Convert with custom settings
transform_json_path = converter.convert_vggt_predictions(
    predictions=predictions,
    output_dir="./output"
)
```

### Scene Normalization Tuning

```python
# Compute custom normalization
scale, offset = converter.compute_scene_normalization(
    world_points=predictions['world_points'],
    confidence=predictions['world_points_conf']
)
print(f"Computed scale: {scale}, offset: {offset}")
```

## Integration with Existing Pipelines

### From VGG-T Demo/Gradio App

If you're using the VGG-T Gradio demo, you can extract predictions:

```python
# In your VGG-T inference code
predictions = model_inference(images)  # Your VGG-T inference
converter = VGGTToNeuS2Converter()
transform_json_path = converter.convert_vggt_predictions(predictions, "./neus2_data")
```

### Batch Processing

```python
# Process multiple scenes
scenes = ["scene1.pkl", "scene2.pkl", "scene3.pkl"]
converter = VGGTToNeuS2Converter()

for scene_file in scenes:
    predictions = load_vggt_predictions(scene_file)
    output_dir = f"./neus2_data_{Path(scene_file).stem}"
    converter.convert_vggt_predictions(predictions, output_dir)
```

## Performance Optimization

### For Large Datasets

```python
# For large point clouds, use confidence filtering
converter = VGGTToNeuS2Converter()

# Filter low-confidence points before conversion
filtered_predictions = {
    'world_points': predictions['world_points'],
    'world_points_conf': predictions['world_points_conf'],
    'images': predictions['images'],
    'extrinsic': predictions['extrinsic']
}

# Convert with confidence threshold
transform_json_path = converter.convert_vggt_predictions(
    filtered_predictions, 
    output_dir="./neus2_data"
)
```

## Troubleshooting

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Inspect predictions structure
from conversion_utils import inspect_vggt_predictions
inspect_vggt_predictions("predictions.pkl")
```

### Validation Checklist

Before running NeuS2 training, ensure:

- [ ] `transform.json` validates without errors
- [ ] All referenced images exist
- [ ] Camera poses look reasonable in visualization
- [ ] Scene scale and offset are appropriate
- [ ] Intrinsic matrices have reasonable focal lengths

## Examples

### Complete Example

See `example_usage.py` for a comprehensive example that demonstrates:
- Loading VGG-T predictions
- Converting to NeuS2 format
- Validating the output
- Visualizing camera poses
- Generating the NeuS2 training command

### Minimal Example

```python
from vggt_to_neus2_converter import VGGTToNeuS2Converter
import pickle

# Load predictions
with open("vggt_predictions.pkl", 'rb') as f:
    predictions = pickle.load(f)

# Convert
converter = VGGTToNeuS2Converter()
transform_json_path = converter.convert_vggt_predictions(
    predictions, "./neus2_data"
)

print(f"Ready for NeuS2: {transform_json_path}")
```

## Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License.

## Citation

If you use this converter in your research, please cite both VGG-T and NeuS2:

```bibtex
@article{vggt2025,
  title={Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  journal={arXiv preprint},
  year={2025}
}

@inproceedings{neus2,
  title={NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction},
  author={Wang, Yiming and Han, Qin and Habermann, Marc and Daniilidis, Kostas and Theobalt, Christian and Liu, Lingjie},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
