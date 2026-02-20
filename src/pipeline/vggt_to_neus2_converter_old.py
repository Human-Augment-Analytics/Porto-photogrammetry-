#!/usr/bin/env python3
"""
VGG-T to NeuS2 Converter

Converts VGG-T predictions to NeuS2-compatible transform.json format.
Author: Assistant
"""

import json
import numpy as np
import os
import shutil
from pathlib import Path
from scipy.spatial.transform import Rotation
import argparse
from typing import Dict, List, Tuple, Any
import pickle

class VGGTToNeuS2Converter:
    """Convert VGG-T predictions to NeuS2 format."""
    
    def __init__(self):
        self.coordinate_transform = self._get_neus2_coordinate_transform()
    
    def _get_neus2_coordinate_transform(self) -> np.ndarray:
        """
        Get the coordinate system transformation matrix for NeuS2.
        NeuS2 uses from_na=true which applies 180° rotation around x-axis.
        """
        # 180 degree rotation around x-axis
        transform = np.eye(4)
        transform[1, 1] = -1  # flip y
        transform[2, 2] = -1  # flip z
        return transform
    
    def fov_to_intrinsic_matrix(self, fov: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Convert field of view values to 4x4 intrinsic matrix.
        
        Args:
            fov: [fov_x, fov_y] in radians or focal length format
            width: Image width
            height: Image height
            
        Returns:
            4x4 intrinsic matrix
        """
        if fov.shape == (2,):
            # Assume fov contains [fx, fy] focal lengths directly
            fx, fy = fov[0], fov[1]
        else:
            # Convert from field of view angles to focal lengths
            fx = width / (2 * np.tan(fov[0] / 2))
            fy = height / (2 * np.tan(fov[1] / 2))
        
        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0
        
        # Create 4x4 intrinsic matrix
        K = np.array([
            [fx,  0, cx, 0],
            [ 0, fy, cy, 0],
            [ 0,  0,  1, 0],
            [ 0,  0,  0, 1]
        ], dtype=np.float64)
        
        return K
    
    def quaternion_translation_to_matrix(self, quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """
        Convert quaternion + translation to 4x4 transformation matrix.
        
        Args:
            quaternion: [qx, qy, qz, qw] quaternion
            translation: [tx, ty, tz] translation vector
            
        Returns:
            4x4 transformation matrix
        """
        # Ensure quaternion is in [x, y, z, w] format for scipy
        if quaternion.shape == (4,):
            q = quaternion
        else:
            raise ValueError(f"Expected quaternion of shape (4,), got {quaternion.shape}")
        
        # Create rotation matrix from quaternion
        rotation = Rotation.from_quat(q)
        R = rotation.as_matrix()
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        
        return T
    
    def compute_scene_normalization(self, world_points: np.ndarray, 
                                  confidence: np.ndarray = None) -> Tuple[float, np.ndarray]:
        """
        Compute scene normalization parameters (scale and offset).
        
        Args:
            world_points: Point cloud coordinates (S, H, W, 3)
            confidence: Confidence scores (S, H, W)
            
        Returns:
            scale: Scene scale factor
            offset: Scene offset [3,]
        """
        # Flatten points
        points = world_points.reshape(-1, 3)
        
        # Filter by confidence if provided
        if confidence is not None:
            conf_flat = confidence.reshape(-1)
            # Use points with confidence > 10th percentile
            conf_threshold = np.percentile(conf_flat, 10)
            valid_mask = conf_flat > conf_threshold
            points = points[valid_mask]
        
        # Remove outliers using percentiles
        if len(points) > 100:
            lower_percentile = np.percentile(points, 5, axis=0)
            upper_percentile = np.percentile(points, 95, axis=0)
            
            # Filter points within percentile bounds
            valid_mask = np.all(
                (points >= lower_percentile) & (points <= upper_percentile), 
                axis=1
            )
            points = points[valid_mask]
        
        if len(points) == 0:
            return 0.5, np.array([0.5, 0.5, 0.5])
        
        # Compute bounding box
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        center = (min_bounds + max_bounds) / 2
        
        # Scale is the diagonal length of the bounding box
        diagonal = np.linalg.norm(max_bounds - min_bounds)
        scale = diagonal / 2.0  # Normalize to unit scale
        
        # Offset centers the scene
        offset = 0.5 - center / scale
        
        return scale, offset
    
    def convert_vggt_predictions(self, predictions: Dict[str, Any], 
                                output_dir: str,
                                images_dir: str = None) -> str:
        """
        Convert VGG-T predictions to NeuS2 format.
        
        Args:
            predictions: VGG-T predictions dictionary containing:
                - world_points: (S, H, W, 3) world coordinates
                - world_points_conf: (S, H, W) confidence scores
                - images: (S, H, W, 3) input images
                - extrinsic: (S, 3, 4) or (S, 4, 4) camera extrinsics
                - camera_params: (S, 9) [qx, qy, qz, qw, tx, ty, tz, fx, fy] (if available)
            output_dir: Output directory for NeuS2 data
            images_dir: Optional source directory containing images
            
        Returns:
            Path to generated transform.json file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract data from predictions
        world_points = predictions.get('world_points', None)
        world_points_conf = predictions.get('world_points_conf', None)
        images = predictions['images']
        extrinsics = predictions['extrinsic']
        
        # Get image dimensions
        if images.ndim == 4:  # (S, H, W, 3)
            S, H, W, _ = images.shape
        else:
            raise ValueError(f"Expected images shape (S, H, W, 3), got {images.shape}")
        
        # Compute scene normalization
        if world_points is not None:
            scale, offset = self.compute_scene_normalization(world_points, world_points_conf)
        else:
            scale, offset = 0.5, np.array([0.5, 0.5, 0.5])
        
        # Create images directory
        images_output_dir = output_path / "images"
        images_output_dir.mkdir(exist_ok=True)
        
        # Process each frame
        frames = []
        for i in range(S):
            # Handle extrinsics - convert (3,4) to (4,4) if needed
            if extrinsics.shape[-2:] == (3, 4):
                extrinsic_44 = np.eye(4)
                extrinsic_44[:3, :4] = extrinsics[i]
            elif extrinsics.shape[-2:] == (4, 4):
                extrinsic_44 = extrinsics[i]
            else:
                raise ValueError(f"Unexpected extrinsic shape: {extrinsics.shape}")
            
            # Apply NeuS2 coordinate transformation
            transform_matrix = extrinsic_44 @ self.coordinate_transform
            
            # Handle intrinsics
            if 'camera_params' in predictions:
                # Extract from camera_params: [qx, qy, qz, qw, tx, ty, tz, fx, fy]
                camera_params = predictions['camera_params'][i]
                fov = camera_params[7:9]  # [fx, fy]
            elif 'intrinsic' in predictions:
                # Direct intrinsic matrices
                if predictions['intrinsic'].shape[-2:] == (4, 4):
                    intrinsic_matrix = predictions['intrinsic'][i]
                elif predictions['intrinsic'].shape[-2:] == (3, 3):
                    # Convert 3x3 to 4x4
                    intrinsic_matrix = np.eye(4)
                    intrinsic_matrix[:3, :3] = predictions['intrinsic'][i]
                else:
                    raise ValueError(f"Unexpected intrinsic shape: {predictions['intrinsic'].shape}")
            else:
                # Estimate intrinsics from image size (fallback)
                fx = fy = max(W, H)  # Reasonable default
                intrinsic_matrix = np.array([
                    [fx,  0, W/2, 0],
                    [ 0, fy, H/2, 0],
                    [ 0,  0,   1, 0],
                    [ 0,  0,   0, 1]
                ])
            
            # If we computed intrinsic_matrix from fov
            if 'camera_params' in predictions and 'intrinsic' not in predictions:
                intrinsic_matrix = self.fov_to_intrinsic_matrix(fov, W, H)
            
            # Save image
            image_filename = f"{i:06d}.png"
            image_path = images_output_dir / image_filename
            
            # Convert and save image
            if not image_path.exists():
                if images_dir and Path(images_dir).exists():
                    # Copy from source directory
                    source_files = list(Path(images_dir).glob(f"*{i:06d}*")) + \
                                  list(Path(images_dir).glob(f"*{i:03d}*")) + \
                                  list(Path(images_dir).glob(f"*{i}.*"))
                    if source_files:
                        shutil.copy2(source_files[0], image_path)
                    else:
                        self._save_image_array(images[i], image_path)
                else:
                    # Save from array
                    self._save_image_array(images[i], image_path)
            
            # Create frame entry
            frame = {
                "file_path": f"images/{image_filename}",
                "transform_matrix": transform_matrix.tolist(),
                "intrinsic_matrix": intrinsic_matrix.tolist()
            }
            frames.append(frame)
        
        # Create transform.json
        transform_data = {
            "w": W,
            "h": H,
            "aabb_scale": 1.0,
            "scale": float(scale),
            "offset": offset.tolist(),
            "from_na": True,
            "frames": frames
        }
        
        # Save transform.json
        transform_json_path = output_path / "transform.json"
        with open(transform_json_path, 'w') as f:
            json.dump(transform_data, f, indent=4)
        
        print(f"Conversion complete!")
        print(f"Output directory: {output_path}")
        print(f"Transform file: {transform_json_path}")
        print(f"Number of frames: {len(frames)}")
        print(f"Image dimensions: {W}x{H}")
        print(f"Scene scale: {scale:.4f}")
        print(f"Scene offset: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")
        
        return str(transform_json_path)
    
    def _save_image_array(self, image_array: np.ndarray, output_path: Path):
        """Save image array to file."""
        try:
            from PIL import Image
            
            # Normalize to 0-255 range
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            # Convert to PIL and save
            pil_image = Image.fromarray(image_array)
            pil_image.save(output_path)
            
        except ImportError:
            # Fallback to OpenCV
            import cv2
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), image_bgr)


def load_vggt_predictions(predictions_path: str) -> Dict[str, Any]:
    """Load VGG-T predictions from file."""
    path = Path(predictions_path)
    
    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.suffix == '.npz':
        data = np.load(path, allow_pickle=True)
        return dict(data)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Convert VGG-T predictions to NeuS2 format")
    parser.add_argument("predictions_path", help="Path to VGG-T predictions file (.pkl or .npz)")
    parser.add_argument("output_dir", help="Output directory for NeuS2 data")
    parser.add_argument("--images_dir", help="Source directory containing original images")
    parser.add_argument("--config", default="dtu.json", help="NeuS2 config name for training")
    parser.add_argument("--n_steps", type=int, default=15000, help="Training steps")
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading VGG-T predictions from: {args.predictions_path}")
    predictions = load_vggt_predictions(args.predictions_path)
    
    # Convert to NeuS2 format
    converter = VGGTToNeuS2Converter()
    transform_json_path = converter.convert_vggt_predictions(
        predictions, 
        args.output_dir,
        args.images_dir
    )
    
    # Print NeuS2 training command
    print("\n" + "="*60)
    print("🎉 Conversion Complete! Ready for NeuS2 training.")
    print("="*60)
    print(f"\nTo train with NeuS2, run:")
    print(f"python scripts/run.py \\")
    print(f"    --scene {transform_json_path} \\")
    print(f"    --name your_experiment_name \\")
    print(f"    --network {args.config} \\")
    print(f"    --n_steps {args.n_steps}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()