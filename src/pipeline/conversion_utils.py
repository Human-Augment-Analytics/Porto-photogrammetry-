#!/usr/bin/env python3
"""
Conversion Utilities

Helper functions for VGG-T to NeuS2 conversion debugging and validation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2

class ConversionValidator:
    """Validate and debug VGG-T to NeuS2 conversion."""
    
    def validate_transform_json(self, transform_json_path: str) -> Dict[str, Any]:
        """
        Validate transform.json file structure and content.
        
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            with open(transform_json_path, 'r') as f:
                data = json.load(f)
                
            # Check required fields
            required_fields = ['w', 'h', 'aabb_scale', 'scale', 'offset', 'from_na', 'frames']
            for field in required_fields:
                if field not in data:
                    results["errors"].append(f"Missing required field: {field}")
                    results["valid"] = False
            
            if not results["valid"]:
                return results
            
            # Validate global parameters
            if data['from_na'] != True:
                results["warnings"].append("from_na should be True for NeuS2")
            
            if data['aabb_scale'] != 1.0:
                results["warnings"].append("aabb_scale should typically be 1.0")
            
            # Check frames
            frames = data['frames']
            if len(frames) == 0:
                results["errors"].append("No frames found")
                results["valid"] = False
                return results
            
            # Validate each frame
            for i, frame in enumerate(frames):
                frame_errors = self._validate_frame(frame, i)
                results["errors"].extend(frame_errors)
                if frame_errors:
                    results["valid"] = False
            
            # Compute statistics
            results["stats"] = {
                "num_frames": len(frames),
                "image_size": [data['w'], data['h']],
                "scale": data['scale'],
                "offset": data['offset'],
                "avg_focal_length": self._compute_avg_focal_length(frames),
                "camera_positions": self._extract_camera_positions(frames)
            }
            
        except Exception as e:
            results["errors"].append(f"Failed to load transform.json: {str(e)}")
            results["valid"] = False
        
        return results
    
    def _validate_frame(self, frame: Dict, frame_idx: int) -> List[str]:
        """Validate individual frame structure."""
        errors = []
        
        # Check required fields
        required_fields = ['file_path', 'transform_matrix', 'intrinsic_matrix']
        for field in required_fields:
            if field not in frame:
                errors.append(f"Frame {frame_idx}: Missing field {field}")
                continue
        
        if errors:
            return errors
        
        # Validate transform matrix
        transform = np.array(frame['transform_matrix'])
        if transform.shape != (4, 4):
            errors.append(f"Frame {frame_idx}: transform_matrix must be 4x4, got {transform.shape}")
        else:
            # Check if bottom row is [0, 0, 0, 1]
            expected_bottom = np.array([0, 0, 0, 1])
            if not np.allclose(transform[3], expected_bottom, atol=1e-6):
                errors.append(f"Frame {frame_idx}: transform_matrix bottom row should be [0,0,0,1]")
        
        # Validate intrinsic matrix
        intrinsic = np.array(frame['intrinsic_matrix'])
        if intrinsic.shape != (4, 4):
            errors.append(f"Frame {frame_idx}: intrinsic_matrix must be 4x4, got {intrinsic.shape}")
        else:
            # Check basic intrinsic structure
            if intrinsic[2, 2] != 1.0:
                errors.append(f"Frame {frame_idx}: intrinsic_matrix[2,2] should be 1.0")
        
        # Check file path
        if not frame['file_path'].startswith('images/'):
            errors.append(f"Frame {frame_idx}: file_path should start with 'images/'")
        
        return errors
    
    def _compute_avg_focal_length(self, frames: List[Dict]) -> float:
        """Compute average focal length across frames."""
        focal_lengths = []
        for frame in frames:
            intrinsic = np.array(frame['intrinsic_matrix'])
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            focal_lengths.append((fx + fy) / 2)
        return np.mean(focal_lengths)
    
    def _extract_camera_positions(self, frames: List[Dict]) -> np.ndarray:
        """Extract camera positions from transform matrices."""
        positions = []
        for frame in frames:
            transform = np.array(frame['transform_matrix'])
            position = transform[:3, 3]
            positions.append(position)
        return np.array(positions)
    
    def visualize_camera_poses(self, transform_json_path: str, output_path: str = None):
        """
        Visualize camera poses from transform.json.
        """
        with open(transform_json_path, 'r') as f:
            data = json.load(f)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = []
        orientations = []
        
        for i, frame in enumerate(data['frames']):
            transform = np.array(frame['transform_matrix'])
            position = transform[:3, 3]
            rotation = transform[:3, :3]
            
            positions.append(position)
            
            # Extract camera orientation (forward direction)
            forward = -rotation[:, 2]  # -Z axis in camera coordinates
            orientations.append(forward)
        
        positions = np.array(positions)
        orientations = np.array(orientations)
        
        # Plot camera positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', s=50, alpha=0.7, label='Camera Positions')
        
        # Plot camera orientations
        for i in range(len(positions)):
            ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                     orientations[i, 0], orientations[i, 1], orientations[i, 2],
                     length=0.3, color='blue', alpha=0.6)
        
        # Add frame numbers
        for i, pos in enumerate(positions[::5]):  # Every 5th frame
            ax.text(pos[0], pos[1], pos[2], f'{i*5}', fontsize=8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Poses')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([positions.max()-positions.min()]).max() / 2.0
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Camera pose visualization saved to: {output_path}")
        else:
            plt.show()
    
    def check_images_exist(self, transform_json_path: str) -> Dict[str, Any]:
        """Check if all referenced images exist."""
        results = {
            "all_exist": True,
            "missing_files": [],
            "existing_files": [],
            "total_frames": 0
        }
        
        transform_dir = Path(transform_json_path).parent
        
        with open(transform_json_path, 'r') as f:
            data = json.load(f)
        
        results["total_frames"] = len(data['frames'])
        
        for frame in data['frames']:
            file_path = transform_dir / frame['file_path']
            
            if file_path.exists():
                results["existing_files"].append(str(file_path))
            else:
                results["missing_files"].append(str(file_path))
                results["all_exist"] = False
        
        return results
    
    def compare_with_reference(self, transform_json_path: str, reference_json_path: str) -> Dict[str, Any]:
        """Compare generated transform.json with reference."""
        with open(transform_json_path, 'r') as f:
            generated = json.load(f)
        
        with open(reference_json_path, 'r') as f:
            reference = json.load(f)
        
        comparison = {
            "structure_match": True,
            "differences": [],
            "stats_comparison": {}
        }
        
        # Compare global parameters
        global_params = ['w', 'h', 'aabb_scale', 'scale', 'offset', 'from_na']
        for param in global_params:
            if param in generated and param in reference:
                if generated[param] != reference[param]:
                    comparison["differences"].append(f"Global param {param}: {generated[param]} vs {reference[param]}")
            else:
                comparison["differences"].append(f"Missing global param: {param}")
                comparison["structure_match"] = False
        
        # Compare number of frames
        gen_frames = len(generated.get('frames', []))
        ref_frames = len(reference.get('frames', []))
        
        comparison["stats_comparison"] = {
            "generated_frames": gen_frames,
            "reference_frames": ref_frames,
            "frame_count_match": gen_frames == ref_frames
        }
        
        if gen_frames != ref_frames:
            comparison["differences"].append(f"Frame count mismatch: {gen_frames} vs {ref_frames}")
        
        return comparison


def inspect_vggt_predictions(predictions_path: str):
    """Inspect VGG-T predictions structure."""
    if predictions_path.endswith('.pkl'):
        import pickle
        with open(predictions_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = np.load(predictions_path, allow_pickle=True)
        if hasattr(data, 'files'):
            print("NPZ file contents:")
            for key in data.files:
                print(f"  {key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")
            return
    
    print("VGG-T Predictions Structure:")
    print("=" * 40)
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape} ({value.dtype})")
            if key in ['world_points', 'images']:
                print(f"  Range: [{value.min():.3f}, {value.max():.3f}]")
        elif isinstance(value, (list, tuple)):
            print(f"{key}: {type(value)} of length {len(value)}")
        else:
            print(f"{key}: {type(value)}")
    
    print("\nDetailed Analysis:")
    print("-" * 20)
    
    # Analyze specific keys
    if 'extrinsic' in data:
        ext = data['extrinsic']
        print(f"Extrinsics shape: {ext.shape}")
        print(f"Sample extrinsic matrix (frame 0):")
        print(ext[0])
    
    if 'world_points' in data:
        wp = data['world_points']
        print(f"World points shape: {wp.shape}")
        print(f"Point cloud bounds:")
        points_flat = wp.reshape(-1, 3)
        print(f"  X: [{points_flat[:, 0].min():.3f}, {points_flat[:, 0].max():.3f}]")
        print(f"  Y: [{points_flat[:, 1].min():.3f}, {points_flat[:, 1].max():.3f}]")
        print(f"  Z: [{points_flat[:, 2].min():.3f}, {points_flat[:, 2].max():.3f}]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VGG-T to NeuS2 conversion utilities")
    parser.add_argument("command", choices=["validate", "visualize", "inspect", "check_images"])
    parser.add_argument("input_path", help="Path to transform.json or predictions file")
    parser.add_argument("--output", help="Output path for visualization")
    parser.add_argument("--reference", help="Reference transform.json for comparison")
    
    args = parser.parse_args()
    
    validator = ConversionValidator()
    
    if args.command == "validate":
        results = validator.validate_transform_json(args.input_path)
        print("Validation Results:")
        print(f"Valid: {results['valid']}")
        if results['errors']:
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")
        if results['warnings']:
            print("Warnings:")  
            for warning in results['warnings']:
                print(f"  - {warning}")
        if results['stats']:
            print("Statistics:")
            for key, value in results['stats'].items():
                print(f"  {key}: {value}")
    
    elif args.command == "visualize":
        validator.visualize_camera_poses(args.input_path, args.output)
    
    elif args.command == "inspect":
        inspect_vggt_predictions(args.input_path)
    
    elif args.command == "check_images":
        results = validator.check_images_exist(args.input_path)
        print(f"Images check: {results['all_exist']}")
        print(f"Total frames: {results['total_frames']}")
        print(f"Existing: {len(results['existing_files'])}")
        print(f"Missing: {len(results['missing_files'])}")
        if results['missing_files']:
            print("Missing files:")
            for f in results['missing_files'][:10]:  # Show first 10
                print(f"  - {f}")