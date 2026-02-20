#!/usr/bin/env python3
"""
Example Usage: VGG-T to NeuS2 Conversion

This script demonstrates how to use the VGG-T to NeuS2 converter.
Run this after you have VGG-T predictions saved.
"""

import numpy as np
import sys
from pathlib import Path

# Add the converter to path
sys.path.append('.')

from vggt_to_neus2_converter import VGGTToNeuS2Converter, load_vggt_predictions
from conversion_utils import ConversionValidator, inspect_vggt_predictions

def main():
    print("🚀 VGG-T to NeuS2 Conversion Example")
    print("=" * 50)
    
    # Configuration
    predictions_file = "path/to/your/vggt_predictions.pkl"  # Update this path
    output_dir = "./neus2_data"
    images_dir = "./images"  # Optional: source images directory
    
    # Step 1: Inspect VGG-T predictions (optional)
    print("\n📋 Step 1: Inspecting VGG-T predictions...")
    if Path(predictions_file).exists():
        try:
            inspect_vggt_predictions(predictions_file)
        except Exception as e:
            print(f"Could not inspect predictions: {e}")
            print("This is optional - continuing with conversion...")
    else:
        print(f"⚠️  Predictions file not found: {predictions_file}")
        print("Please update the predictions_file path in this script.")
        
        # Create a dummy example for demonstration
        print("\n🔧 Creating dummy data for demonstration...")
        predictions = create_dummy_vggt_predictions()
        predictions_file = "./dummy_predictions.pkl"
        
        import pickle
        with open(predictions_file, 'wb') as f:
            pickle.dump(predictions, f)
        print(f"Saved dummy predictions to: {predictions_file}")
    
    # Step 2: Load predictions
    print(f"\n📂 Step 2: Loading predictions from {predictions_file}...")
    try:
        predictions = load_vggt_predictions(predictions_file)
        print("✅ Predictions loaded successfully!")
        
        # Print basic info
        for key, value in predictions.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
                
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return
    
    # Step 3: Convert to NeuS2 format
    print(f"\n🔄 Step 3: Converting to NeuS2 format...")
    converter = VGGTToNeuS2Converter()
    
    try:
        transform_json_path = converter.convert_vggt_predictions(
            predictions=predictions,
            output_dir=output_dir,
            images_dir=images_dir if Path(images_dir).exists() else None
        )
        print("✅ Conversion completed successfully!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return
    
    # Step 4: Validate the output
    print(f"\n✅ Step 4: Validating conversion...")
    validator = ConversionValidator()
    
    # Validate transform.json
    results = validator.validate_transform_json(transform_json_path)
    print(f"Validation status: {'✅ PASS' if results['valid'] else '❌ FAIL'}")
    
    if results['errors']:
        print("Errors found:")
        for error in results['errors']:
            print(f"  ❌ {error}")
    
    if results['warnings']:
        print("Warnings:")
        for warning in results['warnings']:
            print(f"  ⚠️  {warning}")
    
    if results['stats']:
        print("Conversion Statistics:")
        for key, value in results['stats'].items():
            print(f"  📊 {key}: {value}")
    
    # Step 5: Check images
    print(f"\n🖼️  Step 5: Checking images...")
    image_check = validator.check_images_exist(transform_json_path)
    print(f"Images status: {'✅ All found' if image_check['all_exist'] else '⚠️  Some missing'}")
    print(f"Total frames: {image_check['total_frames']}")
    print(f"Existing: {len(image_check['existing_files'])}")
    print(f"Missing: {len(image_check['missing_files'])}")
    
    # Step 6: Generate visualization (optional)
    print(f"\n📈 Step 6: Generating camera pose visualization...")
    try:
        viz_path = Path(output_dir) / "camera_poses.png"
        validator.visualize_camera_poses(transform_json_path, str(viz_path))
        print(f"✅ Visualization saved: {viz_path}")
    except Exception as e:
        print(f"⚠️  Could not generate visualization: {e}")
    
    # Step 7: Print NeuS2 training command
    print(f"\n🎯 Step 7: Ready for NeuS2 training!")
    print("=" * 50)
    print("🔥 Run this command to start NeuS2 training:")
    print()
    print(f"python scripts/run.py \\")
    print(f"    --scene {transform_json_path} \\")
    print(f"    --name my_reconstruction \\")
    print(f"    --network dtu.json \\")
    print(f"    --n_steps 15000")
    print()
    print("For other datasets, try these configs:")
    print("  • DTU dataset: --network dtu.json --n_steps 15000")
    print("  • General scenes: --network base.json --n_steps 20000")
    print()
    print("=" * 50)
    print("🎉 Conversion pipeline completed successfully!")


def create_dummy_vggt_predictions():
    """Create dummy VGG-T predictions for demonstration."""
    print("Creating synthetic VGG-T predictions...")
    
    # Simulate a simple scene with 8 cameras around an object
    S = 8  # Number of frames/cameras
    H, W = 480, 640  # Image dimensions
    
    # Create synthetic world points (a simple cube)
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, W//4),
        np.linspace(-1, 1, H//4), 
        np.linspace(-1, 1, 20)
    )
    
    # Tile to match image dimensions
    world_points = np.zeros((S, H, W, 3))
    for i in range(S):
        # Simple depth-based points
        depth = np.random.uniform(2, 4, (H, W))
        
        # Convert pixel coordinates to world coordinates
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        
        # Simple pinhole projection (inverse)
        fx = fy = 500  # Focal length
        cx, cy = W//2, H//2
        
        world_x = (xx - cx) * depth / fx
        world_y = (yy - cy) * depth / fy
        world_z = depth
        
        world_points[i, :, :, 0] = world_x
        world_points[i, :, :, 1] = world_y  
        world_points[i, :, :, 2] = world_z
    
    # Create synthetic images
    images = np.random.randint(0, 255, (S, H, W, 3), dtype=np.uint8)
    
    # Create synthetic camera extrinsics (cameras around a circle)
    extrinsics = []
    for i in range(S):
        angle = 2 * np.pi * i / S
        radius = 3.0
        
        # Camera position
        x = radius * np.cos(angle)
        y = radius * np.sin(angle) 
        z = 0.5
        
        # Look at origin
        target = np.array([0, 0, 0])
        position = np.array([x, y, z])
        up = np.array([0, 0, 1])
        
        # Create look-at matrix
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Create extrinsic matrix (world to camera)
        R = np.column_stack([right, up, -forward])
        t = -R @ position
        
        extrinsic = np.column_stack([R, t])  # 3x4 matrix
        extrinsics.append(extrinsic)
    
    extrinsics = np.array(extrinsics)
    
    # Create confidence scores
    world_points_conf = np.random.uniform(0.5, 1.0, (S, H, W))
    
    return {
        'world_points': world_points.astype(np.float32),
        'world_points_conf': world_points_conf.astype(np.float32),
        'images': images,
        'extrinsic': extrinsics.astype(np.float32)
    }


if __name__ == "__main__":
    main()