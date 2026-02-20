#!/usr/bin/env python3
"""
NeuS2 Training Diagnostics
Debug common issues with NeuS2 training setup
"""

import json
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def check_data_structure(data_dir):
    """Check if data directory structure is correct for NeuS2"""
    print("=== Data Structure Check ===")
    
    # Check for required files
    transforms_path = os.path.join(data_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        print("transforms.json not found")
        return False
    else:
        print("transforms.json found")
    
    # Check for images directory
    images_dir = os.path.join(data_dir, "images")
    if not os.path.exists(images_dir):
        print("images/ directory not found")
        return False
    else:
        print("images/ directory found")
    
    # Count images using case-insensitive search
    image_count = 0
    for root, dirs, files in os.walk(images_dir):
        if root != images_dir:  # Only look in the top level directory
            continue
        for file in files:
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']):
                image_count += 1
    
    print(f"Found {image_count} images")
    
    # Check for masks
    masks_dir = os.path.join(data_dir, "masks")
    if os.path.exists(masks_dir):
        mask_count = 0
        for root, dirs, files in os.walk(masks_dir):
            if root != masks_dir:  # Only look in the top level directory
                continue
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']):
                    mask_count += 1
        print(f"masks/ directory found with {mask_count} masks")
    else:
        print("masks/ directory not found (optional)")
    
    return True

def analyze_transforms_json(transforms_path):
    """Analyze transforms.json for common issues"""
    print("\n=== Transforms Analysis ===")
    
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    # Check basic structure
    required_fields = ['w', 'h', 'frames']
    for field in required_fields:
        if field in data:
            print(f"OK {field}: {data[field] if field != 'frames' else len(data[field])} {'frames' if field == 'frames' else ''}")
        else:
            print(f"Missing required field: {field}")
    
    # Check camera parameters
    camera_fields = ['camera_angle_x', 'fl_x', 'fl_y', 'cx', 'cy']
    for field in camera_fields:
        if field in data:
            print(f"OK {field}: {data[field]:.2f}")
        else:
            print(f"Missing camera parameter: {field}")
    
    # Check AABB scale
    if 'aabb_scale' in data:
        aabb = data['aabb_scale']
        if aabb in [1, 2, 4, 8, 16, 32, 64, 128]:
            print(f"OK aabb_scale: {aabb} (valid)")
        else:
            print(f"WARNING aabb_scale: {aabb} (should be power of 2)")
    
    # Analyze camera positions
    if 'frames' in data and data['frames']:
        positions = []
        for frame in data['frames']:
            if 'transform_matrix' in frame:
                transform = np.array(frame['transform_matrix'])
                pos = transform[:3, 3]
                positions.append(pos)
        
        if positions:
            positions = np.array(positions)
            print(f"\nCamera Position Analysis:")
            print(f"   Number of cameras: {len(positions)}")
            print(f"   Distance from origin: {np.linalg.norm(positions, axis=1).mean():.3f} ± {np.linalg.norm(positions, axis=1).std():.3f}")
            print(f"   X range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
            print(f"   Y range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
            print(f"   Z range: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    
    return data

def check_image_quality(images_dir, sample_size=5):
    """Check image quality and properties"""
    print(f"\n=== Image Quality Check ===")
    
    # Get image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    if not image_files:
        print("No images found")
        return
    
    image_files.sort()
    sample_files = image_files[:min(sample_size, len(image_files))]
    
    dimensions = []
    formats = []
    file_sizes = []
    
    for img_path in sample_files:
        try:
            with Image.open(img_path) as img:
                dimensions.append(img.size)
                formats.append(img.format)
                file_sizes.append(os.path.getsize(img_path) / 1024 / 1024)  # MB
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    if dimensions:
        unique_dims = list(set(dimensions))
        print(f"Image dimensions: {unique_dims}")
        if len(unique_dims) > 1:
            print("Warning: Multiple image dimensions found")
        
        print(f"Image formats: {list(set(formats))}")
        print(f"Average file size: {np.mean(file_sizes):.2f} MB")
        
        # Check if images are too large
        if np.mean(file_sizes) > 10:
            print("Warning: Large image files detected. Consider resizing for faster training.")

def analyze_mask_quality(masks_dir, images_dir):
    """Analyze mask quality and coverage"""
    if not os.path.exists(masks_dir):
        return
    
    print(f"\n=== Mask Quality Check ===")
    
    # Get mask files
    extensions = ['*.png', '*.jpg', '*.jpeg']
    mask_files = []
    for ext in extensions:
        mask_files.extend(glob.glob(os.path.join(masks_dir, ext)))
    
    if not mask_files:
        print("No mask files found")
        return
    
    mask_files.sort()
    
    # Check a sample of masks
    sample_masks = mask_files[:5]
    
    for mask_path in sample_masks:
        try:
            with Image.open(mask_path) as mask:
                mask_array = np.array(mask)
                
                # Check if it's binary
                unique_values = np.unique(mask_array)
                if len(unique_values) == 2 and 0 in unique_values and 255 in unique_values:
                    print(f"{os.path.basename(mask_path)}: Binary mask")
                else:
                    print(f"{os.path.basename(mask_path)}: Non-binary mask (values: {unique_values})")
                
                # Check coverage
                coverage = np.sum(mask_array > 127) / mask_array.size
                print(f"   Coverage: {coverage:.1%}")
                
        except Exception as e:
            print(f"Error reading {mask_path}: {e}")

def suggest_fixes(data_dir):
    """Suggest fixes for common issues"""
    print(f"\n=== Suggested Fixes ===")
    
    transforms_path = os.path.join(data_dir, "transforms.json")
    
    if not os.path.exists(transforms_path):
        print("1. Generate transforms.json:")
        print("   python neus2_fixed_generator.py --image_dir images/ --output transforms.json")
        return
    
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    # Check image dimensions
    if 'w' in data and 'h' in data:
        w, h = data['w'], data['h']
        total_pixels = w * h
        
        if total_pixels > 2048 * 1536:
            print("1. Resize images for faster training:")
            print("   python image_resizer.py --input_dir images/ --output_dir images_small/ --preset medium")
            print("   python neus2_fixed_generator.py --image_dir images_small/ --output transforms_small.json")
    
    # Check AABB scale
    aabb = data.get('aabb_scale', 1)
    if aabb == 1:
        print("2. Consider increasing AABB scale for better scene coverage:")
        print("   Edit transforms.json and change 'aabb_scale' to 4 or 8")
    
    # Check mask setup
    if not os.path.exists(os.path.join(data_dir, "masks")):
        print("3. Add masks for better background handling:")
        print("   Create masks/ directory with binary masks for each image")
    
    print("4. Start with shorter training for testing:")
    print("   python run.py --name test --scene . --n_steps 1000 --save_mesh")

def create_visualization(data_dir, output_path="camera_debug.png"):
    """Create a visualization of the camera setup"""
    transforms_path = os.path.join(data_dir, "transforms.json")
    
    if not os.path.exists(transforms_path):
        print("No transforms.json found for visualization")
        return
    
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    if 'frames' not in data or not data['frames']:
        print("No frames found in transforms.json")
        return
    
    # Extract camera positions
    positions = []
    for frame in data['frames']:
        if 'transform_matrix' in frame:
            transform = np.array(frame['transform_matrix'])
            pos = transform[:3, 3]
            positions.append(pos)
    
    if not positions:
        print("No valid camera positions found")
        return
    
    positions = np.array(positions)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
              c='red', s=20, alpha=0.6, label='Cameras')
    
    # Plot turntable center
    ax.scatter(0, 0, 0, c='green', s=100, marker='s', label='Object Center')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Setup Visualization')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.abs(positions).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Camera visualization saved to: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose NeuS2 training setup")
    parser.add_argument("--data_dir", type=str, default=".", 
                       help="Data directory containing transforms.json and images/")
    parser.add_argument("--create_viz", action="store_true",
                       help="Create camera visualization")
    
    args = parser.parse_args()
    
    print("NeuS2 Training Diagnostics")
    print("=" * 50)
    
    # Run diagnostics
    if check_data_structure(args.data_dir):
        transforms_path = os.path.join(args.data_dir, "transforms.json")
        analyze_transforms_json(transforms_path)
        
        images_dir = os.path.join(args.data_dir, "images")
        check_image_quality(images_dir)
        
        masks_dir = os.path.join(args.data_dir, "masks")
        analyze_mask_quality(masks_dir, images_dir)
        
        if args.create_viz:
            create_visualization(args.data_dir)
    
    suggest_fixes(args.data_dir)

if __name__ == "__main__":
    main()