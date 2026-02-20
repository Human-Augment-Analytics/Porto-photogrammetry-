#!/usr/bin/env python3
"""
Manual Transform JSON Generator for NeuS2
Generates transforms.json for lab setup with:
- 3 Canon EOS RP cameras in line array
- Object on rotating turntable (8-degree increments)
- Known camera intrinsics from EXIF data
"""

import numpy as np
import json
import argparse
import math
from pathlib import Path
from PIL import Image
import os
import glob

def get_image_dimensions(image_dir):
    """
    Get image dimensions from the first image in the directory
    
    Args:
        image_dir: Path to directory containing images
        
    Returns:
        tuple: (width, height) or None if no images found
    """
    if not os.path.exists(image_dir):
        return None
        
    # Look for common image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        return None
    
    # Sort to get consistent first image
    image_files.sort()
    
    try:
        with Image.open(image_files[0]) as img:
            return img.size  # PIL returns (width, height)
    except Exception as e:
        print(f"Warning: Could not read image {image_files[0]}: {e}")
        return None

def validate_directories(image_dir, mask_dir=None, output_dir=None):
    """
    Validate input and output directories
    
    Args:
        image_dir: Path to images directory
        mask_dir: Optional path to masks directory
        output_dir: Path to output directory
        
    Returns:
        dict: Validation results
    """
    results = {
        'image_dir_exists': os.path.exists(image_dir),
        'image_count': 0,
        'mask_dir_exists': False,
        'mask_count': 0,
        'output_dir_writable': True
    }
    
    # Check image directory
    if results['image_dir_exists']:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
        for ext in extensions:
            results['image_count'] += len(glob.glob(os.path.join(image_dir, ext)))
            results['image_count'] += len(glob.glob(os.path.join(image_dir, ext.upper())))
    
    # Check mask directory
    if mask_dir and os.path.exists(mask_dir):
        results['mask_dir_exists'] = True
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp']
        for ext in extensions:
            results['mask_count'] += len(glob.glob(os.path.join(mask_dir, ext)))
            results['mask_count'] += len(glob.glob(os.path.join(mask_dir, ext.upper())))
    
    # Check output directory
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(output_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception:
            results['output_dir_writable'] = False
    
    return results

def create_camera_intrinsics(focal_length_mm=50, sensor_width_mm=35.9, sensor_height_mm=24.0, 
                           image_width=1600, image_height=1200):
    """
    Calculate camera intrinsics for Canon EOS RP
    
    Canon EOS RP specs:
    - Full frame sensor: 35.9 × 24.0 mm
    - 26.2 MP (6240 × 4160 max resolution)
    """
    # Calculate focal length in pixels
    fx = (focal_length_mm / sensor_width_mm) * image_width
    fy = (focal_length_mm / sensor_height_mm) * image_height
    
    # Principal point (center of image)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Camera angle calculations
    camera_angle_x = 2 * math.atan(sensor_width_mm / (2 * focal_length_mm))
    camera_angle_y = 2 * math.atan(sensor_height_mm / (2 * focal_length_mm))
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'camera_angle_x': camera_angle_x,
        'camera_angle_y': camera_angle_y,
        'k1': 0.0,  # Assume no distortion for high-quality setup
        'k2': 0.0,
        'k3': 0.0,
        'k4': 0.0,
        'p1': 0.0,
        'p2': 0.0
    }

def create_rotation_matrix_y(angle_degrees):
    """Create rotation matrix for Y-axis rotation (turntable)"""
    theta = math.radians(angle_degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    return np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ])

def create_camera_positions(distance_from_center=2.0, camera_spacing=0.5):
    """
    Create camera positions for line array setup
    
    Args:
        distance_from_center: Distance from turntable center to cameras (meters)
        camera_spacing: Vertical spacing between cameras (meters)
    
    Returns:
        List of camera positions and orientations
    """
    camera_configs = []
    
    # Camera positions in line array (stacked vertically)
    base_positions = [
        np.array([distance_from_center, -camera_spacing, 0, 1]),  # Camera 1 (bottom)
        np.array([distance_from_center, 0, 0, 1]),                # Camera 2 (middle)  
        np.array([distance_from_center, camera_spacing, 0, 1])     # Camera 3 (top)
    ]
    
    for i, pos in enumerate(base_positions):
        # Create camera-to-world transform
        # Camera looks at origin from its position
        camera_pos = pos[:3]
        look_at = np.array([0, 0, 0])  # Look at turntable center
        up = np.array([0, 0, 1])       # Z-up coordinate system
        
        # Calculate camera orientation
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        camera_up = np.cross(right, forward)
        camera_up = camera_up / np.linalg.norm(camera_up)
        
        # Create 4x4 transform matrix (camera-to-world)
        transform = np.eye(4)
        transform[0, :3] = right
        transform[1, :3] = camera_up  
        transform[2, :3] = -forward  # Camera looks in -Z direction
        transform[:3, 3] = camera_pos
        
        camera_configs.append({
            'position': camera_pos,
            'transform': transform,
            'camera_id': i
        })
    
    return camera_configs

def apply_coordinate_system_transform(transform):
    """
    Apply NeuS2 coordinate system conventions
    Following the COLMAP->NeRF conversion pattern from the reference script
    """
    # Following the coordinate system from the reference transform.json
    # The reference appears to use a coordinate system where:
    # - Cameras look towards the center
    # - Scene is centered and scaled appropriately
    
    # Apply coordinate system conversion (similar to COLMAP->NeRF)
    # Flip Y and Z axes to match NeRF convention
    flip_matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0], 
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    return np.dot(transform, flip_matrix)

def generate_file_paths(image_dir, mask_dir, num_frames, use_existing_files=True):
    """
    Generate file paths for images and masks
    
    Args:
        image_dir: Path to images directory
        mask_dir: Path to masks directory (optional)
        num_frames: Number of frames to generate
        use_existing_files: If True, use existing filenames; if False, generate sequential names
        
    Returns:
        list: List of (image_path, mask_path) tuples
    """
    file_paths = []
    
    if use_existing_files and os.path.exists(image_dir):
        # Use existing image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        image_files.sort()
        
        # Limit to available files or requested number
        image_files = image_files[:num_frames]
        
        for img_path in image_files:
            # Convert to relative path
            rel_img_path = os.path.relpath(img_path)
            
            # Find corresponding mask if mask_dir exists
            mask_path = None
            if mask_dir and os.path.exists(mask_dir):
                img_name = os.path.basename(img_path)
                name_without_ext = os.path.splitext(img_name)[0]
                
                # Try different mask extensions
                for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                    potential_mask = os.path.join(mask_dir, name_without_ext + ext)
                    if os.path.exists(potential_mask):
                        mask_path = os.path.relpath(potential_mask)
                        break
            
            file_paths.append((rel_img_path, mask_path))
    
    else:
        # Generate sequential filenames
        image_dir_name = os.path.basename(image_dir.rstrip('/'))
        mask_dir_name = os.path.basename(mask_dir.rstrip('/')) if mask_dir else None
        
        for i in range(num_frames):
            img_path = os.path.join(image_dir_name, f"{i:06d}.png")
            mask_path = os.path.join(mask_dir_name, f"{i:06d}.png") if mask_dir_name else None
            file_paths.append((img_path, mask_path))
    
    return file_paths

def generate_transforms_json(image_dir, mask_dir=None, output_path="transforms.json",
                           num_rotation_steps=45, rotation_step_degrees=8, 
                           image_width=None, image_height=None,
                           distance_from_center=2.0, camera_spacing=0.5,
                           aabb_scale=4, scale=0.5, use_existing_files=True):
    """
    Generate transforms.json for NeuS2
    
    Args:
        image_dir: Path to directory containing images
        mask_dir: Path to directory containing masks (optional)
        output_path: Output file path
        num_rotation_steps: Number of rotation positions
        rotation_step_degrees: Degrees per rotation step
        image_width/height: Image dimensions (auto-detected if None)
        distance_from_center: Camera distance from turntable center
        camera_spacing: Vertical spacing between cameras
        aabb_scale: Scene bounding box scale for NeuS2
        scale: Global scale factor
        use_existing_files: Use existing image filenames instead of generating sequential names
    """
    
    # Validate directories
    validation = validate_directories(image_dir, mask_dir, os.path.dirname(output_path))
    
    if not validation['image_dir_exists']:
        print(f"Warning: Image directory '{image_dir}' does not exist")
    else:
        print(f"Found {validation['image_count']} images in '{image_dir}'")
    
    if mask_dir:
        if validation['mask_dir_exists']:
            print(f"Found {validation['mask_count']} masks in '{mask_dir}'")
        else:
            print(f"Warning: Mask directory '{mask_dir}' does not exist")
    
    if not validation['output_dir_writable']:
        raise ValueError(f"Cannot write to output directory: {os.path.dirname(output_path)}")
    
    # Auto-detect image dimensions if not provided
    if image_width is None or image_height is None:
        detected_dims = get_image_dimensions(image_dir)
        if detected_dims:
            image_width, image_height = detected_dims
            print(f"Auto-detected image dimensions: {image_width} x {image_height}")
        else:
            if image_width is None or image_height is None:
                print("Warning: Could not detect image dimensions, using defaults (1600x1200)")
                image_width = image_width or 1600
                image_height = image_height or 1200
    
    # Calculate camera intrinsics
    intrinsics = create_camera_intrinsics(image_width=image_width, image_height=image_height)
    
    # Get camera positions
    camera_configs = create_camera_positions(distance_from_center, camera_spacing)
    
    # Calculate total frames
    total_frames = num_rotation_steps * len(camera_configs)
    
    # Generate file paths
    file_paths = generate_file_paths(image_dir, mask_dir, total_frames, use_existing_files)
    
    if len(file_paths) < total_frames:
        print(f"Warning: Only found {len(file_paths)} files, but need {total_frames} frames")
        if not use_existing_files:
            print("Consider using --use_existing_files flag to use actual image filenames")
    
    # Create the transforms JSON structure
    transforms_data = {
        "w": int(image_width),
        "h": int(image_height),
        "aabb_scale": aabb_scale,
        "scale": scale,
        "offset": [0.5, 0.5, 0.5],
        "from_na": True,  # Indicates this is from neural angular mapping
        "frames": []
    }
    
    frame_idx = 0
    
    # For each rotation step
    for rotation_step in range(num_rotation_steps):
        rotation_angle = rotation_step * rotation_step_degrees
        rotation_matrix = create_rotation_matrix_y(rotation_angle)
        
        # For each camera in the array
        for cam_idx, camera_config in enumerate(camera_configs):
            if frame_idx >= len(file_paths):
                break
                
            # Apply turntable rotation to the world, keeping cameras fixed
            # This rotates the object relative to the cameras
            world_transform = np.dot(rotation_matrix, camera_config['transform'])
            
            # Apply coordinate system transformation for NeuS2/NeRF compatibility
            final_transform = apply_coordinate_system_transform(world_transform)
            
            # Create intrinsic matrix in the format expected by NeuS2
            intrinsic_matrix = [
                [intrinsics['fx'], 0.0, intrinsics['cx'], 0.0],
                [0.0, intrinsics['fy'], intrinsics['cy'], 0.0], 
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
            
            # Get file paths
            img_path, mask_path = file_paths[frame_idx]
            
            # Create frame entry
            frame = {
                "file_path": img_path,
                "transform_matrix": final_transform.tolist(),
                "intrinsic_matrix": intrinsic_matrix
            }
            
            # Add mask path if available
            if mask_path:
                frame["mask_path"] = mask_path
            
            transforms_data["frames"].append(frame)
            frame_idx += 1
    
    # Add global camera parameters (from first camera for consistency)
    transforms_data.update({
        "camera_angle_x": intrinsics['camera_angle_x'],
        "camera_angle_y": intrinsics['camera_angle_y'],
        "fl_x": intrinsics['fx'],
        "fl_y": intrinsics['fy'],
        "k1": intrinsics['k1'],
        "k2": intrinsics['k2'],
        "k3": intrinsics['k3'], 
        "k4": intrinsics['k4'],
        "p1": intrinsics['p1'],
        "p2": intrinsics['p2'],
        "cx": intrinsics['cx'],
        "cy": intrinsics['cy'],
        "is_fisheye": False
    })
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(transforms_data, f, indent=2)
    
    print(f"\nGenerated transforms.json with {len(transforms_data['frames'])} frames")
    print(f"Rotation steps: {num_rotation_steps}")
    print(f"Cameras per step: {len(camera_configs)}")
    print(f"Expected total frames: {num_rotation_steps * len(camera_configs)}")
    print(f"Actual frames generated: {len(transforms_data['frames'])}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Saved to: {output_path}")
    
    return transforms_data

def main():
    parser = argparse.ArgumentParser(description="Generate transforms.json for NeuS2 lab setup")
    
    # Input/Output paths
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Path to directory containing input images")
    parser.add_argument("--mask_dir", type=str, default=None,
                       help="Path to directory containing mask images (optional)")
    parser.add_argument("--output", type=str, default="transforms.json",
                       help="Output file path (default: transforms.json)")
    
    # Image parameters
    parser.add_argument("--width", type=int, default=None,
                       help="Image width in pixels (auto-detected if not specified)")
    parser.add_argument("--height", type=int, default=None, 
                       help="Image height in pixels (auto-detected if not specified)")
    
    # Capture parameters
    parser.add_argument("--rotation_steps", type=int, default=45, 
                       help="Number of rotation positions (default: 45)")
    parser.add_argument("--rotation_degrees", type=float, default=8.0,
                       help="Degrees per rotation step (default: 8.0)")
    
    # Physical setup
    parser.add_argument("--distance", type=float, default=2.0,
                       help="Camera distance from center in meters (default: 2.0)")
    parser.add_argument("--spacing", type=float, default=0.5,
                       help="Camera spacing in meters (default: 0.5)")
    
    # NeuS2 parameters
    parser.add_argument("--aabb_scale", type=int, default=4,
                       help="Scene bounding box scale for NeuS2 (default: 4)")
    parser.add_argument("--scale", type=float, default=0.5,
                       help="Global scale factor (default: 0.5)")
    
    # File handling
    parser.add_argument("--use_existing_files", action="store_true", default=True,
                       help="Use existing image filenames (default: True)")
    parser.add_argument("--generate_sequential", action="store_true",
                       help="Generate sequential filenames instead of using existing files")
    
    args = parser.parse_args()
    
    # Handle conflicting file options
    use_existing = args.use_existing_files and not args.generate_sequential
    
    generate_transforms_json(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_path=args.output,
        num_rotation_steps=args.rotation_steps,
        rotation_step_degrees=args.rotation_degrees,
        image_width=args.width,
        image_height=args.height,
        distance_from_center=args.distance,
        camera_spacing=args.spacing,
        aabb_scale=args.aabb_scale,
        scale=args.scale,
        use_existing_files=use_existing
    )

if __name__ == "__main__":
    main()