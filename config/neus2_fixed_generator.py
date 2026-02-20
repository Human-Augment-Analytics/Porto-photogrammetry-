#!/usr/bin/env python3
"""
NeuS2 Transform Generator with Corrected Camera Positioning Logic

This script generates transforms.json for NeuS2 training from a turntable setup with:
- 3 fixed cameras in vertical line array
- Object rotating on turntable (8° increments, 45 steps = 360°)
- Sequential image naming (000000.jpg, 000001.jpg, etc.)

Key fixes:
1. Proper 4x4 transform matrix multiplication (not simple vector rotation)
2. Two corrected approaches: inverse rotation vs same direction rotation
3. No more elliptical camera paths - clean circular orbits
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

def create_camera_positions(distance_from_center: float = 2.0, camera_spacing: float = 0.5) -> List[Dict]:
    """
    Create camera positions for line array setup.
    
    Args:
        distance_from_center: Distance of cameras from origin (object center)
        camera_spacing: Vertical spacing between cameras
        
    Returns:
        List of camera configurations with position and transform matrix
    """
    camera_configs = []
    
    # Camera positions in line array (stacked vertically)
    base_positions = [
        np.array([distance_from_center, -camera_spacing, 0, 1]),  # Camera 1 (bottom)
        np.array([distance_from_center, 0, 0, 1]),                # Camera 2 (middle)  
        np.array([distance_from_center, camera_spacing, 0, 1])     # Camera 3 (top)
    ]
    
    for i, pos in enumerate(base_positions):
        camera_pos = pos[:3]
        look_at = np.array([0, 0, 0])  # All cameras look at object center
        up = np.array([0, 0, 1])       # Z-up coordinate system
        
        # Calculate camera orientation using standard camera coordinate system
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        camera_up = np.cross(right, forward)
        camera_up = camera_up / np.linalg.norm(camera_up)
        
        # Create 4x4 camera-to-world transform matrix
        transform = np.eye(4)
        transform[0, :3] = right
        transform[1, :3] = camera_up  
        transform[2, :3] = -forward  # Camera looks down negative Z
        transform[:3, 3] = camera_pos
        
        camera_configs.append({
            'position': camera_pos,
            'transform': transform,
            'camera_id': i
        })
    
    return camera_configs

def create_rotation_matrix_y(angle_degrees: float) -> np.ndarray:
    """
    Create 4x4 rotation matrix for Y-axis rotation (turntable rotation).
    
    Args:
        angle_degrees: Rotation angle in degrees
        
    Returns:
        4x4 rotation matrix
    """
    theta = np.radians(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    return np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ])

def generate_transforms_inverse_rotation(
    image_dir: str,
    num_cameras: int = 3,
    rotation_steps: int = 45,
    rotation_increment: float = 8.0,
    distance_from_center: float = 2.0,
    camera_spacing: float = 0.5,
    focal_length: float = 1100.0,
    image_width: int = 1024,
    image_height: int = 768
) -> Dict:
    """
    Generate transforms using INVERSE ROTATION approach with SEQUENTIAL camera organization.
    
    File organization (CORRECTED):
    - 000000-000044: Camera 1 (45 rotation steps, height -0.5)
    - 000046-000090: Camera 2 (45 rotation steps, height 0.0) [skipping 000045]
    - 000092-000135: Camera 3 (44 rotation steps, height +0.5) [skipping 000091]
    
    Each camera captures the full 360° rotation at its own height.
    """
    print("Generating transforms with INVERSE ROTATION and sequential organization...")
    print("Object rotates clockwise → Cameras orbit counter-clockwise relative to object")
    
    camera_configs = create_camera_positions(distance_from_center, camera_spacing)
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {image_dir}")
    
    print(f"Found {len(image_files)} images")
    
    frames = []
    
    # Camera 1: 000000-000044 (45 images, height -0.5)
    camera_1_config = camera_configs[0]  # Bottom camera (-0.5 height)
    for step in range(45):
        if step >= len(image_files):
            break
        
        rotation_angle = step * rotation_increment
        # INVERSE rotation: object rotates +8°, cameras appear to rotate -8° relative to object
        inverse_rotation_matrix = create_rotation_matrix_y(-rotation_angle)
        
        # Apply inverse rotation to camera 1's transform
        relative_transform = np.dot(inverse_rotation_matrix, camera_1_config['transform'])
        
        frame_data = {
            "file_path": image_files[step],
            "transform_matrix": relative_transform.tolist()
        }
        frames.append(frame_data)
    
    # Camera 2: 000046-000090 (45 images, height 0.0, skipping 000045)
    camera_2_config = camera_configs[1]  # Middle camera (0.0 height)
    camera_2_start = 45  # Start after camera 1, but skip 000045
    for step in range(45):
        image_idx = camera_2_start + step
        if image_idx >= len(image_files):
            break
        
        rotation_angle = step * rotation_increment
        inverse_rotation_matrix = create_rotation_matrix_y(-rotation_angle)
        
        # Apply inverse rotation to camera 2's transform
        relative_transform = np.dot(inverse_rotation_matrix, camera_2_config['transform'])
        
        frame_data = {
            "file_path": image_files[image_idx],
            "transform_matrix": relative_transform.tolist()
        }
        frames.append(frame_data)
    
    # Camera 3: 000092-000135 (44 images, height +0.5, skipping 000091)
    camera_3_config = camera_configs[2]  # Top camera (+0.5 height)
    camera_3_start = 91  # Start after camera 2, but skip 000091
    for step in range(45):  # Only 44 steps for camera 3
        image_idx = camera_3_start + step
        if image_idx >= len(image_files):
            break
        
        rotation_angle = step * rotation_increment
        inverse_rotation_matrix = create_rotation_matrix_y(-rotation_angle)
        
        # Apply inverse rotation to camera 3's transform
        relative_transform = np.dot(inverse_rotation_matrix, camera_3_config['transform'])
        
        frame_data = {
            "file_path": image_files[image_idx],
            "transform_matrix": relative_transform.tolist()
        }
        frames.append(frame_data)
    
    # Camera intrinsics
    fl_x = fl_y = focal_length
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    transforms_data = {
        "camera_angle_x": 2 * np.arctan(image_width / (2 * fl_x)),
        "camera_angle_y": 2 * np.arctan(image_height / (2 * fl_y)),
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": image_width,
        "h": image_height,
        "frames": frames
    }
    
    print(f"Generated {len(frames)} camera poses using inverse rotation")
    return transforms_data

def generate_transforms_same_direction(
    image_dir: str,
    num_cameras: int = 3,
    rotation_steps: int = 45,
    rotation_increment: float = 8.0,
    distance_from_center: float = 2.0,
    camera_spacing: float = 0.5,
    focal_length: float = 1100.0,
    image_width: int = 1024,
    image_height: int = 768
) -> Dict:
    """
    Generate transforms using SAME DIRECTION approach with SEQUENTIAL camera organization.
    
    File organization (CORRECTED):
    - 000000-000044: Camera 1 (45 rotation steps, height -0.5)
    - 000046-000090: Camera 2 (45 rotation steps, height 0.0) [skipping 000045]
    - 000092-000135: Camera 3 (44 rotation steps, height +0.5) [skipping 000091]
    
    Each camera captures the full 360° rotation at its own height.
    """
    print("Generating transforms with SEQUENTIAL camera organization...")
    print("Camera 1: all rotation steps at height -0.5")
    print("Camera 2: all rotation steps at height 0.0") 
    print("Camera 3: all rotation steps at height +0.5")
    
    camera_configs = create_camera_positions(distance_from_center, camera_spacing)
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {image_dir}")
    
    print(f"Found {len(image_files)} images")
    
    frames = []
    
    # Camera 1: 000000-000044 (45 images, height -0.5)
    camera_1_config = camera_configs[0]  # Bottom camera (-0.5 height)
    for step in range(45):
        if step >= len(image_files):
            break
        
        rotation_angle = step * rotation_increment
        rotation_matrix = create_rotation_matrix_y(rotation_angle)
        
        # Apply rotation to camera 1's transform
        relative_transform = np.dot(rotation_matrix, camera_1_config['transform'])
        
        frame_data = {
            "file_path": image_files[step],
            "transform_matrix": relative_transform.tolist()
        }
        frames.append(frame_data)
    
    # Camera 2: 000046-000090 (45 images, height 0.0, skipping 000045)
    camera_2_config = camera_configs[1]  # Middle camera (0.0 height)
    camera_2_start = 45  # Start after camera 1, but skip 000045
    for step in range(45):
        image_idx = camera_2_start + step
        if image_idx >= len(image_files):
            break
        
        rotation_angle = step * rotation_increment
        rotation_matrix = create_rotation_matrix_y(rotation_angle)
        
        # Apply rotation to camera 2's transform
        relative_transform = np.dot(rotation_matrix, camera_2_config['transform'])
        
        frame_data = {
            "file_path": image_files[image_idx],
            "transform_matrix": relative_transform.tolist()
        }
        frames.append(frame_data)
    
    # Camera 3: 000092-000136 (45 images, height +0.5, skipping 000091)
    camera_3_config = camera_configs[2]  # Top camera (+0.5 height)
    camera_3_start = 90  # Start after camera 2, but skip 000091
    for step in range(45): 
        image_idx = camera_3_start + step
        if image_idx >= len(image_files):
            break
        
        rotation_angle = step * rotation_increment
        rotation_matrix = create_rotation_matrix_y(rotation_angle)
        
        # Apply rotation to camera 3's transform
        relative_transform = np.dot(rotation_matrix, camera_3_config['transform'])
        
        frame_data = {
            "file_path": image_files[image_idx],
            "transform_matrix": relative_transform.tolist()
        }
        frames.append(frame_data)
    
    # Camera intrinsics
    fl_x = fl_y = focal_length
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    transforms_data = {
        "camera_angle_x": 2 * np.arctan(image_width / (2 * fl_x)),
        "camera_angle_y": 2 * np.arctan(image_height / (2 * fl_y)),
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": image_width,
        "h": image_height,
        "frames": frames
    }
    
    print(f"Generated {len(frames)} camera poses with sequential organization")
    return transforms_data

def validate_transforms(transforms_data: Dict) -> None:
    """
    Validate the generated transforms and print diagnostics.
    """
    frames = transforms_data['frames']
    positions = []
    
    for frame in frames:
        transform = np.array(frame['transform_matrix'])
        pos = transform[:3, 3]  # Extract translation
        positions.append(pos)
    
    positions = np.array(positions)
    
    print(f"\n=== Transform Validation ===")
    print(f"Total frames: {len(frames)}")
    print(f"Camera positions range:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    
    # Check for circular pattern
    distances = np.linalg.norm(positions, axis=1)
    print(f"Distance from origin: [{distances.min():.3f}, {distances.max():.3f}]")
    
    # Check if we have 3 distinct height levels
    unique_z = np.unique(np.round(positions[:, 2], 3))
    print(f"Camera height levels: {len(unique_z)} ({unique_z})")
    
    if len(unique_z) == 3:
        print("✓ Detected 3 camera height levels (vertical line array)")
    else:
        print("⚠ Warning: Expected 3 camera height levels")
    
    # Check for circular motion in X-Y plane
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    
    if x_range > 2.0 and y_range > 2.0:
        print("✓ Detected circular camera motion (good for NeRF)")
    else:
        print("⚠ Warning: Camera motion may not be sufficiently circular")

def main():
    parser = argparse.ArgumentParser(description="Generate NeuS2 transforms.json with corrected camera positioning")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing sequential images")
    parser.add_argument("--output", type=str, default="transforms.json", help="Output transforms file")
    parser.add_argument("--approach", type=str, choices=["inverse", "same"], default="same",
                       help="Rotation approach: 'inverse' (counter-clockwise) or 'same' (clockwise)")
    parser.add_argument("--distance", type=float, default=2.0, help="Camera distance from center")
    parser.add_argument("--spacing", type=float, default=0.5, help="Vertical spacing between cameras")
    parser.add_argument("--focal_length", type=float, default=1100.0, help="Camera focal length")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--rotation_steps", type=int, default=45, help="Number of rotation steps")
    parser.add_argument("--rotation_increment", type=float, default=8.0, help="Degrees per rotation step")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        raise ValueError(f"Image directory not found: {args.image_dir}")
    
    print(f"NeuS2 Transform Generator (Corrected)")
    print(f"=====================================")
    print(f"Image directory: {args.image_dir}")
    print(f"Output file: {args.output}")
    print(f"Approach: {args.approach} rotation")
    print(f"Camera setup: 3 cameras, {args.distance}m from center, {args.spacing}m vertical spacing")
    print(f"Turntable: {args.rotation_steps} steps × {args.rotation_increment}° = {args.rotation_steps * args.rotation_increment}°")
    
    # Generate transforms using selected approach
    if args.approach == "inverse":
        transforms_data = generate_transforms_inverse_rotation(
            args.image_dir,
            rotation_steps=args.rotation_steps,
            rotation_increment=args.rotation_increment,
            distance_from_center=args.distance,
            camera_spacing=args.spacing,
            focal_length=args.focal_length,
            image_width=args.width,
            image_height=args.height
        )
    else:  # same direction
        transforms_data = generate_transforms_same_direction(
            args.image_dir,
            rotation_steps=args.rotation_steps,
            rotation_increment=args.rotation_increment,
            distance_from_center=args.distance,
            camera_spacing=args.spacing,
            focal_length=args.focal_length,
            image_width=args.width,
            image_height=args.height
        )
    
    # Validate the results
    validate_transforms(transforms_data)
    
    # Save transforms.json
    with open(args.output, 'w') as f:
        json.dump(transforms_data, f, indent=2)
    
    print(f"\n✓ Transforms saved to {args.output}")
    print(f"Ready for NeuS2 training!")
    print(f"\nNext steps:")
    print(f"1. python run.py --name test --scene . --n_steps 1000 --save_mesh")
    print(f"2. Check if {args.approach} rotation approach works well")
    print(f"3. If not, try the other approach: --approach {'same' if args.approach == 'inverse' else 'inverse'}")

if __name__ == "__main__":
    main()