#!/usr/bin/env python3
"""
VGG-T to NeuS2 Converter - Center of Attention Transform

This script converts VGG-T camera poses to NeuS2 transforms.json format by:
1. Loading VGG-T camera extrinsics and intrinsics
2. Recentering coordinate system around center of attention (where cameras look)
3. Generating proper transforms.json for NeuS2 training

The key insight: Use InstantNGP's "center of attention" approach instead of 
camera centroid - this finds where camera rays intersect (the object center).
"""

import json
import numpy as np
import argparse
import os
import glob
from pathlib import Path

def closest_point_2_lines(oa, da, ob, db):
    """
    Returns point closest to both rays of form o+t*d, and a weight factor 
    that goes to 0 if the lines are parallel.
    
    From InstantNGP's colmap2nerf.py script.
    """
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta*da + ob + tb*db) * 0.5, denom

def load_vggt_poses(poses_file):
    """Load VGG-T camera poses from JSON file."""
    print(f"Loading VGG-T poses from: {poses_file}")
    
    with open(poses_file, 'r') as f:
        data = json.load(f)
    
    # Convert back to numpy arrays
    extrinsics = np.array(data['extrinsic'])  # Shape: (N, 4, 4) - but VGG-T only gives 3x4
    intrinsics = np.array(data['intrinsic'])  # Shape: (N, 3, 3)
    
    # Handle VGG-T's 3x4 extrinsic format - add [0,0,0,1] row to make 4x4
    if extrinsics.shape[1] == 3:
        print("Converting VGG-T 3x4 extrinsics to 4x4 format...")
        num_cameras = extrinsics.shape[0]
        extrinsics_4x4 = np.zeros((num_cameras, 4, 4))
        extrinsics_4x4[:, :3, :] = extrinsics
        extrinsics_4x4[:, 3, 3] = 1.0  # Set bottom-right to 1
        extrinsics = extrinsics_4x4
    
    print(f"Loaded {len(extrinsics)} camera poses")
    print(f"Extrinsics shape: {extrinsics.shape}")
    print(f"Intrinsics shape: {intrinsics.shape}")
    
    return {
        'extrinsics': extrinsics,
        'intrinsics': intrinsics,
        'metadata': data.get('metadata', {})
    }

def calculate_center_of_attention(extrinsics):
    """
    Calculate center of attention using InstantNGP's method.
    This finds where camera rays intersect (where cameras are looking).
    """
    print(f"Computing center of attention using camera ray intersections...")
    
    # Extract camera positions and look directions
    camera_positions = extrinsics[:, :3, 3]
    
    print(f"Camera position range:")
    print(f"  X: [{camera_positions[:, 0].min():.3f}, {camera_positions[:, 0].max():.3f}]")
    print(f"  Y: [{camera_positions[:, 1].min():.3f}, {camera_positions[:, 1].max():.3f}]")
    print(f"  Z: [{camera_positions[:, 2].min():.3f}, {camera_positions[:, 2].max():.3f}]")
    
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    
    n_cameras = len(extrinsics)
    for i in range(n_cameras):
        for j in range(i + 1, n_cameras):
            # Get camera positions
            oa = extrinsics[i, :3, 3]
            ob = extrinsics[j, :3, 3]
            
            # Get camera look directions (negative Z-axis in camera coordinates)
            da = -extrinsics[i, :3, 2]  # Camera looks down -Z
            db = -extrinsics[j, :3, 2]
            
            # Find closest point between the two rays
            p, w = closest_point_2_lines(oa, da, ob, db)
            
            if w > 0.00001:  # Only use if rays aren't parallel
                totp += p * w
                totw += w
    
    if totw > 0.0:
        center_of_attention = totp / totw
    else:
        # Fallback to camera centroid if all rays are parallel
        print("Warning: All camera rays appear parallel, using camera centroid")
        center_of_attention = camera_positions.mean(axis=0)
    
    print(f"Detected center of attention: [{center_of_attention[0]:.3f}, {center_of_attention[1]:.3f}, {center_of_attention[2]:.3f}]")
    
    # Analyze camera distances for validation
    distances_from_center = np.linalg.norm(camera_positions - center_of_attention, axis=1)
    print(f"Camera distances from center: {distances_from_center.mean():.3f} ± {distances_from_center.std():.3f}")
    
    return center_of_attention, camera_positions

def recenter_camera_poses(extrinsics, center_of_attention):
    """Recenter camera poses so center of attention becomes origin."""
    print(f"Recentering poses around center of attention...")
    
    recentered_extrinsics = extrinsics.copy()
    
    # Translate all camera positions by -center_of_attention
    recentered_extrinsics[:, :3, 3] -= center_of_attention
    
    # Verify recentering worked
    new_positions = recentered_extrinsics[:, :3, 3]
    new_center = new_positions.mean(axis=0)
    
    print(f"After recentering:")
    print(f"  New position centroid: [{new_center[0]:.6f}, {new_center[1]:.6f}, {new_center[2]:.6f}]")
    print(f"  Position range:")
    print(f"    X: [{new_positions[:, 0].min():.3f}, {new_positions[:, 0].max():.3f}]")
    print(f"    Y: [{new_positions[:, 1].min():.3f}, {new_positions[:, 1].max():.3f}]")
    print(f"    Z: [{new_positions[:, 2].min():.3f}, {new_positions[:, 2].max():.3f}]")
    
    return recentered_extrinsics

def fix_camera_orientations(extrinsics):
    """Fix camera orientations by flipping Z-axis (most common fix for VGG-T)."""
    print(f"Applying camera orientation fix (Z-axis flip)...")
    
    fixed_extrinsics = extrinsics.copy()
    
    # Apply Z-axis flip to all cameras
    fixed_extrinsics[:, :3, 2] *= -1  # Flip Z-axis (forward direction)
    
    print(f"Applied Z-axis flip to {len(extrinsics)} cameras")
    return fixed_extrinsics

def validate_camera_orientations(extrinsics, sample_size=5):
    """Validate that cameras are pointing toward origin after recentering."""
    print(f"\nValidating camera orientations (checking {sample_size} cameras)...")
    
    valid_orientations = 0
    
    for i in range(min(sample_size, len(extrinsics))):
        transform = extrinsics[i]
        camera_pos = transform[:3, 3]
        
        # Camera forward direction (negative Z in camera coordinates)
        camera_forward = -transform[:3, 2]
        
        # Direction from camera to origin
        if np.linalg.norm(camera_pos) > 1e-6:  # Avoid division by zero
            to_origin = -camera_pos / np.linalg.norm(camera_pos)
            
            # Calculate angle between camera forward and direction to origin
            dot_product = np.dot(camera_forward, to_origin)
            angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
            
            print(f"  Camera {i:2d}: angle to origin = {angle_deg:5.1f}°", end="")
            
            if angle_deg < 15.0:  # Allow some tolerance
                print(" ✓")
                valid_orientations += 1
            else:
                print(" ⚠️")
        else:
            print(f"  Camera {i:2d}: at origin (skipping)")
    
    print(f"Camera orientations: {valid_orientations}/{sample_size} pointing toward origin")
    return valid_orientations >= sample_size * 0.7  # 70% should be correctly oriented

def find_matching_images(images_dir, num_cameras):
    """Find matching image files for the camera poses."""
    if not images_dir or not os.path.exists(images_dir):
        print(f"Warning: Images directory not found: {images_dir}")
        return [f"frame_{i:03d}.jpg" for i in range(num_cameras)]
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {images_dir}")
        return [f"frame_{i:03d}.jpg" for i in range(num_cameras)]
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    if len(image_files) != num_cameras:
        print(f"Warning: Number of images ({len(image_files)}) != number of cameras ({num_cameras})")
        # Take first N images or pad with duplicates
        if len(image_files) >= num_cameras:
            image_files = image_files[:num_cameras]
        else:
            # Repeat last image to fill gaps
            while len(image_files) < num_cameras:
                image_files.append(image_files[-1])
    
    # Convert to relative paths
    image_filenames = [os.path.basename(f) for f in image_files]
    return image_filenames

def generate_neus2_transforms(extrinsics, intrinsics, image_files):
    """Generate NeuS2 transforms.json format with InstantNGP-style scaling."""
    print(f"Generating NeuS2 transforms for {len(extrinsics)} cameras...")
    
    # Calculate scene scale from camera positions using InstantNGP approach
    camera_positions = extrinsics[:, :3, 3]
    distances_from_origin = np.linalg.norm(camera_positions, axis=1)
    avg_camera_distance = distances_from_origin.mean()
    
    print(f"Average camera distance from origin: {avg_camera_distance:.3f}")
    
    # InstantNGP scaling approach: scale to "NeRF size"
    # From InstantNGP script: scale cameras so they're 4.0 / avglen from origin
    target_scale = 4.0 / avg_camera_distance
    ngp_scale = 0.33
    
    print(f"Calculated scaling factors:")
    print(f"  InstantNGP scale: {target_scale:.3f} (4.0 / avg_distance)")
    print(f"  NGP scale: {ngp_scale} (NGP's default scaling)")
    print(f"  Final camera distance: {avg_camera_distance * target_scale * ngp_scale:.3f}")
    
    frames = []
    
    for i, (extrinsic, intrinsic, image_file) in enumerate(zip(extrinsics, intrinsics, image_files)):
        # Apply InstantNGP scaling to camera positions
        scaled_extrinsic = extrinsic.copy()
        scaled_extrinsic[:3, 3] *= target_scale  # Scale translation
        
        # Convert to list for JSON serialization
        transform_matrix = scaled_extrinsic.tolist()
        
        frame_data = {
            "file_path": f"images/{image_file}",
            "transform_matrix": transform_matrix
        }
        frames.append(frame_data)
    
    # Calculate average intrinsics for global parameters
    avg_intrinsic = intrinsics.mean(axis=0)
    fl_x = avg_intrinsic[0, 0]
    fl_y = avg_intrinsic[1, 1]
    cx = avg_intrinsic[0, 2]
    cy = avg_intrinsic[1, 2]
    
    # Estimate image dimensions from intrinsics (assume centered principal point)
    w = int(cx * 2)
    h = int(cy * 2)
    
    print(f"Average camera intrinsics:")
    print(f"  Focal length: fx={fl_x:.1f}, fy={fl_y:.1f}")
    print(f"  Principal point: cx={cx:.1f}, cy={cy:.1f}")
    print(f"  Estimated image size: {w}x{h}")
    
    # Calculate field of view
    camera_angle_x = 2 * np.arctan(w / (2 * fl_x))
    camera_angle_y = 2 * np.arctan(h / (2 * fl_y))
    
    transforms_data = {
        "camera_angle_x": float(camera_angle_x),
        "camera_angle_y": float(camera_angle_y),
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
        "w": w,
        "h": h,
        "scale": ngp_scale,
        "offset": [0.5, 0.5, 0.5],
        "aabb_scale": 1,
        "frames": frames
    }
    
    return transforms_data

def main():
    parser = argparse.ArgumentParser(
        description="Convert VGG-T camera poses to NeuS2 transforms.json format using center of attention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "poses_file",
        help="JSON file containing VGG-T camera poses"
    )
    
    parser.add_argument(
        "--output",
        default="transforms_vggt_coa.json",
        help="Output transforms.json file"
    )
    
    parser.add_argument(
        "--images_dir",
        help="Directory containing corresponding image files"
    )
    
    parser.add_argument(
        "--validate_orientations",
        action="store_true",
        help="Validate that cameras point toward origin after recentering"
    )
    
    parser.add_argument(
        "--skip_orientation_fix",
        action="store_true",
        help="Skip automatic camera orientation fix (Z-axis flip)"
    )
    
    args = parser.parse_args()
    
    try:
        print("VGG-T to NeuS2 Converter - Center of Attention")
        print("=" * 55)
        
        # Load VGG-T poses
        pose_data = load_vggt_poses(args.poses_file)
        extrinsics = pose_data['extrinsics']
        intrinsics = pose_data['intrinsics']
        
        # Calculate center of attention (where cameras are looking)
        center_of_attention, original_positions = calculate_center_of_attention(extrinsics)
        
        # Recenter poses around center of attention
        recentered_extrinsics = recenter_camera_poses(extrinsics, center_of_attention)
        
        # Fix camera orientations to point toward origin (unless skipped)
        if not args.skip_orientation_fix:
            fixed_extrinsics = fix_camera_orientations(recentered_extrinsics)
        else:
            print("Skipping camera orientation fix (as requested)")
            fixed_extrinsics = recentered_extrinsics
        
        # Validate camera orientations
        if args.validate_orientations:
            orientation_valid = validate_camera_orientations(fixed_extrinsics)
            if not orientation_valid:
                print("⚠️  Warning: Many cameras not pointing toward origin. Results may be suboptimal.")
        
        # Find matching image files
        image_files = find_matching_images(args.images_dir, len(extrinsics))
        
        # Generate NeuS2 transforms with InstantNGP-style scaling
        transforms_data = generate_neus2_transforms(fixed_extrinsics, intrinsics, image_files)
        
        # Save transforms.json
        with open(args.output, 'w') as f:
            json.dump(transforms_data, f, indent=2)
        
        print(f"\n✅ Successfully generated NeuS2 transforms!")
        print(f"   Output file: {args.output}")
        print(f"   Camera poses: {len(transforms_data['frames'])}")
        print(f"   Recentered around: [{center_of_attention[0]:.3f}, {center_of_attention[1]:.3f}, {center_of_attention[2]:.3f}]")
        
        print(f"\n🚀 Ready for NeuS2 training!")
        print(f"   Command: python run.py --name vggt_coa_test --scene . --n_steps 1000 --save_mesh")
        
        # Save conversion metadata
        metadata_file = args.output.replace('.json', '_metadata.json')
        metadata = {
            'source_poses_file': args.poses_file,
            'center_of_attention': center_of_attention.tolist(),
            'num_cameras': len(extrinsics),
            'conversion_method': 'center_of_attention',
            'images_dir': args.images_dir,
            'vggt_metadata': pose_data.get('metadata', {})
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Metadata saved: {metadata_file}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())