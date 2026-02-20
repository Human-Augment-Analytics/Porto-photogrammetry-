#!/usr/bin/env python3
"""
VGG-T to NeuS2 Converter - Object-Centered Version
Uses VGG-T world points to find and center on the actual object of interest
"""

import numpy as np
import json
import argparse
import os
from pathlib import Path
import cv2
import pickle

def load_vggt_predictions(predictions_path):
    """Load VGG-T predictions from .pkl file"""
    print(f"📁 Loading VGG-T predictions from: {predictions_path}")
    
    with open(predictions_path, 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"📊 Loaded predictions with keys: {list(predictions.keys())}")
    return predictions

def find_object_center_from_world_points(predictions, confidence_threshold=0.1):
    """
    Find the object center using VGG-T's world points.
    This is more accurate than camera ray intersections for object-centric scenes.
    """
    print("🎯 Finding object center from VGG-T world points...")
    
    # Try different world point sources
    world_points = None
    confidence = None
    
    if 'world_points' in predictions:
        world_points = predictions['world_points']
        print(f"   Using world_points: {world_points.shape}")
        
        if 'world_points_conf' in predictions:
            confidence = predictions['world_points_conf']
            print(f"   Found confidence: {confidence.shape}")
    
    elif 'world_points_from_depth' in predictions:
        world_points = predictions['world_points_from_depth']
        print(f"   Using world_points_from_depth: {world_points.shape}")
        
        if 'depth_conf' in predictions:
            confidence = predictions['depth_conf']
            print(f"   Found depth confidence: {confidence.shape}")
    
    if world_points is None:
        print("   ❌ No world points found, cannot determine object center")
        return None, None
    
    # Flatten world points to (N, 3)
    original_shape = world_points.shape
    if len(original_shape) == 4:  # (S, H, W, 3)
        points_flat = world_points.reshape(-1, 3)
        if confidence is not None:
            conf_flat = confidence.reshape(-1)
    elif len(original_shape) == 3:  # (S*H*W, 3) already flat
        points_flat = world_points
        conf_flat = confidence.reshape(-1) if confidence is not None else None
    else:
        print(f"   ❌ Unexpected world points shape: {original_shape}")
        return None, None
    
    print(f"   Total points: {len(points_flat)}")
    
    # Filter out invalid points
    valid_mask = np.all(np.isfinite(points_flat), axis=1)
    valid_mask &= ~np.all(points_flat == 0, axis=1)  # Remove zero points
    
    # Apply confidence threshold if available
    if conf_flat is not None:
        valid_mask &= (conf_flat > confidence_threshold)
        print(f"   Points above confidence {confidence_threshold}: {np.sum(valid_mask)}")
    
    if np.sum(valid_mask) < 100:
        print(f"   ⚠️ Only {np.sum(valid_mask)} valid points, using lower threshold")
        # Relax criteria if we don't have enough points
        valid_mask = np.all(np.isfinite(points_flat), axis=1)
        valid_mask &= ~np.all(points_flat == 0, axis=1)
    
    valid_points = points_flat[valid_mask]
    
    if len(valid_points) == 0:
        print("   ❌ No valid points found")
        return None, None
    
    print(f"   Valid points: {len(valid_points)}")
    
    # Remove outliers using percentile filtering
    for axis in range(3):
        axis_points = valid_points[:, axis]
        q1, q99 = np.percentile(axis_points, [1, 99])
        outlier_mask = (axis_points >= q1) & (axis_points <= q99)
        valid_points = valid_points[outlier_mask]
    
    print(f"   After outlier removal: {len(valid_points)}")
    
    if len(valid_points) < 50:
        print("   ⚠️ Very few points after filtering, results may be unreliable")
    
    # Compute robust center
    object_center = np.median(valid_points, axis=0)
    
    # Compute object extent for scale estimation
    distances = np.linalg.norm(valid_points - object_center, axis=1)
    object_extent = np.percentile(distances, 90)
    
    print(f"   📍 Object center: {object_center}")
    print(f"   📏 Object extent (90th percentile): {object_extent:.3f}")
    
    return object_center, object_extent

def extract_camera_params_from_predictions(predictions):
    """Extract camera parameters from VGG-T predictions"""
    extrinsics = predictions['extrinsic']  
    intrinsics = predictions['intrinsic']  
    
    S = extrinsics.shape[0]
    
    # Convert extrinsics to 4x4 if needed
    if extrinsics.shape[-2:] == (3, 4):
        extrinsics_4x4 = np.zeros((S, 4, 4))
        extrinsics_4x4[:, :3, :] = extrinsics
        extrinsics_4x4[:, 3, 3] = 1.0
    else:
        extrinsics_4x4 = extrinsics.copy()
    
    # Extract focal lengths and principal points
    focal_lengths = np.zeros((S, 2))
    principal_points = np.zeros((S, 2))
    
    for i in range(S):
        K = intrinsics[i, :3, :3] if intrinsics.shape[-2:] != (3, 3) else intrinsics[i]
        focal_lengths[i] = [K[0, 0], K[1, 1]]  # fx, fy
        principal_points[i] = [K[0, 2], K[1, 2]]  # cx, cy
    
    return extrinsics_4x4, intrinsics, focal_lengths, principal_points

def convert_vggt_to_neus2_object_centered(predictions_path, images_dir, output_dir="output", 
                                        aabb_scale=128, apply_coordinate_transform=True):
    """
    Convert VGG-T predictions to NeuS2 format, centering on the object of interest
    """
    
    print("🚀 VGG-T to NeuS2 Converter - Object-Centered Version")
    print("="*60)
    
    # Load VGG-T predictions
    predictions = load_vggt_predictions(predictions_path)
    
    # Find object center using world points
    object_center, object_extent = find_object_center_from_world_points(predictions)
    
    # Extract camera parameters
    extrinsics_4x4, intrinsics, focal_lengths, principal_points = extract_camera_params_from_predictions(predictions)
    S = extrinsics_4x4.shape[0]
    
    # Get image dimensions
    if 'images' in predictions:
        h, w = predictions['images'].shape[1], predictions['images'].shape[2]
    else:
        h, w = 480, 640  # Default
    
    print(f"📸 Processing {S} frames of {w}×{h}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Process camera matrices
    c2w_matrices = []
    
    for i in range(S):
        # Start with VGG-T camera-to-world matrix
        c2w = extrinsics_4x4[i].copy()
        
        # 1. Translate to center object at origin (if we found object center)
        if object_center is not None:
            c2w[:3, 3] -= object_center
        
        # 2. Apply coordinate system transformation for NeRF/NeuS2
        if apply_coordinate_transform:
            # Standard NeRF coordinate transform
            c2w[0:3, 2] *= -1  # flip z axis
            c2w[0:3, 1] *= -1  # flip y axis  
            c2w = c2w[[1, 0, 2, 3], :]  # reorder axes
            c2w[2, :] *= -1  # flip world upside down
        
        c2w_matrices.append(c2w)
    
    # Determine appropriate scale
    camera_positions = np.array([c2w[:3, 3] for c2w in c2w_matrices])
    avg_camera_distance = np.mean(np.linalg.norm(camera_positions, axis=1))
    
    # NeRF/InstantNGP expects cameras ~1 unit from origin after scaling
    # Default scale is 0.33, so we adjust based on current scale
    target_distance = 1.0  # Target average distance from origin
    scene_scale = target_distance / avg_camera_distance if avg_camera_distance > 0 else 1.0
    
    # Apply scaling to all camera positions
    for i in range(S):
        c2w_matrices[i][:3, 3] *= scene_scale
    
    print(f"📐 Scene scaling:")
    print(f"   Original avg camera distance: {avg_camera_distance:.3f}")
    print(f"   Scene scale factor: {scene_scale:.3f}")
    print(f"   Final avg camera distance: {avg_camera_distance * scene_scale:.3f}")
    if object_center is not None:
        print(f"   Object centered at: {object_center}")
        if object_extent is not None:
            print(f"   Object extent: {object_extent:.3f} -> {object_extent * scene_scale:.3f}")
    
    # Handle images and create frames
    image_files = [f"{i:06d}.png" for i in range(S)]
    frames = []
    
    for i in range(S):
        # Save image if available in predictions
        if 'images' in predictions:
            image_array = predictions['images'][i]
            target_image = os.path.join(images_output_dir, image_files[i])
            
            if not os.path.exists(target_image):
                try:
                    from PIL import Image
                    if image_array.dtype != np.uint8:
                        image_array = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
                    pil_image = Image.fromarray(image_array)
                    pil_image.save(target_image)
                except ImportError:
                    cv2.imwrite(target_image, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        
        # Create frame
        frame = {
            "file_path": f"./images/{image_files[i]}",
            "sharpness": 1.0,  # Default sharpness
            "transform_matrix": c2w_matrices[i].tolist()
        }
        frames.append(frame)
    
    # Build transform.json
    fx, fy = focal_lengths[0]
    cx, cy = principal_points[0]
    camera_angle_x = 2 * np.arctan(w / (2 * fx))
    camera_angle_y = 2 * np.arctan(h / (2 * fy))
    
    # For object-centered scenes, we've already scaled appropriately
    # Set scale to 1.0 since we've done our own scaling
    transform_data = {
        "camera_angle_x": float(camera_angle_x),
        "camera_angle_y": float(camera_angle_y), 
        "fl_x": float(fx),
        "fl_y": float(fy),
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "p1": 0.0, "p2": 0.0,
        "is_fisheye": False,
        "cx": float(cx),
        "cy": float(cy),
        "w": w, "h": h,
        "aabb_scale": aabb_scale,
        "scale": 1.0,  # We've done our own scaling
        "offset": [0.0, 0.0, 0.0],  # Object already centered
        "frames": frames
    }
    
    # Save files
    transform_path = os.path.join(output_dir, "transforms.json")
    with open(transform_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    
    # Create debug info
    final_camera_positions = np.array([c2w[:3, 3] for c2w in c2w_matrices])
    
    debug_info = {
        "conversion_type": "object_centered_from_world_points",
        "source_file": predictions_path,
        "num_frames": S,
        "image_dimensions": [w, h],
        "object_centering": {
            "object_center_found": object_center is not None,
            "object_center": object_center.tolist() if object_center is not None else None,
            "object_extent": float(object_extent) if object_extent is not None else None,
            "vggt_coordinate_system": "first_camera_at_origin",
            "converted_to": "object_at_origin"
        },
        "scene_scaling": {
            "original_avg_distance": float(avg_camera_distance),
            "scene_scale_factor": float(scene_scale),
            "final_avg_distance": float(np.mean(np.linalg.norm(final_camera_positions, axis=1))),
            "target_distance": target_distance
        },
        "quality_metrics": {
            "camera_motion_adequate": bool(np.std(np.linalg.norm(final_camera_positions, axis=1)) > 0.1),
            "object_properly_centered": object_center is not None,
            "scene_scale_appropriate": bool(0.5 < np.mean(np.linalg.norm(final_camera_positions, axis=1)) < 2.0),
            "ready_for_neus2": object_center is not None
        }
    }
    
    debug_path = os.path.join(output_dir, "object_centered_debug.json")
    with open(debug_path, 'w') as f:
        json.dump(debug_info, f, indent=2)
    
    print(f"\n✅ Object-Centered Conversion Complete!")
    print(f"📁 Output: {output_dir}")
    print(f"📋 Transform: {transform_path}")
    print(f"🐛 Debug: {debug_path}")
    
    if debug_info["quality_metrics"]["ready_for_neus2"]:
        print(f"\n🎯 Object properly centered - ready for NeuS2!")
    else:
        print(f"\n⚠️ Object centering may have issues - check debug file")
    
    return debug_info

def main():
    parser = argparse.ArgumentParser(description="Convert VGG-T predictions to object-centered NeuS2 format")
    parser.add_argument("predictions_path", help="Path to VGG-T predictions .pkl file")
    parser.add_argument("output_dir", help="Output directory for NeuS2 data")
    parser.add_argument("--images_dir", help="Source directory containing original images (optional)")
    parser.add_argument("--aabb_scale", type=int, default=128, choices=[1,2,4,8,16,32,64,128],
                       help="Scene scale factor (default 128 for natural scenes)")
    parser.add_argument("--confidence_threshold", type=float, default=0.1,
                       help="Confidence threshold for world points")
    
    args = parser.parse_args()
    
    try:
        debug_info = convert_vggt_to_neus2_object_centered(
            args.predictions_path,
            args.images_dir,
            args.output_dir,
            args.aabb_scale
        )
        
        print(f"\n🚀 Ready for NeuS2 training:")
        print(f"   python train.py --conf confs/base.conf --data_dir {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())