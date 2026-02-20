#!/usr/bin/env python3
"""
VGG-T Camera Pose Analyzer

This script analyzes VGG-T camera poses to understand the coordinate system
and validate assumptions before building the NeuS2 converter.

Key questions to answer:
1. Where is the world origin relative to the object?
2. Do cameras follow expected circular motion?
3. Are the 3 cameras properly vertically aligned?
4. What coordinate system conventions does VGG-T use?
"""

import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def load_poses(poses_file):
    """Load camera poses from JSON file."""
    with open(poses_file, 'r') as f:
        data = json.load(f)
    
    # Convert back to numpy arrays
    data['extrinsic'] = np.array(data['extrinsic'])
    data['intrinsic'] = np.array(data['intrinsic'])
    data['camera_positions'] = np.array(data['camera_positions'])
    
    return data

def analyze_coordinate_system(pose_data):
    """Analyze VGG-T coordinate system conventions."""
    print("="*60)
    print("VGG-T COORDINATE SYSTEM ANALYSIS")
    print("="*60)
    
    extrinsics = pose_data['extrinsic']
    positions = pose_data['camera_positions']
    num_frames = len(positions)
    
    print(f"Total camera poses: {num_frames}")
    print(f"Expected: 135 poses (45 per camera × 3 cameras)")
    
    # Basic statistics
    print(f"\nPosition ranges:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    
    centroid = positions.mean(axis=0)
    print(f"\nCamera centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    
    # Distance analysis
    distances_from_origin = np.linalg.norm(positions, axis=1)
    distances_from_centroid = np.linalg.norm(positions - centroid, axis=1)
    
    print(f"\nDistance from origin (0,0,0):")
    print(f"  Range: [{distances_from_origin.min():.3f}, {distances_from_origin.max():.3f}]")
    print(f"  Mean: {distances_from_origin.mean():.3f}")
    
    print(f"\nDistance from centroid:")
    print(f"  Range: [{distances_from_centroid.min():.3f}, {distances_from_centroid.max():.3f}]")
    print(f"  Mean: {distances_from_centroid.mean():.3f}")
    
    return {
        'positions': positions,
        'centroid': centroid,
        'distances_from_origin': distances_from_origin,
        'distances_from_centroid': distances_from_centroid
    }

def analyze_camera_grouping(positions, expected_cameras=3, frames_per_camera=45):
    """Analyze if cameras are grouped as expected (sequential organization)."""
    print("\n" + "="*60)
    print("CAMERA GROUPING ANALYSIS")
    print("="*60)
    
    num_frames = len(positions)
    
    if num_frames != expected_cameras * frames_per_camera:
        print(f"⚠️  WARNING: Expected {expected_cameras * frames_per_camera} frames, got {num_frames}")
    
    # Analyze grouping by checking sequential blocks
    for camera_id in range(expected_cameras):
        start_idx = camera_id * frames_per_camera
        end_idx = min(start_idx + frames_per_camera, num_frames)
        
        if start_idx >= num_frames:
            print(f"Camera {camera_id + 1}: No data (not enough frames)")
            continue
            
        camera_positions = positions[start_idx:end_idx]
        
        # Calculate statistics for this camera
        camera_centroid = camera_positions.mean(axis=0)
        height_range = camera_positions[:, 1].max() - camera_positions[:, 1].min()
        xy_range = np.sqrt((camera_positions[:, 0].max() - camera_positions[:, 0].min())**2 + 
                          (camera_positions[:, 2].max() - camera_positions[:, 2].min())**2)
        
        print(f"\nCamera {camera_id + 1} (frames {start_idx:03d}-{end_idx-1:03d}):")
        print(f"  Centroid: [{camera_centroid[0]:.3f}, {camera_centroid[1]:.3f}, {camera_centroid[2]:.3f}]")
        print(f"  Height variation: {height_range:.3f}")
        print(f"  XZ motion range: {xy_range:.3f}")
        
        # Check for circular motion
        if xy_range > 1.0:
            print(f"  ✓ Detected significant XZ motion (likely circular)")
        else:
            print(f"  ⚠️  Limited XZ motion (may not be circular)")
        
        if height_range < 0.5:
            print(f"  ✓ Consistent height (good for fixed camera)")
        else:
            print(f"  ⚠️  High height variation (unexpected for fixed camera)")

def analyze_camera_orientations(pose_data):
    """Analyze camera orientations to understand viewing directions."""
    print("\n" + "="*60)
    print("CAMERA ORIENTATION ANALYSIS")
    print("="*60)
    
    extrinsics = pose_data['extrinsic']
    positions = pose_data['camera_positions']
    centroid = positions.mean(axis=0)
    
    # Analyze first few cameras to understand orientation
    sample_indices = [0, 44, 89]  # First frame of each camera
    
    for i, idx in enumerate(sample_indices):
        if idx >= len(extrinsics):
            continue
            
        camera_matrix = extrinsics[idx]
        camera_pos = positions[idx]
        
        # Extract camera coordinate system
        right = camera_matrix[:3, 0]      # X-axis of camera
        up = camera_matrix[:3, 1]         # Y-axis of camera  
        forward = -camera_matrix[:3, 2]   # Camera looks down -Z axis
        
        # Calculate direction from camera to centroid
        to_centroid = centroid - camera_pos
        to_centroid_normalized = to_centroid / np.linalg.norm(to_centroid)
        
        # Calculate angle between camera forward and direction to centroid
        dot_product = np.dot(forward, to_centroid_normalized)
        angle_to_centroid = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
        
        print(f"\nCamera {i+1} (frame {idx:03d}):")
        print(f"  Position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
        print(f"  Forward direction: [{forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f}]")
        print(f"  To centroid: [{to_centroid_normalized[0]:.3f}, {to_centroid_normalized[1]:.3f}, {to_centroid_normalized[2]:.3f}]")
        print(f"  Angle to centroid: {angle_to_centroid:.1f}°")
        
        if angle_to_centroid < 10:
            print(f"  ✓ Camera pointing toward centroid")
        else:
            print(f"  ⚠️  Camera not pointing toward centroid")

def visualize_camera_positions(positions, save_path=None):
    """Create 3D visualization of camera positions."""
    print("\n" + "="*60)
    print("GENERATING 3D VISUALIZATION")
    print("="*60)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: All cameras together
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=20, alpha=0.6)
    
    # Mark centroid
    centroid = positions.mean(axis=0)
    ax1.scatter(*centroid, c='red', s=100, marker='*', label='Centroid')
    ax1.scatter(0, 0, 0, c='black', s=100, marker='o', label='Origin')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y') 
    ax1.set_zlabel('Z')
    ax1.set_title('All Camera Positions')
    ax1.legend()
    
    # Plot 2: Cameras by group (if we have 135 frames)
    ax2 = fig.add_subplot(132, projection='3d')
    
    if len(positions) == 135:  # Expected number
        colors = ['red', 'green', 'blue']
        for camera_id in range(3):
            start_idx = camera_id * 45
            end_idx = start_idx + 45
            camera_positions = positions[start_idx:end_idx]
            ax2.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                       c=colors[camera_id], s=30, alpha=0.7, label=f'Camera {camera_id+1}')
    else:
        ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=20, alpha=0.6)
    
    ax2.scatter(*centroid, c='red', s=100, marker='*', label='Centroid')
    ax2.scatter(0, 0, 0, c='black', s=100, marker='o', label='Origin')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Cameras by Group')
    ax2.legend()
    
    # Plot 3: Top-down view (X-Z plane)
    ax3 = fig.add_subplot(133)
    if len(positions) == 135:
        for camera_id in range(3):
            start_idx = camera_id * 45
            end_idx = start_idx + 45
            camera_positions = positions[start_idx:end_idx]
            ax3.scatter(camera_positions[:, 0], camera_positions[:, 2], 
                       c=colors[camera_id], s=30, alpha=0.7, label=f'Camera {camera_id+1}')
    else:
        ax3.scatter(positions[:, 0], positions[:, 2], c='blue', s=20, alpha=0.6)
    
    ax3.scatter(centroid[0], centroid[2], c='red', s=100, marker='*', label='Centroid')
    ax3.scatter(0, 0, c='black', s=100, marker='o', label='Origin')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Top View (X-Z plane)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()

def generate_recommendations(analysis):
    """Generate recommendations based on analysis."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR NEUS2 CONVERSION")
    print("="*60)
    
    centroid = analysis['centroid']
    distances = analysis['distances_from_centroid']
    
    print(f"Current world origin: (0, 0, 0)")
    print(f"Detected object center: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
    print(f"Translation needed: ({-centroid[0]:.3f}, {-centroid[1]:.3f}, {-centroid[2]:.3f})")
    
    print(f"\nCamera orbit radius: {distances.mean():.3f} units")
    
    if distances.std() < 0.1:
        print("✓ Cameras maintain consistent distance (good circular motion)")
    else:
        print("⚠️  Inconsistent camera distances (may affect quality)")
    
    print(f"\nNext steps:")
    print(f"1. The recentering approach should work well")
    print(f"2. Apply translation: new_pos = old_pos - ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
    print(f"3. Verify that cameras properly look at new origin (0,0,0)")

def main():
    parser = argparse.ArgumentParser(description="Analyze VGG-T camera poses")
    parser.add_argument("poses_file", help="JSON file containing VGG-T camera poses")
    parser.add_argument("--save_plot", help="Save visualization plot to file")
    
    args = parser.parse_args()
    
    try:
        # Load poses
        print(f"Loading camera poses from: {args.poses_file}")
        pose_data = load_poses(args.poses_file)
        
        # Run analysis
        analysis = analyze_coordinate_system(pose_data)
        analyze_camera_grouping(pose_data['camera_positions'])
        analyze_camera_orientations(pose_data)
        
        # Generate visualization
        visualize_camera_positions(
            pose_data['camera_positions'], 
            save_path=args.save_plot
        )
        
        # Generate recommendations
        generate_recommendations(analysis)
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()