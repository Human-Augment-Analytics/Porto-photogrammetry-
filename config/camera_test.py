#!/usr/bin/env python3
"""
Test script to verify camera positioning logic
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

def create_camera_positions(distance_from_center=2.0, camera_spacing=0.5):
    """Create camera positions for line array setup"""
    camera_configs = []
    
    # Camera positions in line array (stacked vertically)
    base_positions = [
        np.array([distance_from_center, -camera_spacing, 0, 1]),  # Camera 1 (bottom)
        np.array([distance_from_center, 0, 0, 1]),                # Camera 2 (middle)  
        np.array([distance_from_center, camera_spacing, 0, 1])     # Camera 3 (top)
    ]
    
    for i, pos in enumerate(base_positions):
        camera_pos = pos[:3]
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
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
        transform[2, :3] = -forward
        transform[:3, 3] = camera_pos
        
        camera_configs.append({
            'position': camera_pos,
            'transform': transform,
            'camera_id': i
        })
    
    return camera_configs

def create_rotation_matrix_y(angle_degrees):
    """Create rotation matrix for Y-axis rotation (turntable)"""
    theta = np.radians(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    return np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ])

def test_scenario_1_wrong():
    """Wrong approach: rotating cameras instead of object"""
    print("=== SCENARIO 1: WRONG (Rotating Cameras) ===")
    
    camera_configs = create_camera_positions()
    positions = []
    
    for rotation_step in range(10):  # Test first 10 rotations
        rotation_angle = rotation_step * 8
        rotation_matrix = create_rotation_matrix_y(rotation_angle)
        
        for cam_config in camera_configs:
            # WRONG: This rotates the camera positions
            rotated_transform = np.dot(rotation_matrix, cam_config['transform'])
            pos = rotated_transform[:3, 3]
            positions.append(pos)
    
    positions = np.array(positions)
    print(f"Camera positions range:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    print(f"This creates an elliptical pattern - WRONG!\n")
    
    return positions

def test_scenario_2_fixed():
    """Correct approach: cameras stay fixed"""
    print("=== SCENARIO 2: CORRECT (Fixed Cameras) ===")
    
    camera_configs = create_camera_positions()
    positions = []
    
    for rotation_step in range(10):  # Test first 10 rotations
        for cam_config in camera_configs:
            # CORRECT: Camera positions never change
            pos = cam_config['transform'][:3, 3]
            positions.append(pos)
    
    positions = np.array(positions)
    print(f"Camera positions range:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    print(f"Cameras stay in fixed positions - CORRECT!\n")
    
    return positions

def test_scenario_3_inverse():
    """Correct approach: inverse rotation for relative poses"""
    print("=== SCENARIO 3: CORRECT (Inverse Rotation for Relative Poses) ===")
    
    camera_configs = create_camera_positions()
    positions = []
    
    for rotation_step in range(10):  # Test first 10 rotations
        rotation_angle = rotation_step * 8
        # Inverse rotation: if object rotates +8°, cameras appear to rotate -8° relative to object
        inverse_rotation_matrix = create_rotation_matrix_y(-rotation_angle)
        
        for cam_config in camera_configs:
            # Apply inverse rotation to get camera pose relative to rotated object
            relative_transform = np.dot(inverse_rotation_matrix, cam_config['transform'])
            pos = relative_transform[:3, 3]
            positions.append(pos)
    
    positions = np.array(positions)
    print(f"Camera-relative-to-object positions range:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    print(f"This shows how cameras appear to move relative to the object - CORRECT for NeRF!\n")
    
    return positions

def test_scenario_4_same_direction():
    """Alternative: same direction rotation"""
    print("=== SCENARIO 4: ALTERNATIVE (Same Direction Rotation) ===")
    
    camera_configs = create_camera_positions()
    positions = []
    
    for rotation_step in range(10):  # Test first 10 rotations
        rotation_angle = rotation_step * 8
        # Same direction: cameras orbit in same direction as object rotation
        rotation_matrix = create_rotation_matrix_y(rotation_angle)
        
        for cam_config in camera_configs:
            # Apply same direction rotation
            relative_transform = np.dot(rotation_matrix, cam_config['transform'])
            pos = relative_transform[:3, 3]
            positions.append(pos)
    
    positions = np.array(positions)
    print(f"Camera positions with same-direction rotation:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    print(f"Alternative approach - cameras orbit in same direction as object rotation\n")
    
    return positions

def visualize_scenarios():
    """Visualize all scenarios"""
    fig = plt.figure(figsize=(20, 5))
    
    # Scenario 1: Wrong
    ax1 = fig.add_subplot(141, projection='3d')
    pos1 = test_scenario_1_wrong()
    ax1.scatter(pos1[:, 0], pos1[:, 1], pos1[:, 2], c='red', alpha=0.6)
    ax1.scatter(0, 0, 0, c='green', s=100, marker='s')
    ax1.set_title('WRONG: Rotating Cameras')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Scenario 2: Fixed cameras
    ax2 = fig.add_subplot(142, projection='3d')
    pos2 = test_scenario_2_fixed()
    ax2.scatter(pos2[:, 0], pos2[:, 1], pos2[:, 2], c='blue', alpha=0.6)
    ax2.scatter(0, 0, 0, c='green', s=100, marker='s')
    ax2.set_title('REALITY: Fixed Cameras')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Scenario 3: Relative poses (inverse)
    ax3 = fig.add_subplot(143, projection='3d')
    pos3 = test_scenario_3_inverse()
    ax3.scatter(pos3[:, 0], pos3[:, 1], pos3[:, 2], c='purple', alpha=0.6)
    ax3.scatter(0, 0, 0, c='green', s=100, marker='s')
    ax3.set_title('OPTION A: Inverse Rotation')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # Scenario 4: Same direction
    ax4 = fig.add_subplot(144, projection='3d')
    pos4 = test_scenario_4_same_direction()
    ax4.scatter(pos4[:, 0], pos4[:, 1], pos4[:, 2], c='orange', alpha=0.6)
    ax4.scatter(0, 0, 0, c='green', s=100, marker='s')
    ax4.set_title('OPTION B: Same Direction')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('camera_scenarios.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("Camera Position Logic Test")
    print("=" * 50)
    
    wrong_positions = test_scenario_1_wrong()
    fixed_positions = test_scenario_2_fixed()
    relative_positions = test_scenario_3_inverse()
    same_direction_positions = test_scenario_4_same_direction()
    
    print("DIAGNOSIS:")
    print("Your original elliptical pattern = Scenario 1 (wrong)")
    print("Physical reality = Scenario 2 (3 fixed points)")
    print("For NeRF/NeuS2, we need either Option A or Option B")
    print("Both create circular orbits, just in different directions")
    print("Try both and see which gives better NeuS2 results!")
    
    visualize_scenarios()

    # Add missing function call
    test_scenario_4_same_direction()

if __name__ == "__main__":
    main()