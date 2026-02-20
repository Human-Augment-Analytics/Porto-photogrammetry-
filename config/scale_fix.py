#!/usr/bin/env python3
"""
Quick scale fix - manually adjust existing transform.json
"""

import json
import numpy as np
import argparse

def fix_transform_scale(input_path, output_path, scale_factor=0.3):
    """
    Fix the scale of an existing transform.json by making the scene smaller
    relative to the cameras (or equivalently, cameras farther from object)
    """
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"Original scale parameters:")
    print(f"  aabb_scale: {data.get('aabb_scale', 'not set')}")
    print(f"  scale: {data.get('scale', 'not set')}")
    print(f"  offset: {data.get('offset', 'not set')}")
    
    # Calculate current camera distances
    camera_positions = []
    for frame in data['frames']:
        transform = np.array(frame['transform_matrix'])
        pos = transform[:3, 3]
        camera_positions.append(pos)
    
    camera_positions = np.array(camera_positions)
    current_avg_distance = np.mean(np.linalg.norm(camera_positions, axis=1))
    print(f"  Current avg camera distance: {current_avg_distance:.3f}")
    
    # Apply scale factor to all camera positions
    for frame in data['frames']:
        transform = np.array(frame['transform_matrix'])
        transform[:3, 3] *= scale_factor  # Make cameras farther
        frame['transform_matrix'] = transform.tolist()
    
    # Update global scale parameters to match working NeuS2
    data['aabb_scale'] = 1.0
    data['scale'] = 0.5
    data['offset'] = [0.5, 0.5, 0.5]
    
    # Calculate new camera distances
    new_camera_positions = []
    for frame in data['frames']:
        transform = np.array(frame['transform_matrix'])
        pos = transform[:3, 3]
        new_camera_positions.append(pos)
    
    new_camera_positions = np.array(new_camera_positions)
    new_avg_distance = np.mean(np.linalg.norm(new_camera_positions, axis=1))
    
    print(f"Fixed scale parameters:")
    print(f"  aabb_scale: {data['aabb_scale']}")
    print(f"  scale: {data['scale']}")
    print(f"  offset: {data['offset']}")
    print(f"  New avg camera distance: {new_avg_distance:.3f}")
    print(f"  Scale factor applied: {scale_factor}")
    
    # Save fixed version
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved fixed transform.json to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Fix scale in existing transform.json")
    parser.add_argument("input", help="Input transform.json file")
    parser.add_argument("output", help="Output transform.json file")
    parser.add_argument("--scale", type=float, default=0.3, 
                       help="Scale factor to apply (default 0.3 to match working example)")
    
    args = parser.parse_args()
    
    fix_transform_scale(args.input, args.output, args.scale)

if __name__ == "__main__":
    main()