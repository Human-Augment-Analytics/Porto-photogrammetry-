#!/usr/bin/env python3
"""
Quick fix for camera orientations in transforms.json

This script applies simple transformations to fix camera orientations.
"""

import json
import numpy as np
import argparse

def apply_camera_fixes(transforms_file, output_file):
    """Apply various camera orientation fixes."""
    
    print(f"Loading transforms from: {transforms_file}")
    with open(transforms_file, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    print(f"Found {len(frames)} camera frames")
    
    print("\nTrying different camera orientation fixes...")
    
    # Fix 1: Flip Z-axis (camera forward direction)
    print("Fix 1: Flipping camera Z-axis (forward direction)")
    for frame in frames:
        transform = np.array(frame['transform_matrix'])
        transform[:3, 2] *= -1  # Flip Z-axis
        frame['transform_matrix'] = transform.tolist()
    
    # Save fixed version
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Applied Z-axis flip fix")
    print(f"   Saved to: {output_file}")
    print(f"   Test this in NeuS2 debug visualizer")
    
    # Also create alternative fixes for testing
    alt_fixes = [
        ("flip_x", lambda t: np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ t),
        ("flip_y", lambda t: np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ t),
        ("flip_xy", lambda t: np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ t),
        ("flip_xz", lambda t: np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ t),
        ("flip_yz", lambda t: np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ t),
    ]
    
    for fix_name, fix_func in alt_fixes:
        alt_data = json.loads(json.dumps(data))  # Deep copy
        
        for frame in alt_data['frames']:
            transform = np.array(frame['transform_matrix'])
            fixed_transform = fix_func(transform)
            frame['transform_matrix'] = fixed_transform.tolist()
        
        alt_file = output_file.replace('.json', f'_{fix_name}.json')
        with open(alt_file, 'w') as f:
            json.dump(alt_data, f, indent=2)
        
        print(f"   Alternative fix '{fix_name}' saved to: {alt_file}")
    
    print(f"\n🎯 Try these files in order until cameras point inward:")
    print(f"   1. {output_file} (Z-axis flip)")
    for fix_name, _ in alt_fixes:
        alt_file = output_file.replace('.json', f'_{fix_name}.json')
        print(f"   2. {alt_file}")

def main():
    parser = argparse.ArgumentParser(description="Quick fix for camera orientations")
    parser.add_argument("input_file", help="Input transforms.json file")
    parser.add_argument("--output", default="transforms_fixed_simple.json", help="Output file")
    
    args = parser.parse_args()
    
    apply_camera_fixes(args.input_file, args.output)

if __name__ == "__main__":
    main()