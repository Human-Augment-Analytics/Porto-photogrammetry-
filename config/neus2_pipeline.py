#!/usr/bin/env python3
"""
Complete NeuS2 Workflow Script
Automates the entire process from raw images to NeuS2-ready dataset
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n-> {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def setup_directories(base_dir):
    """Set up the directory structure"""
    print(f"\nSetting up directories in {base_dir}")
    
    dirs_to_create = [
        "images_resized",
        "masks_resized", 
        "data"
    ]
    
    for dirname in dirs_to_create:
        full_path = os.path.join(base_dir, dirname)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created: {full_path}")

def check_prerequisites():
    """Check if required scripts exist"""
    required_scripts = [
        "image_resizer.py",
        "neus2_fixed_generator.py",
        "neus2_diagnostics.py"
    ]
    
    missing = []
    for script in required_scripts:
        if not os.path.exists(script):
            missing.append(script)
    
    if missing:
        print(f"Missing required scripts: {missing}")
        print("Please ensure all scripts are in the current directory")
        return False
    
    print("All required scripts found")
    return True

def main():
    parser = argparse.ArgumentParser(description="Complete NeuS2 workflow")
    
    # Input directories
    parser.add_argument("--input_images", type=str, required=True,
                       help="Directory containing original high-res images")
    parser.add_argument("--input_masks", type=str, default=None,
                       help="Directory containing original masks (optional)")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="neus2_data",
                       help="Output directory for NeuS2 dataset")
    
    # Resize options
    parser.add_argument("--size_preset", type=str, choices=['test', 'medium', 'high'], 
                       default='medium', help="Image size preset")
    parser.add_argument("--custom_width", type=int, default=None,
                       help="Custom width (overrides preset)")
    parser.add_argument("--custom_height", type=int, default=None,
                       help="Custom height (overrides preset)")
    
    # Camera setup
    parser.add_argument("--rotation_steps", type=int, default=45,
                       help="Number of rotation steps")
    parser.add_argument("--rotation_degrees", type=float, default=8.0,
                       help="Degrees per rotation step")
    parser.add_argument("--camera_distance", type=float, default=2.0,
                       help="Camera distance from center")
    parser.add_argument("--camera_spacing", type=float, default=0.5,
                       help="Vertical spacing between cameras")
    parser.add_argument("--aabb_scale", type=int, default=4,
                       help="AABB scale for NeuS2")
    
    # Workflow options
    parser.add_argument("--skip_resize", action="store_true",
                       help="Skip image resizing step")
    parser.add_argument("--skip_diagnostics", action="store_true",
                       help="Skip diagnostics step")
    
    args = parser.parse_args()
    
    print("Starting NeuS2 Complete Workflow")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Validate input directories
    if not os.path.exists(args.input_images):
        print(f"Input images directory not found: {args.input_images}")
        sys.exit(1)
    
    if args.input_masks and not os.path.exists(args.input_masks):
        print(f"Input masks directory not found: {args.input_masks}")
        sys.exit(1)
    
    # Create output directory structure
    setup_directories(args.output_dir)
    
    # Step 1: Resize images
    if not args.skip_resize:
        resize_cmd = [
            "python", "image_resizer.py",
            "--input_dir", args.input_images,
            "--output_dir", os.path.join(args.output_dir, "images_resized")
        ]
        
        # Add size parameters
        if args.custom_width and args.custom_height:
            resize_cmd.extend(["--width", str(args.custom_width)])
            resize_cmd.extend(["--height", str(args.custom_height)])
        else:
            resize_cmd.extend(["--preset", args.size_preset])
        
        # Add mask resizing if masks provided
        if args.input_masks:
            resize_cmd.extend([
                "--mask_input_dir", args.input_masks,
                "--mask_output_dir", os.path.join(args.output_dir, "masks_resized")
            ])
        
        if not run_command(resize_cmd, "Resizing images"):
            print("Image resizing failed")
            sys.exit(1)
    
    # Step 2: Generate transforms.json
    transform_cmd = [
        "python", "neus2_fixed_generator.py",
        "--image_dir", os.path.join(args.output_dir, "images_resized"),
        "--output", os.path.join(args.output_dir, "data", "transforms.json"),
        "--rotation_steps", str(args.rotation_steps),
        "--rotation_degrees", str(args.rotation_degrees),
        "--distance", str(args.camera_distance),
        "--spacing", str(args.camera_spacing),
        "--aabb_scale", str(args.aabb_scale)
    ]
    
    # Add mask directory if available
    if args.input_masks:
        transform_cmd.extend([
            "--mask_dir", os.path.join(args.output_dir, "masks_resized")
        ])
    
    if not run_command(transform_cmd, "Generating transforms.json"):
        print("Transform generation failed") 
        sys.exit(1)
    
    # Step 3: Copy images to final data directory
    print("\nSetting up final data structure")
    
    # Create symlinks or copy files to data directory
    data_images_dir = os.path.join(args.output_dir, "data", "images")
    resized_images_dir = os.path.join(args.output_dir, "images_resized")
    
    if os.path.exists(data_images_dir):
        import shutil
        shutil.rmtree(data_images_dir)
    
    # Create symlink for faster setup (or copy on Windows)
    try:
        os.symlink(os.path.abspath(resized_images_dir), data_images_dir)
        print(f"Created symlink: {data_images_dir} -> {resized_images_dir}")
    except OSError:
        # Fallback to copying (Windows or no symlink permissions)
        import shutil
        shutil.copytree(resized_images_dir, data_images_dir)
        print(f"Copied images to: {data_images_dir}")
    
    # Handle masks if they exist
    if args.input_masks:
        data_masks_dir = os.path.join(args.output_dir, "data", "masks")
        resized_masks_dir = os.path.join(args.output_dir, "masks_resized")
        
        if os.path.exists(data_masks_dir):
            import shutil
            shutil.rmtree(data_masks_dir)
        
        try:
            os.symlink(os.path.abspath(resized_masks_dir), data_masks_dir)
            print(f"Created symlink: {data_masks_dir} -> {resized_masks_dir}")
        except OSError:
            import shutil
            shutil.copytree(resized_masks_dir, data_masks_dir)
            print(f"Copied masks to: {data_masks_dir}")
    
    # Step 4: Run diagnostics
    if not args.skip_diagnostics:
        diag_cmd = [
            "python", "neus2_diagnostics.py",
            "--data_dir", os.path.join(args.output_dir, "data"),
            "--create_viz"
        ]
        
        run_command(diag_cmd, "Running diagnostics")
    
    # Step 5: Create training script
    data_dir = os.path.join(args.output_dir, "data")
    training_script = os.path.join(args.output_dir, "train.sh")
    
    script_content = f"""#!/bin/bash

# NeuS2 Training Script
# Generated by neus2_workflow.py

cd "{os.path.abspath(data_dir)}"

echo "Starting NeuS2 training..."
echo "Data directory: {data_dir}"

# Quick test (1000 steps)
echo "Running quick test..."
python ../../../run.py \\
    --name test_quick \\
    --scene . \\
    --n_steps 1000 \\
    --save_mesh \\
    --marching_cubes_res 128

# Full training (5000 steps)
echo "Running full training..."
python ../../../run.py \\
    --name test_full \\
    --scene . \\
    --n_steps 5000 \\
    --save_mesh \\
    --marching_cubes_res 256

echo "Training complete!"
echo "Results saved to: output/test_full/"
"""
    
    with open(training_script, "w") as f:
        f.write(script_content)
    
    os.chmod(training_script, 0o755)
    
    # Final summary
    print("\nWorkflow Complete!")
    print("=" * 50)
    print(f"Dataset ready in: {os.path.join(args.output_dir, 'data')}")
    print(f"Images: {os.path.join(args.output_dir, 'data', 'images')}")
    if args.input_masks:
        print(f"Masks: {os.path.join(args.output_dir, 'data', 'masks')}")
    print(f"Transforms: {os.path.join(args.output_dir, 'data', 'transforms.json')}")
    print(f"Training script: {training_script}")
    
    print(f"\nNext steps:")
    print(f"1. Review diagnostics output above")
    print(f"2. Run training: {training_script}")
    print(f"3. Or run manually:")
    print(f"   cd {os.path.join(args.output_dir, 'data')}")
    print(f"   python ../../../run.py --name test --scene . --n_steps 1000 --save_mesh")

if __name__ == "__main__":
    main()