# Main script to run the VGGT model pipeline
# Eventually will be adapted to pipe outputs to NeuS2 for mesh generation
# Adapted from demo_gradio.py with modifications for direct execution
# Author: Clinton Kunhardt

import os
import torch
import numpy as np
import sys
import glob
import time
import gc

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)

def run_model(target_dir, model) -> dict:
    """
    EXACT BYTE-FOR-BYTE copy of demo_gradio.py run_model function
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions

def parse_arguments():
    """Parse command line arguments for input and output directories"""
    parser = argparse.ArgumentParser(
        description="Run VGGT photogrammetry pipeline on a directory of images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing an 'images' subdirectory with photos"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated GLB file (defaults to input_dir)"
    )
    
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=50.0,
        help="Confidence threshold for GLB generation"
    )
    
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="Depthmap and Camera Branch",
        choices=["Depthmap and Camera Branch", "Points Only", "Cameras Only"],
        help="Prediction mode for GLB generation"
    )
    
    return parser.parse_args()

def validate_input_directory(input_dir):
    """Validate that input directory exists and contains images subdirectory"""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    images_dir = os.path.join(input_dir, "images")
    if not os.path.exists(images_dir):
        raise ValueError(f"Input directory must contain an 'images' subdirectory: {images_dir}")
    
    # Check if images directory has any files
    image_files = glob.glob(os.path.join(images_dir, "*"))
    if len(image_files) == 0:
        raise ValueError(f"No images found in: {images_dir}")
    
    print(f"Found {len(image_files)} files in images directory")
    return True

def main():
    """Main function with command line argument handling"""
    args = parse_arguments()
    
    # Validate input directory
    try:
        validate_input_directory(args.input_dir)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.conf_thres}")
    print(f"Prediction mode: {args.prediction_mode}")
    
    try:
        start_time = time.time()
        predictions = run_model(args.input_dir, model)  # Use global model
        total_time = time.time() - start_time
        
        print(f"SUCCESS! Processing took {total_time:.2f} seconds")
        
        # Generate output filename with timestamp
        timestamp = int(time.time())
        glb_filename = f"vggt_output_{timestamp}.glb"
        glb_path = os.path.join(output_dir, glb_filename)
        
        print(f"Saving GLB to {glb_path}")
        
        scene = predictions_to_glb(
            predictions, 
            conf_thres=args.conf_thres,
            target_dir=args.input_dir,  # Still use input_dir for any image references
            prediction_mode=args.prediction_mode
        )
        scene.export(glb_path)
        
        print(f"Successfully saved to: {glb_path}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)