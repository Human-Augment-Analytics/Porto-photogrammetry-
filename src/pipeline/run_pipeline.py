# Main script to run the VGGT model pipeline
# Now saves predictions for NeuS2 conversion
# Adapted from demo_gradio.py with modifications for direct execution
# Author: Clinton Kunhardt

import os
import torch
import numpy as np
import sys
import glob
import time
import gc
import pickle
import argparse  # Added missing import

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

def save_predictions_for_neus2(predictions, images_raw, output_dir, input_dir):
    """
    Save VGG-T predictions in format compatible with NeuS2 converter.
    
    Args:
        predictions: Dictionary of VGG-T predictions
        images_raw: Raw preprocessed images tensor (before normalization)
        output_dir: Directory to save predictions
        input_dir: Original input directory (for reference)
    """
    print("Preparing predictions for NeuS2 conversion...")
    
    # Create the predictions dictionary in the format expected by the converter
    neus2_predictions = {}
    
    # 1. World points - use both pointmap and depth-based points
    if "world_points" in predictions:
        neus2_predictions["world_points"] = predictions["world_points"]
        print(f"Added world_points: {predictions['world_points'].shape}")
    
    if "world_points_from_depth" in predictions:
        neus2_predictions["world_points_from_depth"] = predictions["world_points_from_depth"]
        print(f"Added world_points_from_depth: {predictions['world_points_from_depth'].shape}")
    
    # 2. Confidence scores
    if "world_points_conf" in predictions:
        neus2_predictions["world_points_conf"] = predictions["world_points_conf"]
        print(f"Added world_points_conf: {predictions['world_points_conf'].shape}")
    elif "depth_conf" in predictions:
        neus2_predictions["depth_conf"] = predictions["depth_conf"]
        print(f"Added depth_conf: {predictions['depth_conf'].shape}")
    else:
        # Create uniform confidence if not available
        if "world_points" in predictions:
            conf_shape = predictions["world_points"].shape[:-1]  # Remove last dimension (xyz)
            neus2_predictions["world_points_conf"] = np.ones(conf_shape, dtype=np.float32)
            print(f"Created uniform confidence: {conf_shape}")
    
    # 3. Images - convert back to uint8 format for NeuS2
    if images_raw is not None:
        # Assume images_raw is preprocessed tensor (S, C, H, W) normalized to [-1, 1] or [0, 1]
        images_np = images_raw.cpu().numpy()
        
        # Convert from (S, C, H, W) to (S, H, W, C)
        if images_np.ndim == 4 and images_np.shape[1] == 3:
            images_np = np.transpose(images_np, (0, 2, 3, 1))
        
        # Denormalize to [0, 255] range
        if images_np.max() <= 1.0:
            # Assuming normalized to [0, 1]
            images_np = (images_np * 255).astype(np.uint8)
        elif images_np.min() >= -1.0 and images_np.max() <= 1.0:
            # Assuming normalized to [-1, 1]
            images_np = ((images_np + 1) * 127.5).astype(np.uint8)
        else:
            # Already in reasonable range
            images_np = np.clip(images_np, 0, 255).astype(np.uint8)
        
        neus2_predictions["images"] = images_np
        print(f"Added images: {images_np.shape}")
    
    # 4. Camera matrices
    neus2_predictions["extrinsic"] = predictions["extrinsic"]
    neus2_predictions["intrinsic"] = predictions["intrinsic"]
    print(f"Added extrinsic: {predictions['extrinsic'].shape}")
    print(f"Added intrinsic: {predictions['intrinsic'].shape}")
    
    # 5. Additional useful data
    if "depth" in predictions:
        neus2_predictions["depth"] = predictions["depth"]
        print(f"Added depth: {predictions['depth'].shape}")
    
    if "pose_enc" in predictions:
        neus2_predictions["pose_enc"] = predictions["pose_enc"]
        print(f"Added pose_enc: {predictions['pose_enc'].shape}")
    
    # 6. Save metadata
    neus2_predictions["metadata"] = {
        "source": "VGGT",
        "input_dir": input_dir,
        "timestamp": time.time(),
        "num_frames": neus2_predictions["extrinsic"].shape[0],
        "image_size": [neus2_predictions["images"].shape[2], neus2_predictions["images"].shape[1]]  # [W, H]
    }
    
    # Save to pickle file
    timestamp = int(time.time())
    predictions_file = os.path.join(output_dir, f"vggt_predictions_{timestamp}.pkl")
    
    with open(predictions_file, 'wb') as f:
        pickle.dump(neus2_predictions, f)
    
    print(f"✅ Saved VGG-T predictions to: {predictions_file}")
    print(f"📊 Prediction summary:")
    for key, value in neus2_predictions.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} ({value.dtype})")
        elif key == "metadata":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value)}")
    
    return predictions_file

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
    
    parser.add_argument(
        "--save_for_neus2",
        action="store_true",
        help="Save predictions in format suitable for NeuS2 conversion"
    )
    
    parser.add_argument(
        "--skip_glb",
        action="store_true", 
        help="Skip GLB generation (useful when only saving predictions)"
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
    print(f"Save for NeuS2: {args.save_for_neus2}")
    print(f"Skip GLB: {args.skip_glb}")
    
    try:
        start_time = time.time()
        
        # ===== MODIFIED SECTION: Capture raw images for NeuS2 =====
        if args.save_for_neus2:
            # Load images separately to keep raw version
            image_names = glob.glob(os.path.join(args.input_dir, "images", "*"))
            image_names = sorted(image_names)
            images_raw = load_and_preprocess_images(image_names)
        else:
            images_raw = None
        
        # Run the model
        predictions = run_model(args.input_dir, model)
        total_time = time.time() - start_time
        
        print(f"SUCCESS! Processing took {total_time:.2f} seconds")
        
        # ===== NEW: Save predictions for NeuS2 conversion =====
        if args.save_for_neus2:
            predictions_file = save_predictions_for_neus2(
                predictions, 
                images_raw, 
                output_dir, 
                args.input_dir
            )
            
            # Print conversion command
            print("\n" + "="*60)
            print("🎯 Ready for NeuS2 conversion!")
            print("="*60)
            print("Run this command to convert to NeuS2 format:")
            print()
            print(f"python vggt_to_neus2_converter.py \\")
            print(f"    {predictions_file} \\")
            print(f"    ./neus2_data \\")
            print(f"    --images_dir {args.input_dir}/images")
            print()
            print("="*60)
        
        # ===== ORIGINAL: Generate GLB (optional) =====
        if not args.skip_glb:
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
            
            print(f"Successfully saved GLB to: {glb_path}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()