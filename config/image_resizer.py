#!/usr/bin/env python3
"""
Image Resizer for NeuS2 Training
Efficiently resize images and masks for faster training while maintaining aspect ratio
"""

import os
import argparse
from PIL import Image, ImageOps
import json
from pathlib import Path
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def calculate_target_size(original_size, target_size):
    """
    Calculate target size maintaining aspect ratio
    
    Args:
        original_size: (width, height) of original image
        target_size: (target_width, target_height) or single dimension
        
    Returns:
        (width, height) that maintains aspect ratio
    """
    orig_w, orig_h = original_size
    
    if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
        target_w, target_h = target_size
    else:
        # Single dimension - calculate other dimension maintaining aspect ratio
        if orig_w >= orig_h:
            target_w = target_size
            target_h = int(target_size * orig_h / orig_w)
        else:
            target_h = target_size
            target_w = int(target_size * orig_w / orig_h)
    
    # Ensure even dimensions (helpful for some neural networks)
    target_w = target_w if target_w % 2 == 0 else target_w + 1
    target_h = target_h if target_h % 2 == 0 else target_h + 1
    
    return target_w, target_h

def resize_single_image(args):
    """
    Resize a single image (for parallel processing)
    
    Args:
        args: tuple of (input_path, output_path, target_size, quality, is_mask)
    """
    input_path, output_path, target_size, quality, is_mask = args
    
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if needed (except for masks)
            if not is_mask and img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            elif is_mask and img.mode != 'L':
                # Convert masks to grayscale
                img = img.convert('L')
            
            # Calculate target size maintaining aspect ratio
            new_size = calculate_target_size(img.size, target_size)
            
            # Resize using high-quality resampling
            if is_mask:
                # For masks, use nearest neighbor to preserve crisp edges
                resized = img.resize(new_size, Image.NEAREST)
            else:
                # For images, use Lanczos for high quality
                resized = img.resize(new_size, Image.LANCZOS)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save with appropriate quality
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                resized.save(output_path, 'JPEG', quality=quality, optimize=True)
            else:
                resized.save(output_path, optimize=True)
            
            return input_path, new_size, True
    
    except Exception as e:
        return input_path, None, f"Error: {e}"

def resize_images_parallel(input_dir, output_dir, target_size=(1024, 768), 
                          quality=95, max_workers=4, is_mask=False, sequential_naming=True):
    """
    Resize images in parallel
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        target_size: Target size (width, height) or single dimension
        quality: JPEG quality (1-100)
        max_workers: Number of parallel workers
        is_mask: Whether these are mask images
        sequential_naming: Use sequential naming (000000.jpg, 000001.jpg, etc.)
        
    Returns:
        dict: Results summary
    """
    # Find all image files (case-insensitive, avoiding duplicates)
    image_files = []
    
    # Use case-insensitive search to avoid duplicates
    for root, dirs, files in os.walk(input_dir):
        if root != input_dir:  # Only look in the top level directory
            continue
        for file in files:
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']):
                image_files.append(os.path.join(root, file))
    
    # Sort for consistent ordering
    image_files.sort()
    
    if not image_files:
        return {"success": 0, "failed": 0, "files": [], "errors": []}
    
    print(f"Found {len(image_files)} unique image files")
    
    # Prepare arguments for parallel processing
    resize_args = []
    for i, input_path in enumerate(image_files):
        if sequential_naming:
            # Use sequential naming: 000000.jpg, 000001.jpg, etc.
            if is_mask:
                filename = f"{i:06d}.png"  # Masks as PNG for better quality
            else:
                filename = f"{i:06d}.jpg"  # Images as JPG to match input format
        else:
            # Keep original filename
            filename = os.path.basename(input_path)
            # Change extension to PNG for masks, keep original for images
            if is_mask and not filename.lower().endswith('.png'):
                name_without_ext = os.path.splitext(filename)[0]
                filename = name_without_ext + '.png'
        
        output_path = os.path.join(output_dir, filename)
        resize_args.append((input_path, output_path, target_size, quality, is_mask))
    
    # Process images in parallel
    results = {"success": 0, "failed": 0, "files": [], "errors": [], "final_size": None}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(resize_single_image, args): args[0] 
                         for args in resize_args}
        
        # Process results with progress bar
        for future in tqdm(as_completed(future_to_path), total=len(resize_args), 
                          desc=f"Resizing {'masks' if is_mask else 'images'}"):
            input_path, size, result = future.result()
            
            if result is True:
                results["success"] += 1
                results["files"].append(input_path)
                if size and not results["final_size"]:
                    results["final_size"] = size
            else:
                results["failed"] += 1
                results["errors"].append(f"{input_path}: {result}")
    
    return results

def update_transforms_json(transforms_path, new_width, new_height):
    """
    Update transforms.json with new image dimensions
    
    Args:
        transforms_path: Path to transforms.json
        new_width: New image width
        new_height: New image height
    """
    if not os.path.exists(transforms_path):
        print(f"Warning: {transforms_path} not found, skipping update")
        return
    
    try:
        with open(transforms_path, 'r') as f:
            data = json.load(f)
        
        # Store original dimensions for reference
        original_w = data.get('w', 'unknown')
        original_h = data.get('h', 'unknown')
        
        # Update dimensions
        data['w'] = new_width
        data['h'] = new_height
        
        # Update camera intrinsics proportionally
        if 'fl_x' in data and original_w != 'unknown':
            scale_x = new_width / original_w
            scale_y = new_height / original_h
            
            data['fl_x'] *= scale_x
            data['fl_y'] *= scale_y
            data['cx'] = new_width / 2.0
            data['cy'] = new_height / 2.0
            
            # Update per-frame intrinsics if they exist
            for frame in data.get('frames', []):
                if 'intrinsic_matrix' in frame:
                    intrinsics = frame['intrinsic_matrix']
                    intrinsics[0][0] *= scale_x  # fx
                    intrinsics[1][1] *= scale_y  # fy
                    intrinsics[0][2] = new_width / 2.0   # cx
                    intrinsics[1][2] = new_height / 2.0  # cy
        
        # Save updated transforms
        with open(transforms_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Updated {transforms_path}: {original_w}x{original_h} → {new_width}x{new_height}")
        
    except Exception as e:
        print(f"Error updating transforms.json: {e}")

def main():
    parser = argparse.ArgumentParser(description="Resize images for NeuS2 training")
    
    # Input/Output
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for resized images")
    parser.add_argument("--mask_input_dir", type=str, default=None,
                       help="Input directory containing masks (optional)")
    parser.add_argument("--mask_output_dir", type=str, default=None,
                       help="Output directory for resized masks (optional)")
    
    # Size options
    parser.add_argument("--width", type=int, default=1024,
                       help="Target width (default: 1024)")
    parser.add_argument("--height", type=int, default=768,
                       help="Target height (default: 768)")
    parser.add_argument("--size", type=int, default=None,
                       help="Single dimension (maintains aspect ratio)")
    
    # Quality options
    parser.add_argument("--quality", type=int, default=95,
                       help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    
    # Transforms update
    parser.add_argument("--transforms", type=str, default="transforms.json",
                       help="Path to transforms.json to update (default: transforms.json)")
    parser.add_argument("--skip_transforms_update", action="store_true",
                       help="Skip updating transforms.json")
    
    # Optional: Keep original naming
    parser.add_argument("--keep_original_names", action="store_true",
                       help="Keep original filenames instead of sequential naming")
    
    # Common presets
    parser.add_argument("--preset", type=str, choices=['test', 'medium', 'high'], default=None,
                       help="Size presets: test(512x384), medium(1024x768), high(2048x1536)")
    
    args = parser.parse_args()
    
    # Handle presets
    if args.preset:
        presets = {
            'test': (512, 384),
            'medium': (1024, 768),
            'high': (2048, 1536)
        }
        args.width, args.height = presets[args.preset]
        print(f"Using {args.preset} preset: {args.width}x{args.height}")
    
    # Determine target size
    if args.size:
        target_size = args.size
        print(f"Resizing to {args.size}px (maintaining aspect ratio)")
    else:
        target_size = (args.width, args.height)
        print(f"Resizing to {args.width}x{args.height}")
    
    # Resize images
    print("Resizing images...")
    img_results = resize_images_parallel(
        args.input_dir, args.output_dir, target_size, 
        args.quality, args.workers, is_mask=False, sequential_naming=True
    )
    
    print(f"Images: {img_results['success']} successful, {img_results['failed']} failed")
    
    # Resize masks if specified
    if args.mask_input_dir and args.mask_output_dir:
        print("Resizing masks...")
        mask_results = resize_images_parallel(
            args.mask_input_dir, args.mask_output_dir, target_size,
            100, args.workers, is_mask=True, sequential_naming=True  # Use max quality for masks
        )
        print(f"Masks: {mask_results['success']} successful, {mask_results['failed']} failed")
    
    # Update transforms.json
    if not args.skip_transforms_update and img_results['final_size']:
        final_w, final_h = img_results['final_size']
        update_transforms_json(args.transforms, final_w, final_h)
    
    # Print any errors
    if img_results['errors']:
        print("\nImage errors:")
        for error in img_results['errors']:
            print(f"  {error}")
    
    if args.mask_input_dir and 'mask_results' in locals() and mask_results['errors']:
        print("\nMask errors:")
        for error in mask_results['errors']:
            print(f"  {error}")
    
    print(f"\nDone! Resized images saved to: {args.output_dir}")
    if args.mask_output_dir:
        print(f"Resized masks saved to: {args.mask_output_dir}")

if __name__ == "__main__":
    main()