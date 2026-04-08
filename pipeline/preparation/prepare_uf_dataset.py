#!/usr/bin/env python3
"""
Prepare dataset by separating mixed images and masks into separate folders.

Handles data with the naming convention:
  - Images: camera3_camera 3_IMG_7477.JPG
  - Masks:  camera3_camera 3_IMG_7477.jpg.mask.png

Output structure:
  output_dir/
    images/
      camera3_camera 3_IMG_7477.jpg
    masks/
      camera3_camera 3_IMG_7477.png
"""
import re
import os
import shutil
import argparse
import logging
from pathlib import Path
from typing import Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_images_and_masks(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Find all images and masks in the input directory."""
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    images = []
    masks = []
    
    for file_path in input_dir.iterdir():
        if not file_path.is_file():
            continue
            
        name = file_path.name
        
        # Check if it's a mask file (ends with .mask.png or similar patterns)
        if '.mask.png' in name.lower() or name.endswith('.mask.png'):
            masks.append(file_path)
        elif file_path.suffix in image_extensions:
            images.append(file_path)
    
    return sorted(images), sorted(masks)


def get_mask_base_name(mask_path: Path) -> str:
    """
    Extract base name from mask file.
    
    Examples:
      'camera3_camera 3_IMG_7477.jpg.mask.png' -> 'camera3_camera 3_IMG_7477'
      'camera3_camera 3_IMG_7477.JPG.mask.png' -> 'camera3_camera 3_IMG_7477'
    """
    name = mask_path.name
    
    # Handle various mask naming patterns
    # Pattern: basename.jpg.mask.png or basename.JPG.mask.png
    patterns = [
        r'(.+)\.[jJ][pP][eE]?[gG]\.mask\.png$',  # .jpg.mask.png or .jpeg.mask.png
        r'(.+)\.[pP][nN][gG]\.mask\.png$',        # .png.mask.png
        r'(.+)\.mask\.png$',                       # .mask.png (fallback)
    ]
    
    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            return match.group(1)
    
    # Fallback: just remove .mask.png
    if name.lower().endswith('.mask.png'):
        return name[:-9]  # Remove '.mask.png'
    
    return mask_path.stem


def match_images_to_masks(images: List[Path], masks: List[Path]) -> List[Tuple[Path, Path]]:
    """Match images to their corresponding masks."""
    # Create lookup dict for masks by base name (case-insensitive)
    mask_lookup = {}
    for mask in masks:
        base = get_mask_base_name(mask).lower()
        mask_lookup[base] = mask
    
    pairs = []
    unmatched_images = []
    
    for image in images:
        # Get image base name (without extension)
        image_base = image.stem.lower()
        
        if image_base in mask_lookup:
            pairs.append((image, mask_lookup[image_base]))
        else:
            unmatched_images.append(image)
    
    if unmatched_images:
        logger.warning(f"Found {len(unmatched_images)} images without masks:")
        for img in unmatched_images[:5]:  # Show first 5
            logger.warning(f"  - {img.name}")
        if len(unmatched_images) > 5:
            logger.warning(f"  ... and {len(unmatched_images) - 5} more")
    
    return pairs


def create_symlink(src, dest):
    """Create a symbolic link, removing existing link if present."""
    dest = Path(dest)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(os.path.abspath(src), dest)


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    mode: str = "copy",
    include_unmatched: bool = False
) -> dict:
    """
    Organize dataset into separate images/ and masks/ directories.
    
    Args:
        input_dir: Directory containing mixed images and masks
        output_dir: Output directory (will create images/ and masks/ subdirs)
        mode: One of "copy", "move", or "symlink"
        include_unmatched: If True, include images without masks
        
    Returns:
        Dictionary with statistics about the operation
    """
    # Create output directories
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files
    images, masks = find_images_and_masks(input_dir)
    logger.info(f"Found {len(images)} images and {len(masks)} masks")
    
    if not images:
        logger.error("No images found!")
        return {"error": "No images found"}
    
    # Match images to masks
    pairs = match_images_to_masks(images, masks)
    logger.info(f"Matched {len(pairs)} image-mask pairs")
    
    # Select operation based on mode
    operations = {
        "copy": (shutil.copy2, "Copying"),
        "move": (shutil.move, "Moving"),
        "symlink": (create_symlink, "Symlinking"),
    }
    operation, op_name = operations[mode]
    
    processed = 0
    for image_path, mask_path in pairs:
        # Process image (preserve original name)
        dest_image = images_dir / f"{image_path.stem}.jpg"
        operation(image_path, dest_image)
        
        # Process mask (rename to match image stem + .png)
        mask_dest_name = f"{image_path.stem}.png"
        dest_mask = masks_dir / mask_dest_name
        operation(mask_path, dest_mask)
        
        processed += 1
        if processed % 50 == 0:
            logger.info(f"{op_name} {processed}/{len(pairs)} pairs...")
    
    # Handle unmatched images if requested
    unmatched_count = 0
    if include_unmatched:
        matched_images = {p[0] for p in pairs}
        unmatched = [img for img in images if img not in matched_images]
        
        for image_path in unmatched:
            dest_image = images_dir / f"{image_path.stem}.jpg"
            operation(image_path, dest_image)
            unmatched_count += 1
        
        if unmatched_count > 0:
            logger.info(f"Included {unmatched_count} images without masks")
    
    stats = {
        "total_images": len(images),
        "total_masks": len(masks),
        "matched_pairs": len(pairs),
        "unmatched_included": unmatched_count,
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "mode": mode
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset preparation complete!")
    logger.info(f"  Images: {images_dir}")
    logger.info(f"  Masks:  {masks_dir}")
    logger.info(f"  Pairs:  {len(pairs)}")
    logger.info(f"{'='*60}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset by separating images and masks into folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - copy files to organized structure (default)
  python prepare_dataset.py /path/to/mixed/data --out /path/to/output

  # Move files instead of copying (saves disk space)
  python prepare_dataset.py /path/to/mixed/data --out /path/to/output --mode move

  # Create symbolic links (saves disk space, keeps originals)
  python prepare_dataset.py /path/to/mixed/data --out /path/to/output --mode symlink

  # Include images that don't have matching masks
  python prepare_dataset.py /path/to/mixed/data --out /path/to/output --include-unmatched
        """
    )
    parser.add_argument("input_dir", type=Path, 
                        help="Input directory containing mixed images and masks")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: input_dir/organized)")
    parser.add_argument("--mode", type=str, choices=["copy", "move", "symlink"], default="copy",
                        help="How to handle files: copy (default), move, or symlink")
    parser.add_argument("--include-unmatched", action="store_true",
                        help="Include images without masks")
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    if args.out is None:
        args.out = args.input_dir.parent / f"{args.input_dir.name}_organized"
    
    logger.info(f"Input:  {args.input_dir}")
    logger.info(f"Output: {args.out}")
    logger.info(f"Mode:   {args.mode}")
    
    stats = prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.out,
        mode=args.mode,
        include_unmatched=args.include_unmatched
    )
    
    if "error" in stats:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
