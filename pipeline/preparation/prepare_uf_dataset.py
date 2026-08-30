#!/usr/bin/env python3
"""
Separate a specimen's mixed images and masks into a COLMAP-ready scene.

Walks the input directory, so one run over a specimen's image bundle gathers
every camera into a single output scene.

Naming convention:
  Images: camera3_camera 3_IMG_7477.JPG
  Masks:  camera3_camera 3_IMG_7477.jpg.mask.png  ('.<ext>.mask' may repeat)

Output:
  output_dir/images/camera3_camera 3_IMG_7477.jpg
  output_dir/masks/camera3_camera 3_IMG_7477.png
"""
import re
import os
import shutil
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


def find_images_and_masks(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Collect every image and mask under input_dir, recursively.

    When subdirectories hold images, files loose at the top level are treated as
    leftovers rather than capture output and are skipped.

    Args:
        input_dir: Directory to search, typically a specimen's image bundle.

    Returns:
        Sorted (images, masks) lists.
    """
    top_images, top_masks = [], []
    nested_images, nested_masks = [], []

    for dir_path, _dir_names, file_names in os.walk(input_dir):
        directory = Path(dir_path)
        at_top = directory == input_dir

        for file_name in file_names:
            file_path = directory / file_name

            # Masks end in .png too, so they must be claimed before the image test.
            if '.mask.png' in file_name.lower():
                (top_masks if at_top else nested_masks).append(file_path)
            elif file_path.suffix in IMAGE_EXTENSIONS:
                (top_images if at_top else nested_images).append(file_path)

    if not nested_images:
        return sorted(top_images), sorted(top_masks)

    if top_images:
        logger.info(f"Ignoring {len(top_images)} image(s) loose in {input_dir}; using subdirectories")

    return sorted(nested_images), sorted(nested_masks)


def get_mask_base_name(mask_path: Path) -> str:
    """
    Recover the image base name a mask belongs to.

    Strips the trailing '.png' and every '.<ext>.mask' layer, since repeated mask
    exports stack that suffix (camera1_IMG_3009.jpg.mask.jpg.mask.png).
    """
    name = mask_path.name

    if name.lower().endswith('.png'):
        name = name[:-4]

    patterns = [
        r'(.+)\.[jJ][pP][eE]?[gG]\.mask$',
        r'(.+)\.[pP][nN][gG]\.mask$',
        r'(.+)\.mask$',
    ]

    stripped = True
    while stripped:
        stripped = False
        for pattern in patterns:
            match = re.match(pattern, name)
            if match:
                name = match.group(1)
                stripped = True
                break

    return name if name else mask_path.stem


def match_images_to_masks(images: List[Path], masks: List[Path]) -> List[Tuple[Path, Path]]:
    """Pair each image with its mask, matching on base name."""
    mask_lookup = {}
    for mask in masks:
        base = get_mask_base_name(mask).lower()
        mask_lookup[base] = mask

    pairs = []
    unmatched_images = []

    for image in images:
        image_base = image.stem.lower()

        if image_base in mask_lookup:
            pairs.append((image, mask_lookup[image_base]))
        else:
            unmatched_images.append(image)

    if unmatched_images:
        logger.warning(f"Found {len(unmatched_images)} images without masks:")
        for img in unmatched_images[:5]:
            logger.warning(f"  - {img.name}")
        if len(unmatched_images) > 5:
            logger.warning(f"  ... and {len(unmatched_images) - 5} more")

    return pairs


def resolve_output_stems(images: List[Path]) -> Dict[Path, str]:
    """
    Assign each image a unique output stem.

    Merging subdirectories can bring together identical file names, which would
    otherwise overwrite each other, so clashes take their parent directory as a prefix.

    Args:
        images: Source image paths, in output order.

    Returns:
        Mapping of image path to output stem.
    """
    repeated = {stem for stem, count in Counter(i.stem for i in images).items() if count > 1}
    stems: Dict[Path, str] = {}
    used = set()

    for image_path in images:
        stem = f"{image_path.parent.name}_{image_path.stem}" if image_path.stem in repeated else image_path.stem

        candidate, suffix = stem, 2
        while candidate in used:
            candidate = f"{stem}_{suffix}"
            suffix += 1

        if candidate != image_path.stem:
            logger.warning(f"Duplicate name '{image_path.stem}' from {image_path.parent} written as '{candidate}'")

        used.add(candidate)
        stems[image_path] = candidate

    return stems


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
    Organize a specimen into one output directory with images/ and masks/.

    Args:
        input_dir: Directory of mixed images and masks, searched recursively.
        output_dir: Destination, gains images/ and masks/ subdirectories.
        mode: One of "copy", "move", or "symlink".
        include_unmatched: Also emit images that have no mask.

    Returns:
        Dictionary with statistics about the operation.
    """
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    images, masks = find_images_and_masks(input_dir)
    source_dirs = {image.parent for image in images}
    logger.info(f"Found {len(images)} images and {len(masks)} masks across {len(source_dirs)} directories")

    if not images:
        logger.error("No images found!")
        return {"error": "No images found"}

    pairs = match_images_to_masks(images, masks)
    logger.info(f"Matched {len(pairs)} image-mask pairs")

    matched_images = {image for image, _ in pairs}
    unmatched = [image for image in images if image not in matched_images] if include_unmatched else []
    stems = resolve_output_stems([image for image, _ in pairs] + unmatched)

    operations = {
        "copy": (shutil.copy2, "Copying"),
        "move": (shutil.move, "Moving"),
        "symlink": (create_symlink, "Symlinking"),
    }
    operation, op_name = operations[mode]

    processed = 0
    for image_path, mask_path in pairs:
        stem = stems[image_path]
        operation(image_path, images_dir / f"{stem}.jpg")

        # COLMAP pairs a mask to its image by name, so the mask takes the image stem.
        operation(mask_path, masks_dir / f"{stem}.png")

        processed += 1
        if processed % 50 == 0:
            logger.info(f"{op_name} {processed}/{len(pairs)} pairs...")

    for image_path in unmatched:
        operation(image_path, images_dir / f"{stems[image_path]}.jpg")

    if unmatched:
        logger.info(f"Included {len(unmatched)} images without masks")

    stats = {
        "total_images": len(images),
        "total_masks": len(masks),
        "source_dirs": len(source_dirs),
        "matched_pairs": len(pairs),
        "unmatched_included": len(unmatched),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "mode": mode
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset preparation complete!")
    logger.info(f"  Images: {images_dir}")
    logger.info(f"  Masks:  {masks_dir}")
    logger.info(f"  Pairs:  {len(pairs)}")
    logger.info(f"  Dirs:   {len(source_dirs)}")
    logger.info(f"{'='*60}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset by separating images and masks into folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # A whole specimen - every camera folder lands in one scene (default: copy)
  python pipeline/preparation/prepare_uf_dataset.py \\
      data/morphosource/000381689/UF_Fish_181080__000816964/UF_Fish_181080_images \\
      --out data/uf_fish_181080

  # Move files instead of copying (saves disk space)
  python pipeline/preparation/prepare_uf_dataset.py /path/to/specimen --out /path/to/output --mode move

  # Create symbolic links (saves disk space, keeps originals)
  python pipeline/preparation/prepare_uf_dataset.py /path/to/specimen --out /path/to/output --mode symlink

  # Include images that don't have matching masks
  python pipeline/preparation/prepare_uf_dataset.py /path/to/specimen --out /path/to/output --include-unmatched
        """
    )
    parser.add_argument("input_dir", type=Path, 
                        help="Input directory containing mixed images and masks, searched recursively")
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
