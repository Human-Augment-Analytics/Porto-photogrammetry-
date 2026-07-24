import argparse
import os
import time
from pathlib import Path

import numpy as np
import pycolmap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="dir with images/ and masks/")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_image_size", type=int, default=2400)
    p.add_argument("--camera_model", default="SIMPLE_PINHOLE")
    args = p.parse_args()

    t0 = time.time()
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    images_dir = in_dir / "images"
    masks_dir = in_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "images").exists():
        os.symlink(images_dir, out_dir / "images")
    if masks_dir.is_dir() and not (out_dir / "masks").exists():
        os.symlink(masks_dir, out_dir / "masks")

    mask_path = None
    if masks_dir.is_dir():
        mc = out_dir / "masks_colmap"
        mc.mkdir(exist_ok=True)
        for m in os.listdir(masks_dir):
            link = mc / f"{m.rsplit('.', 1)[0]}.jpg.png"
            if not link.exists():
                os.symlink(masks_dir / m, link)
        mask_path = str(mc)

    db_path = str(out_dir / "database.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    ro = pycolmap.ImageReaderOptions()
    if mask_path:
        ro.mask_path = mask_path
    so = pycolmap.SiftExtractionOptions()
    so.max_image_size = args.max_image_size
    so.num_threads = 8

    t = time.time()
    pycolmap.extract_features(db_path, str(out_dir / "images"),
                              camera_mode=pycolmap.CameraMode.PER_IMAGE,
                              camera_model=args.camera_model,
                              reader_options=ro, sift_options=so)
    print(f"[colmap] extraction {time.time()-t:.0f}s", flush=True)

    t = time.time()
    pycolmap.match_exhaustive(db_path)
    print(f"[colmap] matching {time.time()-t:.0f}s", flush=True)

    t = time.time()
    maps_dir = out_dir / "sparse"
    maps_dir.mkdir(exist_ok=True)
    recs = pycolmap.incremental_mapping(db_path, str(out_dir / "images"), str(maps_dir))
    print(f"[colmap] mapping {time.time()-t:.0f}s -> {len(recs)} model(s)", flush=True)

    if not recs:
        print("COLMAP_FAIL: no model reconstructed", flush=True)
        raise SystemExit(2)

    best_id = max(recs, key=lambda k: recs[k].num_reg_images())
    best = recs[best_id]
    print(f"[colmap] best model {best_id}: {best.num_reg_images()} images, "
          f"{best.num_points3D()} points", flush=True)

    final = maps_dir / "0"
    final.mkdir(exist_ok=True)
    best.write(str(final))
    print(f"COLMAP_DONE {best.num_reg_images()}img {best.num_points3D()}pts "
          f"in {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
