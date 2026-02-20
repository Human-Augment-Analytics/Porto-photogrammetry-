#!/usr/bin/env python3
"""
VGGT to COLMAP Converter

Converts VGGT predictions (.pkl) to COLMAP binary format for use with
3D Gaussian Splatting and SuGaR mesh extraction.

VGGT extrinsics are world-to-camera (w2c) in OpenCV convention,
which matches COLMAP's convention directly — no axis flipping needed.

Output structure:
    output_dir/
    ├── images/
    │   ├── 000000.jpg
    │   └── ...
    └── sparse/
        └── 0/
            ├── cameras.bin
            ├── images.bin
            └── points3D.bin
"""

import struct
import numpy as np
import pickle
import os
import shutil
import argparse
from pathlib import Path


# COLMAP camera model IDs
CAMERA_MODEL_PINHOLE = 1


def rotation_matrix_to_quaternion_colmap(R):
    """
    Convert 3x3 rotation matrix to COLMAP quaternion (qw, qx, qy, qz).

    Uses Shepperd's method for numerical stability.
    Enforces qw >= 0 to match COLMAP convention.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        q: (4,) quaternion in (qw, qx, qy, qz) scalar-first format
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)

    # COLMAP convention: ensure qw >= 0
    if q[0] < 0:
        q = -q

    return q


def write_cameras_binary(cameras, path):
    """
    Write cameras.bin in COLMAP binary format.

    Args:
        cameras: list of dicts with keys:
            camera_id (int), model_id (int), width (int), height (int),
            params (list of float) — for PINHOLE: [fx, fy, cx, cy]
        path: output file path
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("<I", cam["camera_id"]))
            f.write(struct.pack("<i", cam["model_id"]))
            f.write(struct.pack("<Q", cam["width"]))
            f.write(struct.pack("<Q", cam["height"]))
            for p in cam["params"]:
                f.write(struct.pack("<d", p))


def write_images_binary(images, path):
    """
    Write images.bin in COLMAP binary format.

    Args:
        images: list of dicts with keys:
            image_id (int), qvec (4,) array [qw,qx,qy,qz],
            tvec (3,) array [tx,ty,tz], camera_id (int),
            name (str), points2D (list of (x, y, point3D_id))
        path: output file path
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            f.write(struct.pack("<I", img["image_id"]))
            for q in img["qvec"]:
                f.write(struct.pack("<d", float(q)))
            for t in img["tvec"]:
                f.write(struct.pack("<d", float(t)))
            f.write(struct.pack("<I", img["camera_id"]))
            # Null-terminated image name
            for c in img["name"]:
                f.write(c.encode("utf-8"))
            f.write(b"\x00")
            # 2D point observations
            pts2d = img.get("points2D", [])
            f.write(struct.pack("<Q", len(pts2d)))
            for x, y, pid in pts2d:
                f.write(struct.pack("<d", float(x)))
                f.write(struct.pack("<d", float(y)))
                f.write(struct.pack("<q", int(pid)))


def write_points3D_binary(points3D, path):
    """
    Write points3D.bin in COLMAP binary format.

    Args:
        points3D: list of dicts with keys:
            point3D_id (int), xyz (3,) array, rgb (3,) array uint8,
            error (float), track (list of (image_id, point2D_idx))
        path: output file path
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points3D)))
        for pt in points3D:
            f.write(struct.pack("<Q", pt["point3D_id"]))
            for c in pt["xyz"]:
                f.write(struct.pack("<d", float(c)))
            for c in pt["rgb"]:
                f.write(struct.pack("<B", int(c)))
            f.write(struct.pack("<d", float(pt["error"])))
            track = pt.get("track", [])
            f.write(struct.pack("<Q", len(track)))
            for img_id, pt2d_idx in track:
                f.write(struct.pack("<I", int(img_id)))
                f.write(struct.pack("<I", int(pt2d_idx)))


def subsample_world_points(
    world_points,
    confidence,
    images,
    conf_threshold_percentile=70.0,
    max_points=200_000,
    stride=4,
):
    """
    Subsample VGGT world points by confidence for use as COLMAP points3D.

    Args:
        world_points: (S, H, W, 3) world coordinates
        confidence: (S, H, W) confidence scores
        images: (S, H, W, 3) uint8 RGB
        conf_threshold_percentile: percentile above which to keep points
        max_points: maximum points to return
        stride: spatial stride for initial subsampling

    Returns:
        points_xyz: (N, 3) world coordinates
        points_rgb: (N, 3) uint8 colors
        points_frame_idx: (N,) source frame index
        points_pixel_xy: (N, 2) pixel coordinates in source image
    """
    S, H, W, _ = world_points.shape

    # Spatial subsampling
    wp_sub = world_points[:, ::stride, ::stride, :]
    conf_sub = confidence[:, ::stride, ::stride]
    img_sub = images[:, ::stride, ::stride, :]

    Hs, Ws = conf_sub.shape[1], conf_sub.shape[2]

    # Build pixel coordinate arrays (in original image space)
    ys = np.arange(0, H, stride)[:Hs]
    xs = np.arange(0, W, stride)[:Ws]
    xx, yy = np.meshgrid(xs, ys)
    pixel_xy = np.stack([xx, yy], axis=-1)  # (Hs, Ws, 2)
    pixel_xy = np.broadcast_to(pixel_xy[None], (S, Hs, Ws, 2)).copy()
    frame_idx = np.broadcast_to(
        np.arange(S)[:, None, None], (S, Hs, Ws)
    ).copy()

    # Flatten
    flat_pts = wp_sub.reshape(-1, 3)
    flat_conf = conf_sub.reshape(-1)
    flat_rgb = img_sub.reshape(-1, 3)
    flat_pixel = pixel_xy.reshape(-1, 2).astype(np.float64)
    flat_frame = frame_idx.reshape(-1)

    # Filter: finite, non-zero
    valid = np.all(np.isfinite(flat_pts), axis=1)
    valid &= ~np.all(flat_pts == 0, axis=1)
    valid &= np.isfinite(flat_conf)

    if np.sum(valid) == 0:
        print("Warning: no valid world points found")
        return (
            np.zeros((0, 3)),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 2)),
        )

    # Confidence threshold
    conf_thresh = np.percentile(flat_conf[valid], conf_threshold_percentile)
    valid &= flat_conf >= conf_thresh

    # Outlier removal per axis (1st-99th percentile)
    valid_pts_temp = flat_pts[valid]
    outlier_mask = np.ones(len(valid_pts_temp), dtype=bool)
    for axis in range(3):
        q1, q99 = np.percentile(valid_pts_temp[:, axis], [1, 99])
        outlier_mask &= (valid_pts_temp[:, axis] >= q1) & (
            valid_pts_temp[:, axis] <= q99
        )

    valid_indices = np.where(valid)[0][outlier_mask]

    pts = flat_pts[valid_indices]
    rgb = flat_rgb[valid_indices]
    pxy = flat_pixel[valid_indices]
    fidx = flat_frame[valid_indices]

    # Random subsample if too many
    if len(pts) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pts), max_points, replace=False)
        pts = pts[indices]
        rgb = rgb[indices]
        pxy = pxy[indices]
        fidx = fidx[indices]

    print(f"Subsampled {len(pts)} points from {S} frames")
    return pts, rgb, fidx, pxy


def convert_vggt_to_colmap(
    predictions_path,
    output_dir,
    images_source_dir=None,
    conf_threshold_percentile=70.0,
    max_points=200_000,
    point_stride=4,
    use_shared_camera=True,
):
    """
    Convert VGGT predictions pickle to COLMAP binary format for SuGaR/3DGS.

    VGGT extrinsics are w2c in OpenCV convention — same as COLMAP.
    No coordinate system transformation is needed.

    Args:
        predictions_path: path to .pkl file from run_pipeline.py
        output_dir: output directory (creates images/ and sparse/0/)
        images_source_dir: optional path to original full-res images
        conf_threshold_percentile: percentile threshold for point filtering
        max_points: maximum number of 3D points
        point_stride: spatial stride for subsampling world points
        use_shared_camera: if True, all images share one camera model

    Returns:
        path to output_dir
    """
    print("=" * 60)
    print("VGGT -> COLMAP Converter (for SuGaR / 3DGS)")
    print("=" * 60)

    # Load predictions
    print(f"Loading predictions from: {predictions_path}")
    with open(predictions_path, "rb") as f:
        predictions = pickle.load(f)

    extrinsics = predictions["extrinsic"]  # (S, 3, 4) w2c
    intrinsics = predictions["intrinsic"]  # (S, 3, 3)
    S = extrinsics.shape[0]

    # Get VGGT image dimensions from predictions
    if "images" in predictions:
        vggt_h, vggt_w = predictions["images"].shape[1], predictions["images"].shape[2]
    elif "metadata" in predictions:
        vggt_w, vggt_h = predictions["metadata"]["image_size"]
    else:
        raise ValueError("Cannot determine image dimensions from predictions")

    print(f"VGGT dimensions: {vggt_w}x{vggt_h}, {S} frames")

    # Determine output image dimensions and scaling
    use_originals = False
    scale_factor = 1.0
    output_w, output_h = vggt_w, vggt_h

    if images_source_dir:
        source_images = sorted(
            [
                f
                for f in os.listdir(images_source_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
            ]
        )
        if len(source_images) >= S:
            from PIL import Image

            first_img = Image.open(os.path.join(images_source_dir, source_images[0]))
            orig_w, orig_h = first_img.size
            first_img.close()

            if orig_w != vggt_w or orig_h != vggt_h:
                scale_factor = orig_w / vggt_w
                output_w, output_h = orig_w, orig_h
                use_originals = True
                print(
                    f"Using original images: {orig_w}x{orig_h} "
                    f"(scale factor: {scale_factor:.2f}x)"
                )
        else:
            print(
                f"Warning: found {len(source_images)} source images "
                f"but need {S}, using VGGT images"
            )

    # Create output directory structure
    images_out_dir = os.path.join(output_dir, "images")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # --- Save images ---
    image_names = []
    if use_originals:
        for i in range(S):
            src = os.path.join(images_source_dir, source_images[i])
            dst_name = f"{i:06d}{Path(source_images[i]).suffix}"
            dst = os.path.join(images_out_dir, dst_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            image_names.append(dst_name)
    elif "images" in predictions:
        from PIL import Image as PILImage

        for i in range(S):
            img_array = predictions["images"][i]
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            dst_name = f"{i:06d}.jpg"
            pil_img = PILImage.fromarray(img_array)
            pil_img.save(os.path.join(images_out_dir, dst_name), quality=95)
            image_names.append(dst_name)
    else:
        raise ValueError("No images available in predictions or source directory")

    print(f"Saved {len(image_names)} images to {images_out_dir}")

    # --- Build cameras.bin ---
    # Scale intrinsics if using original resolution
    scaled_intrinsics = intrinsics.copy()
    if scale_factor != 1.0:
        for i in range(S):
            scaled_intrinsics[i, 0, 0] *= scale_factor  # fx
            scaled_intrinsics[i, 1, 1] *= scale_factor  # fy
            scaled_intrinsics[i, 0, 2] *= scale_factor  # cx
            scaled_intrinsics[i, 1, 2] *= scale_factor  # cy

    if use_shared_camera:
        # Average intrinsics across all frames
        avg_fx = float(np.mean(scaled_intrinsics[:, 0, 0]))
        avg_fy = float(np.mean(scaled_intrinsics[:, 1, 1]))
        avg_cx = float(np.mean(scaled_intrinsics[:, 0, 2]))
        avg_cy = float(np.mean(scaled_intrinsics[:, 1, 2]))

        cameras = [
            {
                "camera_id": 1,
                "model_id": CAMERA_MODEL_PINHOLE,
                "width": output_w,
                "height": output_h,
                "params": [avg_fx, avg_fy, avg_cx, avg_cy],
            }
        ]
    else:
        cameras = []
        for i in range(S):
            K = scaled_intrinsics[i]
            cameras.append(
                {
                    "camera_id": i + 1,
                    "model_id": CAMERA_MODEL_PINHOLE,
                    "width": output_w,
                    "height": output_h,
                    "params": [
                        float(K[0, 0]),
                        float(K[1, 1]),
                        float(K[0, 2]),
                        float(K[1, 2]),
                    ],
                }
            )

    write_cameras_binary(cameras, os.path.join(sparse_dir, "cameras.bin"))
    print(f"Wrote cameras.bin ({len(cameras)} camera{'s' if len(cameras) > 1 else ''})")

    # --- Build images.bin ---
    colmap_images = []
    for i in range(S):
        R_w2c = extrinsics[i, :3, :3]
        t_w2c = extrinsics[i, :3, 3]

        qvec = rotation_matrix_to_quaternion_colmap(R_w2c)

        colmap_images.append(
            {
                "image_id": i + 1,
                "qvec": qvec,
                "tvec": t_w2c.astype(np.float64),
                "camera_id": 1 if use_shared_camera else i + 1,
                "name": image_names[i],
                "points2D": [],  # Will be populated with point associations
            }
        )

    # --- Build points3D.bin ---
    world_points = predictions.get("world_points")
    confidence = predictions.get("world_points_conf")

    if world_points is None:
        world_points = predictions.get("world_points_from_depth")
        confidence = predictions.get("depth_conf")

    if world_points is None:
        print("Warning: no world points available, writing empty points3D.bin")
        points3D_list = []
    else:
        pred_images = predictions.get("images")
        if pred_images is None:
            pred_images = np.zeros((S, vggt_h, vggt_w, 3), dtype=np.uint8)

        if confidence is None:
            confidence = np.ones((S, vggt_h, vggt_w), dtype=np.float32)

        pts_xyz, pts_rgb, pts_frame, pts_pixel = subsample_world_points(
            world_points,
            confidence,
            pred_images,
            conf_threshold_percentile=conf_threshold_percentile,
            max_points=max_points,
            stride=point_stride,
        )

        # Build points3D list with track information
        points3D_list = []
        # Also build 2D point associations for images.bin
        image_points2D = {i + 1: [] for i in range(S)}

        for pid in range(len(pts_xyz)):
            frame_id = int(pts_frame[pid]) + 1  # 1-indexed
            pt2d_idx = len(image_points2D[frame_id])

            # Scale pixel coordinates if using original resolution
            px = pts_pixel[pid, 0] * scale_factor
            py = pts_pixel[pid, 1] * scale_factor

            image_points2D[frame_id].append((px, py, pid + 1))

            points3D_list.append(
                {
                    "point3D_id": pid + 1,
                    "xyz": pts_xyz[pid].astype(np.float64),
                    "rgb": pts_rgb[pid].astype(np.uint8),
                    "error": 0.0,
                    "track": [(frame_id, pt2d_idx)],
                }
            )

        # Update images with 2D point associations
        for img in colmap_images:
            img["points2D"] = image_points2D.get(img["image_id"], [])

    write_images_binary(colmap_images, os.path.join(sparse_dir, "images.bin"))
    print(f"Wrote images.bin ({S} images)")

    write_points3D_binary(points3D_list, os.path.join(sparse_dir, "points3D.bin"))
    print(f"Wrote points3D.bin ({len(points3D_list)} points)")

    # Summary
    print()
    print("=" * 60)
    print("COLMAP export complete!")
    print(f"  Output:    {output_dir}")
    print(f"  Images:    {len(image_names)} ({output_w}x{output_h})")
    print(f"  Points3D:  {len(points3D_list)}")
    print(f"  Cameras:   {len(cameras)} (PINHOLE)")
    print("=" * 60)

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert VGGT predictions to COLMAP format for SuGaR/3DGS"
    )
    parser.add_argument(
        "predictions_path", help="Path to VGGT predictions .pkl file"
    )
    parser.add_argument(
        "output_dir", help="Output directory for COLMAP data"
    )
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Source directory with original full-resolution images",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=70.0,
        help="Confidence percentile threshold for point filtering (0-100)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=200_000,
        help="Maximum number of 3D points",
    )
    parser.add_argument(
        "--point_stride",
        type=int,
        default=4,
        help="Spatial stride for subsampling world points",
    )
    parser.add_argument(
        "--per_image_camera",
        action="store_true",
        help="Use a separate camera model per image (default: shared)",
    )

    args = parser.parse_args()

    convert_vggt_to_colmap(
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        images_source_dir=args.images_dir,
        conf_threshold_percentile=args.conf_threshold,
        max_points=args.max_points,
        point_stride=args.point_stride,
        use_shared_camera=not args.per_image_camera,
    )

    print()
    print("Next steps:")
    print("  # Train 3D Gaussian Splatting:")
    print(f"  python gaussian-splatting/train.py -s {args.output_dir}")
    print()
    print("  # Or run full SuGaR pipeline:")
    print(f"  python SuGaR/train_full_pipeline.py -s {args.output_dir} \\")
    print("      -r dn_consistency --high_poly True --export_obj True")


if __name__ == "__main__":
    main()
