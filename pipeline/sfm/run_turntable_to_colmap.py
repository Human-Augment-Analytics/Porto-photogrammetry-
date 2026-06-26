#!/usr/bin/env python3
"""Turntable-aware SfM refinement (VGGT/COLMAP poses -> exact rig -> COLMAP).

For turntable captures (object rotates, cameras static), every view is a fixed
camera seeing the object at a known angle. Given an existing COLMAP scene, this
fits the rig (axis, angular step, one pose + intrinsic per physical camera),
regenerates exact poses on circular orbits, and re-triangulates with masked SIFT.

"""
import argparse
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as Rot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def group_key(name, regex):
    m = re.search(regex, os.path.basename(name))
    return m.group(0) if m else "cam"


def order_key(name):
    nums = re.findall(r'\d+', os.path.basename(name))
    return int(nums[-1]) if nums else 0


def Rt(image):
    T = image.cam_from_world.matrix()
    return T[:, :3], T[:, 3]


def centers(ims):
    return np.array([-(Rt(im)[0].T @ Rt(im)[1]) for im in ims])


def Rf(axis, theta_deg):
    return Rot.from_rotvec(np.radians(theta_deg) * axis).as_matrix()


def fit_axis_step(groups):
    normals, mids = [], []
    for ims in groups.values():
        C = centers(ims)
        c = C.mean(0)
        _, _, Vt = np.linalg.svd(C - c)
        n = Vt[2]
        n = -n if n[1] < 0 else n
        normals.append(n)
        mids.append(c)
    axis = np.mean(normals, 0)
    axis /= np.linalg.norm(axis)
    p0 = np.mean(mids, 0)

    steps = []
    for ims in groups.values():
        v = centers(ims) - p0
        v = v - np.outer(v @ axis, axis)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        for i in range(len(v) - 1):
            cr = np.cross(v[i], v[i + 1])
            steps.append(np.degrees(np.arctan2(
                np.sign(cr @ axis) * np.linalg.norm(cr),
                np.clip(v[i] @ v[i + 1], -1, 1))))
    return axis, p0, float(np.median(steps))


def fit_rig_poses(groups, axis, p0, step):
    out, dc, dr = {}, [], []
    for ims in groups.values():
        Rc_est, Cc_est = [], []
        for i, im in enumerate(ims):
            R, t = Rt(im)
            C = -R.T @ t
            Rc_est.append(R @ Rf(axis, i * step).T)
            Cc_est.append(p0 + Rf(axis, i * step) @ (C - p0))
        Rc = Rot.from_matrix(np.array(Rc_est)).mean().as_matrix()
        Cc = np.mean(Cc_est, 0)
        for i, im in enumerate(ims):
            Rn = Rc @ Rf(axis, i * step)
            Cn = p0 + Rf(axis, i * step).T @ (Cc - p0)
            R, t = Rt(im)
            dc.append(np.linalg.norm(Cn - (-R.T @ t)))
            dr.append(np.degrees(np.linalg.norm(Rot.from_matrix(Rn @ R.T).as_rotvec())))
            out[im.name] = (Rn, Cn)
    return out, float(np.mean(dc)), float(np.mean(dr))


def main():
    p = argparse.ArgumentParser(description="Turntable-aware SfM refinement to COLMAP")
    p.add_argument("--input_dir", required=True,
                   help="COLMAP scene with images/, optional masks/, and sparse/0/")
    p.add_argument("--output_dir", required=True, help="Output COLMAP scene directory")
    p.add_argument("--use_masks", action="store_true", default=False,
                   help="Restrict SIFT to masks/ (auto-on if masks/ exists)")
    p.add_argument("--camera_regex", default=r"camera\d+",
                   help="Regex grouping images into physical cameras")
    p.add_argument("--step_deg", type=float, default=None,
                   help="Override the turntable angular step in degrees")
    p.add_argument("--max_image_size", type=int, default=2400)
    args = p.parse_args()

    t_start = time.time()
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    images_dir = in_dir / "images"
    masks_dir = in_dir / "masks"
    use_masks = args.use_masks or masks_dir.is_dir()

    rec = pycolmap.Reconstruction(str(in_dir / "sparse" / "0"))
    groups = defaultdict(list)
    for im in rec.images.values():
        groups[group_key(im.name, args.camera_regex)].append(im)
    for k in groups:
        groups[k].sort(key=lambda im: order_key(im.name))
    logger.info("Grouped %d images into %d cameras: %s", rec.num_images(), len(groups),
                ", ".join(f"{k}:{len(v)}" for k, v in sorted(groups.items())))

    axis, p0, step_meas = fit_axis_step(groups)
    step = args.step_deg if args.step_deg is not None else step_meas
    candidates = [step] if args.step_deg is not None else [step, -step]

    # the measured step sign is ambiguous; keep whichever fits the input centers best
    poses = best = None
    for s in candidates:
        ps, dc, dr = fit_rig_poses(groups, axis, p0, s)
        if best is None or dc < best[0]:
            best, poses, step = (dc, dr), ps, s
    dc, dr = best
    logger.info("Rig axis %s, step %.3f deg; analytic vs input: dC %.4f, dR %.3f deg",
                np.round(axis, 4), step, dc, dr)

    foc = defaultdict(list)
    W = H = None
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        foc[group_key(im.name, args.camera_regex)].append(cam.params[0])
        W, H = cam.width, cam.height
    shared_focal = {k: float(np.median(v)) for k, v in foc.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    if not (out_dir / "images").exists():
        os.symlink(images_dir, out_dir / "images")

    mask_path = None
    if use_masks:
        if not (out_dir / "masks").exists():
            os.symlink(masks_dir, out_dir / "masks")
        # COLMAP looks for <image_name>.png; images are .jpg, so link <stem>.jpg.png
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

    t0 = time.time()
    pycolmap.extract_features(db_path, str(out_dir / "images"),
                              camera_mode=pycolmap.CameraMode.PER_IMAGE,
                              camera_model="SIMPLE_PINHOLE", reader_options=ro, sift_options=so)
    logger.info("SIFT extraction %.0fs", time.time() - t0)
    t0 = time.time()
    pycolmap.match_exhaustive(db_path)
    logger.info("Matching %.0fs", time.time() - t0)

    db = pycolmap.Database()
    db.open(db_path)
    db_images = db.read_all_images()
    db.close()

    out_rec = pycolmap.Reconstruction()
    cam_id = {}
    for ci, k in enumerate(sorted(shared_focal), start=1):
        cam = pycolmap.Camera.create(ci, "SIMPLE_PINHOLE", shared_focal[k], W, H)
        cam.camera_id = ci
        out_rec.add_camera(cam)
        cam_id[k] = ci
    for dbi in db_images:
        R, C = poses[dbi.name]
        q = Rot.from_matrix(R).as_quat()  # x, y, z, w
        rigid = pycolmap.Rigid3d(pycolmap.Rotation3d([q[0], q[1], q[2], q[3]]), -R @ C)
        img = pycolmap.Image(name=dbi.name, camera_id=cam_id[group_key(dbi.name, args.camera_regex)],
                             image_id=dbi.image_id)
        img.cam_from_world = rigid
        out_rec.add_image(img)
        out_rec.register_image(dbi.image_id)

    out_sparse = str(out_dir / "sparse" / "0")
    tri = pycolmap.triangulate_points(out_rec, db_path, str(out_dir / "images"), out_sparse)
    tl = [pt.track.length() for pt in tri.points3D.values()]
    tri.write(out_sparse)
    logger.info("Triangulated %d points, %d images, mean track %.2f",
                tri.num_points3D(), tri.num_reg_images(), np.mean(tl))
    logger.info("Done in %.1fs -> %s", time.time() - t_start, out_dir)


if __name__ == "__main__":
    main()
