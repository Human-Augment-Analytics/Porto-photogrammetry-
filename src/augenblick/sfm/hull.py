"""Visual-hull space carving: a silhouette-derived point cloud to initialise 2DGS from."""
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_sfm
from augenblick.core.scene import Scene
from augenblick.sfm.base import SceneRefiner, SfMResult

logger = logging.getLogger(__name__)

_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


@dataclass(frozen=True)
class HullConfig:
    """Space-carving grid, voting, and init-cloud sampling parameters."""

    res: int = field(default=256, metadata={"help": "Voxel grid resolution per axis"})
    tau: float = field(default=0.92, metadata={
        "help": "Fraction of seeing cameras that must vote a voxel inside the silhouette"})
    min_seen_frac: float = field(default=0.5, metadata={
        "help": "Voxel must be inside the frustum of at least this fraction of cameras"})
    mask_downscale: int = field(default=4, metadata={"help": "Mask downsample factor for carving"})
    img_downscale: int = field(default=4, metadata={"help": "Image downsample factor for colouring"})
    pad: float = field(default=0.08, metadata={"help": "Bounding-box padding fraction"})
    bbox_percentile: float = field(default=0.5, metadata={
        "help": "Percentile trimmed off each end of the SfM cloud when sizing the grid"})
    n_points: int = field(default=300_000, metadata={"help": "Points sampled from the hull surface"})
    save_mesh: bool = field(default=True, metadata={"help": "Also write visual_hull.ply"})


def _cameras_from_model(rec) -> list[dict]:
    """Flatten a pycolmap reconstruction into per-image pose and intrinsic dictionaries."""
    cams = []
    for image in rec.images.values():
        camera = rec.cameras[image.camera_id]
        # Method in pycolmap 4.1, property in 4.2.
        cam_from_world = image.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        world_to_cam = cam_from_world.matrix()
        params = dict(zip(camera.params_info.split(", "), camera.params))
        fx = params.get("fx", params.get("f"))
        fy = params.get("fy", fx)
        cams.append(dict(
            name=image.name,
            R=np.asarray(world_to_cam[:, :3], np.float64),
            t=np.asarray(world_to_cam[:, 3], np.float64),
            fx=float(fx), fy=float(fy),
            cx=float(params.get("cx", camera.width / 2)),
            cy=float(params.get("cy", camera.height / 2)),
            W=int(camera.width), H=int(camera.height)))
    return cams


def _project(points, cam, torch):
    """Project world points into one camera, returning pixel coordinates and depth."""
    x_cam = points @ cam["R_t"].T + cam["t_t"]
    z = x_cam[:, 2]
    safe_z = z + 1e-9
    u = cam["fx"] * x_cam[:, 0] / safe_z + cam["cx"]
    v = cam["fy"] * x_cam[:, 1] / safe_z + cam["cy"]
    return u, v, z


@register_sfm
class VisualHullInit(SceneRefiner):
    """Carves a visual hull from masks and poses and writes it as the 2DGS init cloud."""

    name: ClassVar[str] = "hull"
    config_cls: ClassVar[type] = HullConfig

    def validate(self, scene: Scene) -> None:
        """Require images/, a sparse/0 model to take poses from, and masks to carve with."""
        super().validate(scene)
        if not scene.has_masks():
            raise SceneError(f"hull carving needs masks/, none at {scene.masks_dir}")

    def run(self, scene: Scene, output_dir: Path) -> SfMResult:
        """Carve the hull, colour a point cloud from it, and emit a ready-to-train scene.

        Args:
            scene: Input scene with images/, masks/, and a COLMAP model in sparse/0.
            output_dir: Directory to write the hull-initialised scene into.

        Returns:
            An SfMResult whose num_points is the size of the hull init cloud.

        Raises:
            SceneError: If no mask matches an image, or the carve collapses to an empty volume.
        """
        import open3d as o3d
        import pycolmap
        import torch
        from PIL import Image
        from skimage.measure import marching_cubes

        Image.MAX_IMAGE_PIXELS = None
        self.validate(scene)
        t0 = time.time()
        out_dir = output_dir.resolve()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        rec = pycolmap.Reconstruction(str(scene.sparse_dir))
        cams = _cameras_from_model(rec)
        logger.info(f"[hull] {len(cams)} registered cameras, device={device}")

        cams, masks = self._load_masks(scene, cams, device, torch, Image)
        if not cams:
            raise SceneError(f"no mask matched an image name under {scene.masks_dir}")

        lo, hi = self._grid_bounds(rec)
        logger.info(f"[hull] bbox lo={lo.round(4)} hi={hi.round(4)}")

        occupancy = self._carve(cams, masks, lo, hi, device, torch)
        hull = self._to_mesh(occupancy, lo, hi, o3d, marching_cubes)
        logger.info(f"[hull] mesh {len(hull.vertices)} verts, {len(hull.triangles)} faces")

        points, colours, coloured = self._sample_and_colour(
            hull, scene, cams, device, torch, Image, o3d)

        self._write_scene(scene, out_dir, points, colours, hull, o3d)
        elapsed = time.time() - t0
        logger.info(f"HULL_DONE {len(points)}pts ({coloured} coloured) "
                    f"in {elapsed:.0f}s -> {out_dir}")

        return SfMResult(
            output_dir=out_dir,
            elapsed=elapsed,
            scene=Scene(out_dir),
            num_images=len(cams),
            num_points=len(points),
            details={"hull_faces": len(hull.triangles), "coloured_points": coloured},
        )

    def _load_masks(self, scene, cams, device, torch, Image):
        """Load each camera's mask as a downsampled boolean tensor, dropping cameras without one."""
        step = self.config.mask_downscale
        kept, masks = [], []
        for cam in cams:
            path = scene.masks_dir / f"{Path(cam['name']).stem}.png"
            if not path.exists():
                continue
            mask = np.asarray(Image.open(path).convert("L"))
            if step > 1:
                mask = mask[::step, ::step]
            masks.append(torch.from_numpy(mask > 127).to(device))
            cam["R_t"] = torch.tensor(cam["R"], device=device, dtype=torch.float64)
            cam["t_t"] = torch.tensor(cam["t"], device=device, dtype=torch.float64)
            kept.append(cam)
        logger.info(f"[hull] matched {len(kept)}/{len(cams)} masks")
        return kept, masks

    def _grid_bounds(self, rec):
        """Size the voxel grid from the SfM cloud, trimming outliers so floaters cannot inflate it."""
        xyz = np.array([p.xyz for p in rec.points3D.values()], np.float64)
        if len(xyz) == 0:
            raise SceneError("SfM model has no 3D points to size the hull grid from")
        q = self.config.bbox_percentile
        lo = np.percentile(xyz, q, axis=0)
        hi = np.percentile(xyz, 100 - q, axis=0)
        centre, radius = (lo + hi) / 2, (hi - lo) / 2 * (1 + self.config.pad)
        return centre - radius, centre + radius

    def _carve(self, cams, masks, lo, hi, device, torch):
        """Intersect the silhouette cones over a voxel grid, by majority vote rather than strictly."""
        n = self.config.res
        axes = [torch.linspace(float(lo[i]), float(hi[i]), n, device=device) for i in range(3)]
        grid = torch.meshgrid(*axes, indexing="ij")
        points = torch.stack([g.reshape(-1) for g in grid], 1).double()
        votes = torch.zeros(points.shape[0], dtype=torch.int16, device=device)
        seen = torch.zeros(points.shape[0], dtype=torch.int16, device=device)

        for cam, mask in zip(cams, masks):
            mh, mw = mask.shape
            scale = cam["W"] / mw
            u, v, z = _project(points, cam, torch)
            ui = (u / scale).long()
            vi = (v / scale).long()
            in_view = (z > 0) & (ui >= 0) & (ui < mw) & (vi >= 0) & (vi < mh)
            seen += in_view.to(torch.int16)
            sel = in_view.nonzero(as_tuple=True)[0]
            inside = torch.zeros_like(in_view)
            inside[sel] = mask[vi[sel], ui[sel]]
            votes += inside.to(torch.int16)

        ratio = votes.float() / seen.float().clamp(min=1)
        occupied = (seen.float() >= self.config.min_seen_frac * len(cams)) & (ratio >= self.config.tau)
        count = int(occupied.sum())
        logger.info(f"[hull] occupied voxels {count}/{points.shape[0]}")
        if count < 8:
            raise SceneError(
                f"hull carve collapsed to {count} voxels; --tau {self.config.tau} is likely "
                f"too strict, or the masks do not correspond to these poses")
        return occupied.reshape(n, n, n).cpu().numpy().astype(np.float32)

    def _to_mesh(self, occupancy, lo, hi, o3d, marching_cubes):
        """Marching-cubes the occupancy volume and keep its largest connected component."""
        spacing = (hi - lo) / (self.config.res - 1)
        verts, faces, _, _ = marching_cubes(occupancy, level=0.5, spacing=tuple(spacing))
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts + lo),
            o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()
        labels, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        if len(counts) > 1:
            mesh.remove_triangles_by_mask(np.asarray(labels) != counts.argmax())
            mesh.remove_unreferenced_vertices()
        return mesh

    def _sample_and_colour(self, hull, scene, cams, device, torch, Image, o3d):
        """Sample the hull surface and colour each point from the cameras facing it."""
        pcd = hull.sample_points_uniformly(
            number_of_points=self.config.n_points, use_triangle_normal=True)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        pt = torch.from_numpy(points).to(device).double()
        nt = torch.from_numpy(normals).to(device).double()
        accum = torch.zeros(len(points), 3, device=device)
        weights = torch.zeros(len(points), device=device)
        step = self.config.img_downscale

        for cam in cams:
            path = scene.images_dir / cam["name"]
            if not path.exists():
                continue
            image = np.asarray(Image.open(path).convert("RGB"))[::step, ::step]
            ih, iw, _ = image.shape
            tex = torch.from_numpy(image).to(device).float() / 255.0
            scale = cam["W"] / iw
            u, v, z = _project(pt, cam, torch)
            ui = (u / scale).long()
            vi = (v / scale).long()
            in_view = (z > 0) & (ui >= 0) & (ui < iw) & (vi >= 0) & (vi < ih)
            centre = -(cam["R_t"].T @ cam["t_t"])
            view = centre - pt
            view = view / (view.norm(dim=1, keepdim=True) + 1e-9)
            # Facing angle stands in for a visibility test; training refines the colours anyway.
            facing = (nt * view).sum(1).clamp(min=0.0) * in_view.double()
            idx = (facing > 1e-3).nonzero(as_tuple=True)[0]
            accum[idx] += tex[vi[idx], ui[idx]] * facing[idx, None].float()
            weights[idx] += facing[idx].float()

        rgb = (accum / weights.clamp(min=1e-6)[:, None]).clamp(0, 1)
        rgb[weights < 1e-6] = 0.5
        coloured = int((weights > 1e-6).sum())
        return points.astype(np.float32), (rgb.cpu().numpy() * 255).astype(np.uint8), coloured

    def _write_scene(self, scene, out_dir, points, colours, hull, o3d):
        """Link the media, copy the model unchanged, and write the hull cloud as points3D.ply."""
        from plyfile import PlyData, PlyElement

        sparse_out = out_dir / "sparse" / "0"
        sparse_out.mkdir(parents=True, exist_ok=True)
        for name, source in (("images", scene.images_dir), ("masks", scene.masks_dir)):
            link = out_dir / name
            if source.is_dir() and not link.exists():
                os.symlink(source, link)
        for name in _MODEL_FILES:
            src = scene.sparse_dir / name
            if src.exists():
                shutil.copy2(src, sparse_out / name)

        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                 ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        elements = np.empty(len(points), dtype=dtype)
        elements[:] = list(map(tuple, np.concatenate(
            [points, np.zeros_like(points), colours], 1)))
        PlyData([PlyElement.describe(elements, "vertex")]).write(str(sparse_out / "points3D.ply"))

        if self.config.save_mesh:
            o3d.io.write_triangle_mesh(str(out_dir / "visual_hull.ply"), hull)
