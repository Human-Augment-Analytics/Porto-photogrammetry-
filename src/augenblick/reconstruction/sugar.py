"""SuGaR: a nested vanilla-3DGS training run followed by SuGaR coarse/mesh/refine/texture."""
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Optional

from augenblick.core.registry import register_reconstruction
from augenblick.core.scene import Scene
from augenblick.reconstruction.base import LIBS_DIR, Stage, SubprocessBackend

logger = logging.getLogger(__name__)

SUGAR_DIR = LIBS_DIR / "sugar"
GS_TRAIN_SCRIPT = SUGAR_DIR / "gaussian_splatting" / "train.py"
SUGAR_TRAIN_SCRIPT = SUGAR_DIR / "train.py"


@dataclass(frozen=True)
class SugarConfig:
    """Vanilla-3DGS and SuGaR parameters."""

    gs_iterations: int = field(default=20_000, metadata={"help": "Vanilla 3DGS training iterations"})
    gs_densify_grad_threshold: float = field(default=0.0002, metadata={
        "help": "Lower → denser Gaussian cloud (more detail, more memory)"})
    gs_densify_until_iter: int = field(default=15_000, metadata={
        "help": "Iteration to stop densifying; extend alongside --gs_iterations"})
    gs_lambda_dssim: float = field(default=0.2, metadata={"help": "SSIM loss weight for 3DGS"})
    gs_sh_degree: int = field(default=3, metadata={"help": "Max spherical harmonics degree"})
    iteration_to_load: int = field(default=7_000, metadata={"help": "3DGS iteration to load for SuGaR"})
    regularization: Literal["sdf", "density", "dn_consistency"] = field(
        default="dn_consistency", metadata={"help": "Coarse SuGaR regularization type"})
    surface_level: float = field(default=0.1, metadata={"help": "Isosurface level for mesh extraction"})
    n_vertices: int = field(default=1_000_000, metadata={"help": "Target vertex count"})
    gaussians_per_triangle: int = field(default=1, metadata={"help": "Gaussians per mesh triangle"})
    refinement_iterations: int = field(default=15_000, metadata={"help": "Refinement training iterations"})
    low_poly: bool = field(default=False, metadata={"help": "200k vertices, 6 gaussians/triangle"})
    high_poly: bool = field(default=False, metadata={"help": "1M vertices, 1 gaussian/triangle"})
    refinement_time: Optional[Literal["short", "medium", "long"]] = field(
        default=None, metadata={"help": "Preset refinement duration (2k/7k/15k iterations)"})
    square_size: int = field(default=4, metadata={
        "help": "UV texture square size (larger → finer baked texture)"})
    postprocess_mesh: bool = field(default=False, metadata={
        "help": "Remove low-density border triangles (risky; can help single-sided objects)"})
    white_background: bool = field(default=False, metadata={"help": "Use white background"})
    gpu: int = field(default=0, metadata={"help": "GPU device index"})


@register_reconstruction
class SugarBackend(SubprocessBackend):
    """Runs vanilla 3DGS, then SuGaR's combined coarse/mesh/refine/texture script."""

    name: ClassVar[str] = "sugar"
    config_cls: ClassVar[type] = SugarConfig
    backend_dir: ClassVar[Path] = SUGAR_DIR
    title: ClassVar[str] = "SuGaR Reconstruction Pipeline"

    def __init__(self, config):
        super().__init__(config)
        self._scene_name = ""

    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        """Build the 3DGS and SuGaR invocations; SuGaR takes booleans as string arguments."""
        c = self.config
        scene_dir = scene.root
        self._scene_name = scene_dir.name
        gs_model_dir = output_dir / "gs_model"
        sugar_output_dir = output_dir / "sugar"

        gs_cmd = [
            sys.executable, str(GS_TRAIN_SCRIPT),
            "-s", str(scene_dir),
            "-m", str(gs_model_dir),
            "--iterations", str(c.gs_iterations),
            "--densify_grad_threshold", str(c.gs_densify_grad_threshold),
            "--densify_until_iter", str(c.gs_densify_until_iter),
            "--lambda_dssim", str(c.gs_lambda_dssim),
            "--sh_degree", str(c.gs_sh_degree),
        ]

        sugar_cmd = [
            sys.executable, str(SUGAR_TRAIN_SCRIPT),
            "-s", str(scene_dir),
            "-c", str(gs_model_dir),
            "-o", str(sugar_output_dir),
            "-i", str(c.iteration_to_load),
            "-r", c.regularization,
            "-l", str(c.surface_level),
            "-v", str(c.n_vertices),
            "-g", str(c.gaussians_per_triangle),
            "-f", str(c.refinement_iterations),
            "--square_size", str(c.square_size),
            "--eval", "False",
            "--gpu", str(c.gpu),
        ]
        if c.postprocess_mesh:
            sugar_cmd += ["--postprocess_mesh", "True"]
        if c.low_poly:
            sugar_cmd += ["--low_poly", "True"]
        if c.high_poly:
            sugar_cmd += ["--high_poly", "True"]
        if c.refinement_time:
            sugar_cmd += ["--refinement_time", c.refinement_time]
        if c.white_background:
            sugar_cmd += ["--white_background", "True"]

        return [
            Stage("3DGS training", gs_cmd),
            Stage("SuGaR training", sugar_cmd),
        ]

    def mesh_path(self, output_dir: Path) -> Path:
        """Refined-mesh directory; upstream names it after the scene dir, not the output dir."""
        return output_dir / "sugar" / "refined_mesh" / self._scene_name

    def footer(self, output_dir: Path) -> dict[str, object]:
        """SuGaR writes several artifacts, so report its output root alongside the mesh dir."""
        return {"Output": output_dir / "sugar", "Refined mesh": self.mesh_path(output_dir)}
