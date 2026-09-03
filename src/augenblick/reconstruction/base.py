"""Shared shape of the reconstruction backends: validate, prepare, run staged subprocesses."""
import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from augenblick.core import process
from augenblick.core.method import Method, StageResult
from augenblick.core.scene import Scene
from augenblick.core.timing import StageTimer

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBS_DIR = REPO_ROOT / "src" / "libs"


@dataclass
class Stage:
    """One backend invocation: a label and the argv to run for it.

    Args:
        name: Label used in the "Step i/N" log header.
        cmd: Full argv passed to the subprocess.
    """

    name: str
    cmd: list[str]


class ReconstructionMethod(Method):
    """Consumes a COLMAP scene, produces a mesh."""

    def validate(self, scene: Scene) -> None:
        """Require both images/ and a non-empty sparse/0/ model."""
        scene.require_images()
        scene.require_reconstruction()

    @abstractmethod
    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        """Build the ordered list of backend invocations for this run.

        Args:
            scene: The scene the backend will read, after prepare().
            output_dir: Directory the backend writes into.

        Returns:
            The stages to run, in order.
        """

    @abstractmethod
    def mesh_path(self, output_dir: Path) -> Path:
        """Return where this backend writes its final mesh for the given output dir."""


class SubprocessBackend(ReconstructionMethod):
    """A reconstruction backend driven by running upstream scripts as subprocesses."""

    backend_dir: ClassVar[Path]
    use_cwd: ClassVar[bool] = True
    title: ClassVar[str]

    def prepare(self, scene: Scene, output_dir: Path) -> Scene:
        """Hook for backends needing a modified scene; default returns it unchanged."""
        return scene

    def header(self, scene: Scene, output_dir: Path) -> dict[str, object]:
        """Key/value lines logged under the banner; backends may extend this."""
        return {"Scene": scene.root, "Output": output_dir}

    def footer(self, output_dir: Path) -> dict[str, object]:
        """Key/value lines logged in the closing summary."""
        return {"Output": output_dir, "Mesh": self.mesh_path(output_dir)}

    def run(self, scene: Scene, output_dir: Path) -> StageResult:
        """Validate, prepare the scene, then run each stage under a timer.

        Args:
            scene: Input COLMAP scene.
            output_dir: Directory the backend writes into.

        Returns:
            A StageResult carrying the total elapsed time and the mesh path.
        """
        self.validate(scene)
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared = self.prepare(scene, output_dir)
        stages = self.stages(prepared, output_dir)

        timer = StageTimer(self.title, len(stages), self.header(prepared, output_dir))
        cwd = str(self.backend_dir) if self.use_cwd else None
        for stage in stages:
            with timer.stage(stage.name):
                process.run(stage.cmd, cwd=cwd)
        timer.summary(self.footer(output_dir))

        return StageResult(
            output_dir=output_dir,
            elapsed=timer.total,
            details={"mesh": self.mesh_path(output_dir), "stages": dict(timer.elapsed)},
        )
