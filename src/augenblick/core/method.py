"""The Method ABC that every SfM and reconstruction stage implements."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.config import config_from_namespace
from augenblick.core.scene import Scene

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """What a completed stage produced, for logging and for chaining stages.

    Args:
        output_dir: Directory the stage wrote its results into.
        elapsed: Wall-clock seconds the stage took.
        details: Backend-specific extras, such as mesh paths or point counts.
    """

    output_dir: Path
    elapsed: float
    details: dict[str, object] = field(default_factory=dict)


class Method(ABC):
    """Base for any pipeline stage that transforms a scene directory.

    Subclasses set `name` and `config_cls`, then implement `run`. Registration is
    performed by the register_sfm / register_reconstruction decorators.
    """

    name: ClassVar[str]
    config_cls: ClassVar[type]
    accepts_passthrough: ClassVar[bool] = False

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_namespace(cls, ns):
        """Build the method from parsed CLI arguments."""
        return cls(config_from_namespace(cls.config_cls, ns))

    def validate(self, scene: Scene) -> None:
        """Raise SceneError if the scene lacks what this method requires."""
        scene.require_images()

    @abstractmethod
    def run(self, scene: Scene, output_dir: Path) -> StageResult:
        """Execute the stage.

        Args:
            scene: Input scene to consume.
            output_dir: Directory to write results into.

        Returns:
            A StageResult describing what was produced.
        """
