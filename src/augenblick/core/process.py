"""The single subprocess helper shared by every backend wrapper."""
import logging
import os
import subprocess
from pathlib import Path

from augenblick.core.errors import BackendError

logger = logging.getLogger(__name__)


def run(cmd: list[str], cwd: str | Path | None = None) -> None:
    """Run a backend command, streaming its output, raising BackendError on failure.

    Args:
        cmd: Full argv of the command to run.
        cwd: Working directory, or None to inherit the caller's.

    Raises:
        BackendError: If the command exits non-zero, carrying its return code.
    """
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"  cwd: {cwd or os.getcwd()}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        raise BackendError(
            f"command failed with return code {result.returncode}: {' '.join(cmd)}",
            result.returncode,
        )
