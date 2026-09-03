"""Banner, per-stage headers, and timing summaries matching the original wrappers' logs."""
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

RULE = "=" * 60


class StageTimer:
    """Log a banner, per-stage headers, and a per-stage + total timing summary.

    Args:
        title: Pipeline name shown in the banner.
        total_stages: Number of stages, used for the "Step i/N" headers.
        header: Key/value lines printed under the banner.
    """

    def __init__(self, title: str, total_stages: int, header: dict[str, object]):
        self.title = title
        self.total_stages = total_stages
        self.elapsed: dict[str, float] = {}
        self._index = 0
        logger.info(RULE)
        logger.info(title)
        for key, value in header.items():
            logger.info(f"  {key}: {value}")
        logger.info(RULE)

    @contextmanager
    def stage(self, name: str):
        """Time one stage, logging its header on entry and its duration on exit."""
        self._index += 1
        logger.info(f"Step {self._index}/{self.total_stages}: {name}")
        t0 = time.time()
        yield
        elapsed = time.time() - t0
        self.elapsed[name] = elapsed
        logger.info(f"{name} completed in {elapsed:.1f}s")

    @property
    def total(self) -> float:
        """Total time across all completed stages."""
        return sum(self.elapsed.values())

    def summary(self, footer: dict[str, object]) -> None:
        """Log the closing block: per-stage times, the total, then the footer lines."""
        logger.info(RULE)
        logger.info("Pipeline complete")
        for name, elapsed in self.elapsed.items():
            logger.info(f"  {name} time: {elapsed:.1f}s")
        logger.info(f"  Total: {self.total:.1f}s")
        for key, value in footer.items():
            logger.info(f"  {key}: {value}")
        logger.info(RULE)
