"""Command-line entry point: `augenblick {sfm,recon} <method> --scene <dir> --output <dir>`."""
import argparse
import logging
import sys
from pathlib import Path

from augenblick.core.config import add_dataclass_arguments
from augenblick.core.errors import BackendError, MethodNotFound, SceneError
from augenblick.core.registry import (
    RECONSTRUCTION_REGISTRY,
    SFM_REGISTRY,
    get_reconstruction,
    get_sfm,
)
from augenblick.core.scene import Scene

logger = logging.getLogger(__name__)

# Importing the packages populates the registries the subparsers are built from.
import augenblick.reconstruction  # noqa: E402,F401
import augenblick.sfm  # noqa: E402,F401


def _add_stage_parser(subparsers, stage: str, registry: dict[str, type], help_text: str):
    """Build the parser for one stage, with a per-method subparser drawn from the registry."""
    parser = subparsers.add_parser(stage, help=help_text)
    parser.add_argument("--list", action="store_true", help="List available methods and exit")
    method_subs = parser.add_subparsers(dest="method")
    for name, cls in sorted(registry.items()):
        sub = method_subs.add_parser(name, help=cls.__doc__)
        sub.add_argument("--scene", type=Path, required=True, help="Input scene directory")
        sub.add_argument("--output", type=Path, required=True, help="Output directory")
        add_dataclass_arguments(sub, cls.config_cls)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the full CLI parser, one subparser per registered method."""
    parser = argparse.ArgumentParser(
        prog="augenblick",
        description="SfM initialisation and Gaussian-primitive surface reconstruction.",
    )
    subparsers = parser.add_subparsers(dest="stage")
    _add_stage_parser(subparsers, "sfm", SFM_REGISTRY, "Run an SfM method")
    _add_stage_parser(subparsers, "recon", RECONSTRUCTION_REGISTRY, "Run a reconstruction backend")
    return parser


def _list_methods(registry: dict[str, type]) -> int:
    """Print the registered method names with their one-line descriptions."""
    for name, cls in sorted(registry.items()):
        summary = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        print(f"{name:<12} {summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, resolve the method, and run it.

    Args:
        argv: Argument list, defaulting to sys.argv[1:].

    Returns:
        A process exit code: 2 for scene/lookup errors, a backend's own code on failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    # GW forwards unknown flags to its training step, so parse leniently and gate below.
    args, extras = parser.parse_known_args(argv)

    if args.stage is None:
        parser.print_help()
        return 2

    registry = SFM_REGISTRY if args.stage == "sfm" else RECONSTRUCTION_REGISTRY
    if getattr(args, "list", False):
        return _list_methods(registry)
    if args.method is None:
        parser.parse_args([args.stage, "--help"], argv)
        return 2

    try:
        cls = get_sfm(args.method) if args.stage == "sfm" else get_reconstruction(args.method)
    except MethodNotFound as exc:
        logger.error(str(exc))
        return 2

    if extras and not cls.accepts_passthrough:
        logger.error(f"unrecognised arguments: {' '.join(extras)}")
        return 2

    method = cls.from_namespace(args, extras) if cls.accepts_passthrough else cls.from_namespace(args)

    try:
        method.run(Scene(args.scene.resolve()), args.output.resolve())
    except SceneError as exc:
        logger.error(str(exc))
        return 2
    except BackendError as exc:
        logger.error(str(exc))
        return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
