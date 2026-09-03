"""Exception hierarchy shared by every pipeline stage."""


class AugenblickError(Exception):
    """Base for all first-party pipeline errors."""


class SceneError(AugenblickError):
    """A scene directory is missing something a stage requires."""


class BackendError(AugenblickError):
    """A backend subprocess exited non-zero.

    Args:
        message: Human-readable description of the failure.
        returncode: The subprocess exit code, so callers can propagate it.
    """

    def __init__(self, message: str, returncode: int):
        super().__init__(message)
        self.returncode = returncode


class MethodNotFound(AugenblickError):
    """A method name was not present in the registry."""
