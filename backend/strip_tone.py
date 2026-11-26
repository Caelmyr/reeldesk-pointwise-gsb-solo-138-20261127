"""Compatibility namespace re-exporting :mod:`backend.strip_grade`."""

from .strip_grade import *  # noqa: F401,F403
from . import strip_grade as _implementation

__all__ = [name for name in dir(_implementation) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementation, name)
