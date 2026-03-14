"""Thin public config API built around one merged config object."""

from __future__ import annotations

from .io import load_config
from .validate import validate_config, validate_defaults

__all__ = [
  "load_config",
  "validate_config",
  "validate_defaults",
]
