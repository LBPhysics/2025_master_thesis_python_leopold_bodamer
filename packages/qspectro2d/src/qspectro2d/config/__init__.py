"""Thin public config API built around one merged config object."""

from __future__ import annotations

from .config import resolve_config, validate_config

__all__ = [
    "resolve_config",
    "validate_config",
]
