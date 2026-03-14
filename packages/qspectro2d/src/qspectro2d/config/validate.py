"""Compatibility validation API.

This module provides the target-style import path while delegating to the
existing validation implementation.
"""

from __future__ import annotations

from .validation import validate, validate_defaults

# Alias used by the newer API naming.
validate_config = validate

__all__ = [
    "validate",
    "validate_config",
    "validate_defaults",
]
