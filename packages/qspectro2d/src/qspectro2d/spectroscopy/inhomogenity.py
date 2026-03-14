"""Compatibility shim. Real implementation lives in broadening.py."""

from .broadening import normalized_gauss, sample_from_gaussian

__all__ = ["normalized_gauss", "sample_from_gaussian"]
