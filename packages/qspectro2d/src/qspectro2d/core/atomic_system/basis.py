"""Basis indexing helpers for truncated excitation manifolds."""

from __future__ import annotations

import math


def pair_to_index(i: int, j: int, n_atoms: int) -> int:
    """Return the basis index for the double excitation |i,j| with 1-based sites."""
    if not 1 <= i < j <= n_atoms:
        raise ValueError(f"Expected 1 <= i < j <= {n_atoms}, got ({i}, {j})")
    return n_atoms + math.comb(j - 1, 2) + i


def excitation_number_from_index(index: int, n_atoms: int) -> int:
    """Return 0 for ground, 1 for singles, and 2 for doubles."""
    if index == 0:
        return 0
    if 1 <= index <= n_atoms:
        return 1
    return 2


__all__ = ["excitation_number_from_index", "pair_to_index"]
