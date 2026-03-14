"""Geometry helpers for atomic-system layouts and couplings."""

from __future__ import annotations

import numpy as np


def cylindrical_positions(n_atoms: int, n_chains: int, distance: float = 1.0) -> np.ndarray:
    """Return atom positions for a linear chain or cylindrical arrangement."""
    n_rings = n_atoms // n_chains
    if n_chains == 1:
        return np.array([[0.0, 0.0, z * distance] for z in range(n_rings)], dtype=float)

    dphi = 2.0 * np.pi / n_chains
    radius = distance / (2.0 * np.sin(np.pi / n_chains))
    ring_centers = np.array(
        [[radius * np.cos(k * dphi), radius * np.sin(k * dphi), 0.0] for k in range(n_chains)],
        dtype=float,
    )
    return np.array(
        [
            ring_centers[chain] + np.array([0.0, 0.0, ring * distance], dtype=float)
            for chain in range(n_chains)
            for ring in range(n_rings)
        ],
        dtype=float,
    )


def isotropic_coupling_matrix(
    positions: np.ndarray,
    dip_moments: list[float],
    base_coupling_fs: float,
    power: float = 3.0,
) -> np.ndarray:
    """Return the symmetric coupling matrix J_ij ~ coupling * mu_i * mu_j / r^power."""
    n_atoms = len(dip_moments)
    matrix = np.zeros((n_atoms, n_atoms), dtype=float)
    if n_atoms == 2:
        matrix[0, 1] = base_coupling_fs
        matrix[1, 0] = base_coupling_fs
        return matrix

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = float(np.linalg.norm(positions[j] - positions[i]))
            if distance == 0:
                raise ValueError("Duplicate positions encountered (zero distance).")
            coupling_ij = base_coupling_fs * dip_moments[i] * dip_moments[j] / (distance**power)
            matrix[i, j] = coupling_ij
            matrix[j, i] = coupling_ij
    return matrix


__all__ = ["cylindrical_positions", "isotropic_coupling_matrix"]