import numpy as np


def phase_cycle_component(phases, P_grid, lm):
    """
    phases: uniform grid on [0, 2π)
    P_grid: (len(phases)~[phi1], len(phases)~[phi2], T)
    lm    : (l, m)
    returns: (T,) ≈ ∬ e^{-i(l φ1 + m φ2)} P(t; φ1, φ2) dφ1 dφ2
    """
    l, m = lm
    phases = np.asarray(phases)
    L, M, T = P_grid.shape
    assert L == M == len(phases)

    dphi = np.diff(phases).mean()
    u1 = np.exp(-1j * l * phases)  # corresponds to φ1
    u2 = np.exp(-1j * m * phases)  # corresponds to φ2

    # start with zeros
    P_out = np.zeros(T, dtype=complex)

    # explicit double summation
    for i in range(L):
        for k in range(M):
            P_out += u1[i] * u2[k] * P_grid[i, k, :]

    return P_out * dphi * dphi
