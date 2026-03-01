"""Paper-specific time independent Liouvillian builders. for paper_eqs solver."""

from __future__ import annotations

import numpy as np
from qutip import Qobj, stacked_index

from .simulation_class import SimulationModuleOQS
from ..laser_system.laser_fcts import e_pulses
from ...config.atomic_system import DEPH_RATE_FS, DOWN_RATE_FS, UP_RATE_FS

__all__ = [
    "matrix_ODE_paper",
]


def _paper_liouvillian_context(sim_oqs: SimulationModuleOQS) -> dict:
    """Build and cache static Liouvillian pieces for fast runtime assembly.

    Returns a dict with
        - "L0": static (field-independent) matrix
        - "L_plus": coefficient multiplying E_plus(t)
        - "L_minus": coefficient multiplying E_minus(t)=conj(E_plus)
        - "dims": qutip dims for Qobj wrapping
    """
    cached = getattr(sim_oqs, "_paper_liouvillian_ctx", None)
    if cached is not None:
        return cached

    n_atoms = sim_oqs.system.n_atoms
    if n_atoms == 1:
        size = 2
        pulse_seq = sim_oqs.laser
        mu = sim_oqs.system.dip_moments[0]
        detuning = sim_oqs.system.frequencies_fs[0] - pulse_seq.carrier_freq_fs

        deph_rate = DEPH_RATE_FS
        down_rate = DOWN_RATE_FS
        up_rate = UP_RATE_FS
        deph_rate_tot = deph_rate + 0.5 * (down_rate + up_rate)

        idx_gg = stacked_index(size, 0, 0)
        idx_ge = stacked_index(size, 0, 1)
        idx_eg = stacked_index(size, 1, 0)
        idx_ee = stacked_index(size, 1, 1)

        L0 = np.zeros((size * size, size * size), dtype=complex)
        L_plus = np.zeros_like(L0)
        L_minus = np.zeros_like(L0)

        # Static population terms
        L0[idx_gg, idx_gg] = -up_rate
        L0[idx_gg, idx_ee] = down_rate
        L0[idx_ee, idx_gg] = up_rate
        L0[idx_ee, idx_ee] = -down_rate

        # Static coherence terms
        L0[idx_eg, idx_eg] = -deph_rate_tot - 1j * detuning
        L0[idx_ge, idx_ge] = -deph_rate_tot + 1j * detuning

        # E_plus terms
        L_plus[idx_gg, idx_ge] = -1j * mu
        L_plus[idx_ee, idx_ge] = +1j * mu
        L_plus[idx_eg, idx_gg] = +1j * mu
        L_plus[idx_eg, idx_ee] = -1j * mu

        # E_minus terms
        L_minus[idx_gg, idx_eg] = +1j * mu
        L_minus[idx_ee, idx_eg] = -1j * mu
        L_minus[idx_ge, idx_gg] = -1j * mu
        L_minus[idx_ge, idx_ee] = +1j * mu

        ctx = {
            "L0": L0,
            "L_plus": L_plus,
            "L_minus": L_minus,
            "dims": [[[size], [size]], [[size], [size]]],
        }
        setattr(sim_oqs, "_paper_liouvillian_ctx", ctx)
        return ctx

    if n_atoms == 2:
        size = 4
        omega_laser = sim_oqs.laser.carrier_freq_fs
        dip_op = sim_oqs.system.to_eigenbasis(sim_oqs.system.dipole_op)

        # Precompute all static rates once
        gamma_12 = sim_oqs.sb_coupling.paper_gamma_ab(1, 2)
        gamma_21 = sim_oqs.sb_coupling.paper_gamma_ab(2, 1)
        Gamma_10 = sim_oqs.sb_coupling.paper_Gamma_ab(1, 0)
        Gamma_20 = sim_oqs.sb_coupling.paper_Gamma_ab(2, 0)
        Gamma_30 = sim_oqs.sb_coupling.paper_Gamma_ab(3, 0)
        Gamma_12 = sim_oqs.sb_coupling.paper_Gamma_ab(1, 2)
        Gamma_31 = sim_oqs.sb_coupling.paper_Gamma_ab(3, 1)
        Gamma_32 = sim_oqs.sb_coupling.paper_Gamma_ab(3, 2)
        Gamma_11 = sim_oqs.sb_coupling.paper_Gamma_ab(1, 1)
        Gamma_22 = sim_oqs.sb_coupling.paper_Gamma_ab(2, 2)

        w10 = sim_oqs.system.omega_ij(1, 0)
        w20 = sim_oqs.system.omega_ij(2, 0)
        w30 = sim_oqs.system.omega_ij(3, 0)
        w12 = sim_oqs.system.omega_ij(1, 2)
        w31 = sim_oqs.system.omega_ij(3, 1)
        w32 = sim_oqs.system.omega_ij(3, 2)

        # Indices
        idx_00 = stacked_index(size, 0, 0)
        idx_01 = stacked_index(size, 0, 1)
        idx_02 = stacked_index(size, 0, 2)
        idx_03 = stacked_index(size, 0, 3)
        idx_10 = stacked_index(size, 1, 0)
        idx_11 = stacked_index(size, 1, 1)
        idx_12 = stacked_index(size, 1, 2)
        idx_13 = stacked_index(size, 1, 3)
        idx_20 = stacked_index(size, 2, 0)
        idx_21 = stacked_index(size, 2, 1)
        idx_22 = stacked_index(size, 2, 2)
        idx_23 = stacked_index(size, 2, 3)
        idx_30 = stacked_index(size, 3, 0)
        idx_31 = stacked_index(size, 3, 1)
        idx_32 = stacked_index(size, 3, 2)
        idx_33 = stacked_index(size, 3, 3)

        L0 = np.zeros((size * size, size * size), dtype=complex)
        L_plus = np.zeros_like(L0)
        L_minus = np.zeros_like(L0)

        # 1) One-excitation coherences (static diagonals)
        term = -1j * (w10 - omega_laser) - Gamma_10
        L0[idx_10, idx_10] = term
        L0[idx_01, idx_01] = np.conj(term)

        term = -1j * (w20 - omega_laser) - Gamma_20
        L0[idx_20, idx_20] = term
        L0[idx_02, idx_02] = np.conj(term)

        # 2) Double-excited coherences
        term = -1j * (w30 - 2 * omega_laser) - Gamma_30
        L0[idx_30, idx_30] = term
        L0[idx_03, idx_03] = np.conj(term)

        # 3) Cross-coherences
        term = -1j * w12 - Gamma_12
        L0[idx_12, idx_12] = term
        L0[idx_21, idx_21] = np.conj(term)

        term = -1j * (w31 - omega_laser) - Gamma_31
        L0[idx_31, idx_31] = term
        L0[idx_13, idx_13] = np.conj(term)

        term = -1j * (w32 - omega_laser) - Gamma_32
        L0[idx_32, idx_32] = term
        L0[idx_23, idx_23] = np.conj(term)

        # 4) Static population transfer terms
        L0[idx_11, idx_11] = -Gamma_11
        L0[idx_11, idx_22] = gamma_12
        L0[idx_22, idx_22] = -Gamma_22
        L0[idx_22, idx_11] = gamma_21

        # Field-coupled terms (linear in E_plus / E_minus)
        # -- idx_10 row
        L_plus[idx_10, idx_00] = +1j * dip_op[1, 0]
        L_plus[idx_10, idx_11] = -1j * dip_op[1, 0]
        L_plus[idx_10, idx_12] = -1j * dip_op[2, 0]
        L_minus[idx_10, idx_30] = +1j * dip_op[3, 1]
        # -- idx_01 row
        L_minus[idx_01, idx_00] = np.conj(L_plus[idx_10, idx_00])
        L_minus[idx_01, idx_11] = np.conj(L_plus[idx_10, idx_11])
        L_minus[idx_01, idx_21] = np.conj(L_plus[idx_10, idx_12])
        L_plus[idx_01, idx_03] = np.conj(L_minus[idx_10, idx_30])

        # -- idx_20 row
        L_plus[idx_20, idx_00] = +1j * dip_op[2, 0]
        L_plus[idx_20, idx_22] = -1j * dip_op[2, 0]
        L_plus[idx_20, idx_21] = -1j * dip_op[1, 0]
        L_minus[idx_20, idx_30] = +1j * dip_op[3, 2]
        # -- idx_02 row
        L_minus[idx_02, idx_00] = np.conj(L_plus[idx_20, idx_00])
        L_minus[idx_02, idx_22] = np.conj(L_plus[idx_20, idx_22])
        L_minus[idx_02, idx_12] = np.conj(L_plus[idx_20, idx_21])
        L_plus[idx_02, idx_03] = np.conj(L_minus[idx_20, idx_30])

        # -- idx_30 row
        L_plus[idx_30, idx_10] = +1j * dip_op[3, 1]
        L_plus[idx_30, idx_20] = +1j * dip_op[3, 2]
        L_plus[idx_30, idx_31] = -1j * dip_op[1, 0]
        L_plus[idx_30, idx_32] = -1j * dip_op[2, 0]
        # -- idx_03 row
        L_minus[idx_03, idx_01] = np.conj(L_plus[idx_30, idx_10])
        L_minus[idx_03, idx_02] = np.conj(L_plus[idx_30, idx_20])
        L_minus[idx_03, idx_13] = np.conj(L_plus[idx_30, idx_31])
        L_minus[idx_03, idx_23] = np.conj(L_plus[idx_30, idx_32])

        # -- idx_12 row
        L_plus[idx_12, idx_02] = +1j * dip_op[1, 0]
        L_plus[idx_12, idx_13] = -1j * dip_op[3, 2]
        L_minus[idx_12, idx_32] = +1j * dip_op[3, 1]
        L_minus[idx_12, idx_10] = -1j * dip_op[2, 0]
        # -- idx_21 row
        L_minus[idx_21, idx_20] = np.conj(L_plus[idx_12, idx_02])
        L_minus[idx_21, idx_31] = np.conj(L_plus[idx_12, idx_13])
        L_plus[idx_21, idx_23] = np.conj(L_minus[idx_12, idx_32])
        L_plus[idx_21, idx_01] = np.conj(L_minus[idx_12, idx_10])

        # -- idx_31 row
        L_plus[idx_31, idx_11] = +1j * dip_op[3, 1]
        L_plus[idx_31, idx_21] = +1j * dip_op[3, 2]
        L_minus[idx_31, idx_30] = -1j * dip_op[1, 0]
        # -- idx_13 row
        L_minus[idx_13, idx_11] = np.conj(L_plus[idx_31, idx_11])
        L_minus[idx_13, idx_12] = np.conj(L_plus[idx_31, idx_21])
        L_plus[idx_13, idx_03] = np.conj(L_minus[idx_31, idx_30])

        # -- idx_32 row
        L_plus[idx_32, idx_22] = +1j * dip_op[3, 2]
        L_plus[idx_32, idx_12] = +1j * dip_op[3, 1]
        L_minus[idx_32, idx_30] = -1j * dip_op[2, 0]
        # -- idx_23 row
        L_minus[idx_23, idx_22] = np.conj(L_plus[idx_32, idx_22])
        L_minus[idx_23, idx_21] = np.conj(L_plus[idx_32, idx_12])
        L_plus[idx_23, idx_03] = np.conj(L_minus[idx_32, idx_30])

        # 4) Population rows with driving
        L_plus[idx_00, idx_01] = -1j * dip_op[1, 0]
        L_plus[idx_00, idx_02] = -1j * dip_op[2, 0]
        L_minus[idx_00, idx_10] = +1j * dip_op[1, 0]
        L_minus[idx_00, idx_20] = +1j * dip_op[2, 0]

        L_plus[idx_11, idx_01] = +1j * dip_op[1, 0]
        L_plus[idx_11, idx_13] = -1j * dip_op[3, 1]
        L_minus[idx_11, idx_31] = +1j * dip_op[3, 1]
        L_minus[idx_11, idx_10] = -1j * dip_op[1, 0]

        L_plus[idx_22, idx_02] = +1j * dip_op[2, 0]
        L_plus[idx_22, idx_23] = -1j * dip_op[3, 2]
        L_minus[idx_22, idx_32] = +1j * dip_op[3, 2]
        L_minus[idx_22, idx_20] = -1j * dip_op[2, 0]

        # Enforce trace conservation row-by-row for static and field components
        L0[idx_33, :] = -L0[idx_00, :] - L0[idx_11, :] - L0[idx_22, :]
        L_plus[idx_33, :] = -L_plus[idx_00, :] - L_plus[idx_11, :] - L_plus[idx_22, :]
        L_minus[idx_33, :] = -L_minus[idx_00, :] - L_minus[idx_11, :] - L_minus[idx_22, :]

        ctx = {
            "L0": L0,
            "L_plus": L_plus,
            "L_minus": L_minus,
            "dims": [[[size], [size]], [[size], [size]]],
        }
        setattr(sim_oqs, "_paper_liouvillian_ctx", ctx)
        return ctx

    raise ValueError("Only n_atoms=1 or 2 are supported.")


def matrix_ODE_paper(t: float, sim_oqs: SimulationModuleOQS) -> Qobj:
    """Dispatcher returning the time dependent Liouvillian L(t).
    base on the papers:
    https://pubs.aip.org/jcp/article/124/23/234504/930650
    https://pubs.aip.org/jcp/article/124/23/234505/930637

    Chooses implementation based on number of atoms (1 or 2). For other sizes
    a ValueError is raised.
    """
    ctx = _paper_liouvillian_context(sim_oqs)
    E_plus = e_pulses(t, sim_oqs.laser)
    E_minus = np.conj(E_plus)
    L = ctx["L0"] + E_plus * ctx["L_plus"] + E_minus * ctx["L_minus"]
    return Qobj(L, dims=ctx["dims"])