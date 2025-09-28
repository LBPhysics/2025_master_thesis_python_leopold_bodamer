# TODO the rates of this implementation are currently time-dependent BUT NOT WORKING!!!!

"""Paper-specific time dependent Liouvillian builders."""
from __future__ import annotations

import numpy as np
from qutip import Qobj, stacked_index

from .simulation_class import SimulationModuleOQS
from ..laser_system.laser_fcts import e_pulses


__all__ = [
    "matrix_ODE_paper",
]


def matrix_ODE_paper(t: float, sim_oqs: SimulationModuleOQS) -> Qobj:
    """Dispatcher returning the time dependent Liouvillian L(t).
    base on the papers:
    https://pubs.aip.org/jcp/article/124/23/234504/930650
    https://pubs.aip.org/jcp/article/124/23/234505/930637

    Chooses implementation based on number of atoms (1 or 2). For other sizes
    a ValueError is raised.
    """
    n_atoms = sim_oqs.system.n_atoms
    if n_atoms == 1:
        return _matrix_ODE_paper_1atom(t, sim_oqs)
    if n_atoms == 2:
        return _matrix_ODE_paper_2atom(t, sim_oqs)
    raise ValueError("Only n_atoms=1 or 2 are supported.")


def _matrix_ODE_paper_1atom(t: float, sim_oqs: SimulationModuleOQS) -> Qobj:
    """Liouvillian L(t) for a single two-level system including driving & bath.
    drho/dt = L(t) * vec(rho)
    """

    pulse_seq = sim_oqs.laser
    E_RWA_plus = e_pulses(t, pulse_seq)
    E_RWA_minus = np.conj(E_RWA_plus)
    mu = sim_oqs.system.dip_moments[0]  # dipole op looks the same in both bases
    """
    w0 = sim_oqs.system.frequencies_fs[0]
    wL = pulse_seq.carrier_freq_fs
    from qspectro2d.core.bath_system.bath_fcts import bath_to_rates
    #deph_rate_pure = bath_to_rates(sim_oqs.bath, mode="deph")
    #down_rate, up_rate = bath_to_rates(sim_oqs.bath, w0, mode="decay")
    # Dephasing rate (assumed identical structure for all single excitations)
    deph_rate_tot = deph_rate_pure + 0.5 * (down_rate + up_rate)
    """

    # Dephasing rate (assumed identical structure for all single excitations)
    deph_rate = 1 / 100
    down_rate = 1 / 300
    up_rate = 0.0
    deph_rate_tot = deph_rate + 0.5 * (down_rate + up_rate)
    size = 2
    idx_gg = stacked_index(size, 0, 0)
    idx_ge = stacked_index(size, 0, 1)
    idx_eg = stacked_index(size, 1, 0)
    idx_ee = stacked_index(size, 1, 1)

    L = np.zeros((4, 4), dtype=complex)

    # Populations
    L[idx_gg, idx_gg] = -up_rate
    L[idx_gg, idx_ee] = down_rate
    L[idx_ee, idx_gg] = up_rate
    L[idx_ee, idx_ee] = -down_rate
    # Driving contributions to populations
    L[idx_gg, idx_eg] = +1j * E_RWA_minus * mu
    L[idx_gg, idx_ge] = -1j * E_RWA_plus * mu
    L[idx_ee, idx_ge] = +1j * E_RWA_plus * mu
    L[idx_ee, idx_eg] = -1j * E_RWA_minus * mu

    # Coherences
    L[idx_eg, idx_gg] = +1j * E_RWA_plus * mu
    L[idx_eg, idx_ee] = -1j * E_RWA_plus * mu
    L[idx_eg, idx_eg] = -deph_rate_tot
    L[idx_ge, idx_gg] = -1j * E_RWA_minus * mu
    L[idx_ge, idx_ee] = +1j * E_RWA_minus * mu
    L[idx_ge, idx_ge] = -deph_rate_tot

    return Qobj(L, dims=[[[size], [size]], [[size], [size]]])


'''
def _matrix_ODE_paper_2atom(t: float, sim_oqs: SimulationModuleOQS) -> Qobj:
    """Column-stacked Liouvillian for a coupled dimer (n_atoms=2)."""
    pulse_seq = sim_oqs.laser
    E_RWA_plus = e_pulses(t, pulse_seq)
    E_RWA_minus = np.conj(E_RWA_plus)
    omega_laser = pulse_seq.carrier_freq_fs

    size = 4
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

    L = np.zeros((size * size, size * size), dtype=complex)

    # 1) One-excitation coherences
    term = -1j * (sim_oqs.system.omega_ij(1, 0) - omega_laser) - sim_oqs.sb_coupling.paper_Gamma_ij(
        1, 0
    )
    L[idx_10, idx_10] = term
    L[idx_10, idx_00] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_10, idx_11] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_10, idx_12] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_10, idx_30] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[3, 1]
    L[idx_01, idx_01] = np.conj(term)
    L[idx_01, idx_00] = np.conj(L[idx_10, idx_00])
    L[idx_01, idx_11] = np.conj(L[idx_10, idx_11])
    L[idx_01, idx_21] = np.conj(L[idx_10, idx_12])
    L[idx_01, idx_03] = np.conj(L[idx_10, idx_30])

    term = -1j * (sim_oqs.system.omega_ij(2, 0) - omega_laser) - sim_oqs.sb_coupling.paper_Gamma_ij(
        2, 0
    )
    L[idx_20, idx_20] = term
    L[idx_20, idx_00] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_20, idx_22] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_20, idx_21] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_20, idx_30] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[3, 2]
    L[idx_02, idx_02] = np.conj(term)
    L[idx_02, idx_00] = np.conj(L[idx_20, idx_00])
    L[idx_02, idx_22] = np.conj(L[idx_20, idx_22])
    L[idx_02, idx_12] = np.conj(L[idx_20, idx_21])
    L[idx_02, idx_03] = np.conj(L[idx_20, idx_30])

    # 2) Double-excited coherences
    term = -1j * (
        sim_oqs.system.omega_ij(3, 0) - 2 * omega_laser
    ) - sim_oqs.sb_coupling.paper_Gamma_ij(3, 0)
    L[idx_30, idx_30] = term
    L[idx_30, idx_10] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 1]
    L[idx_30, idx_20] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 2]
    L[idx_30, idx_31] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_30, idx_32] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_03, idx_03] = np.conj(term)
    L[idx_03, idx_01] = np.conj(L[idx_30, idx_10])
    L[idx_03, idx_02] = np.conj(L[idx_30, idx_20])
    L[idx_03, idx_13] = np.conj(L[idx_30, idx_31])
    L[idx_03, idx_23] = np.conj(L[idx_30, idx_32])

    # 3) Cross-coherences
    term = -1j * sim_oqs.system.omega_ij(1, 2) - sim_oqs.sb_coupling.paper_Gamma_ij(1, 2)
    L[idx_12, idx_12] = term
    L[idx_12, idx_02] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_12, idx_13] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 2]
    L[idx_12, idx_32] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[3, 1]
    L[idx_12, idx_10] = -1j * E_RWA_minus * sim_oqs.system.dipole_op[2, 0]
    L[idx_21, idx_21] = np.conj(term)
    L[idx_21, idx_20] = np.conj(L[idx_12, idx_02])
    L[idx_21, idx_31] = np.conj(L[idx_12, idx_13])
    L[idx_21, idx_23] = np.conj(L[idx_12, idx_32])
    L[idx_21, idx_01] = np.conj(L[idx_12, idx_10])

    term = -1j * (sim_oqs.system.omega_ij(3, 1) - omega_laser) - sim_oqs.sb_coupling.paper_Gamma_ij(
        3, 1
    )
    L[idx_31, idx_31] = term
    L[idx_31, idx_11] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 1]
    L[idx_31, idx_21] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 2]
    L[idx_31, idx_30] = -1j * E_RWA_minus * sim_oqs.system.dipole_op[1, 0]
    L[idx_13, idx_13] = np.conj(term)
    L[idx_13, idx_11] = np.conj(L[idx_31, idx_11])
    L[idx_13, idx_12] = np.conj(L[idx_31, idx_21])
    L[idx_13, idx_03] = np.conj(L[idx_31, idx_30])

    term = -1j * (sim_oqs.system.omega_ij(3, 2) - omega_laser) - sim_oqs.sb_coupling.paper_Gamma_ij(
        3, 2
    )
    L[idx_32, idx_32] = term
    L[idx_32, idx_22] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 2]
    L[idx_32, idx_12] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 1]
    L[idx_32, idx_30] = -1j * E_RWA_minus * sim_oqs.system.dipole_op[2, 0]
    L[idx_23, idx_23] = np.conj(term)
    L[idx_23, idx_22] = np.conj(L[idx_32, idx_22])
    L[idx_23, idx_21] = np.conj(L[idx_32, idx_12])
    L[idx_23, idx_03] = np.conj(L[idx_32, idx_30])

    # 4) Populations
    L[idx_00, idx_01] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_00, idx_02] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_00, idx_10] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[1, 0]
    L[idx_00, idx_20] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[2, 0]

    L[idx_11, idx_11] = -sim_oqs.sb_coupling.paper_Gamma_ij(1, 1)
    L[idx_11, idx_22] = sim_oqs.sb_coupling.paper_gamma_ij(1, 2)
    L[idx_11, idx_01] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_11, idx_13] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 1]
    L[idx_11, idx_31] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[3, 1]
    L[idx_11, idx_10] = -1j * E_RWA_minus * sim_oqs.system.dipole_op[1, 0]

    L[idx_22, idx_22] = -sim_oqs.sb_coupling.paper_Gamma_ij(2, 2)
    L[idx_22, idx_11] = sim_oqs.sb_coupling.paper_gamma_ij(2, 1)
    L[idx_22, idx_02] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_22, idx_23] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 2]
    L[idx_22, idx_32] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[3, 2]
    L[idx_22, idx_20] = -1j * E_RWA_minus * sim_oqs.system.dipole_op[2, 0]

    L[idx_33, :] = -L[idx_00, :] - L[idx_11, :] - L[idx_22, :]

    return Qobj(L, dims=[[[size], [size]], [[size], [size]]])

'''


def _matrix_ODE_paper_2atom(t: float, sim_oqs: SimulationModuleOQS) -> Qobj:
    """Column-stacked Liouvillian for a coupled dimer (n_atoms=2).
    NOW WITH TIME-DEPENDENT RATES!!!"""
    pulse_seq = sim_oqs.laser
    E_RWA_plus = e_pulses(t, pulse_seq)
    E_RWA_minus = np.conj(E_RWA_plus)
    omega_laser = pulse_seq.carrier_freq_fs

    size = 4
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

    L = np.zeros((size * size, size * size), dtype=complex)

    # 1) One-excitation coherences
    term = -1j * (
        sim_oqs.system.omega_ij(1, 0) - omega_laser
    ) - sim_oqs.time_dep_paper_Gamma_ij(1, 0, t)
    L[idx_10, idx_10] = term
    L[idx_10, idx_00] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_10, idx_11] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_10, idx_12] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_10, idx_30] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[3, 1]
    L[idx_01, idx_01] = np.conj(term)
    L[idx_01, idx_00] = np.conj(L[idx_10, idx_00])
    L[idx_01, idx_11] = np.conj(L[idx_10, idx_11])
    L[idx_01, idx_21] = np.conj(L[idx_10, idx_12])
    L[idx_01, idx_03] = np.conj(L[idx_10, idx_30])

    term = -1j * (
        sim_oqs.system.omega_ij(2, 0) - omega_laser
    ) - sim_oqs.time_dep_paper_Gamma_ij(2, 0, t)
    L[idx_20, idx_20] = term
    L[idx_20, idx_00] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_20, idx_22] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_20, idx_21] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_20, idx_30] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[3, 2]
    L[idx_02, idx_02] = np.conj(term)
    L[idx_02, idx_00] = np.conj(L[idx_20, idx_00])
    L[idx_02, idx_22] = np.conj(L[idx_20, idx_22])
    L[idx_02, idx_12] = np.conj(L[idx_20, idx_21])
    L[idx_02, idx_03] = np.conj(L[idx_20, idx_30])

    # 2) Double-excited coherences
    term = -1j * (
        sim_oqs.system.omega_ij(3, 0) - 2 * omega_laser
    ) - sim_oqs.time_dep_paper_Gamma_ij(3, 0, t)
    L[idx_30, idx_30] = term
    L[idx_30, idx_10] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 1]
    L[idx_30, idx_20] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 2]
    L[idx_30, idx_31] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_30, idx_32] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_03, idx_03] = np.conj(term)
    L[idx_03, idx_01] = np.conj(L[idx_30, idx_10])
    L[idx_03, idx_02] = np.conj(L[idx_30, idx_20])
    L[idx_03, idx_13] = np.conj(L[idx_30, idx_31])
    L[idx_03, idx_23] = np.conj(L[idx_30, idx_32])

    # 3) Cross-coherences
    term = -1j * sim_oqs.system.omega_ij(1, 2) - sim_oqs.time_dep_paper_Gamma_ij(
        1, 2, t
    )
    L[idx_12, idx_12] = term
    L[idx_12, idx_02] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_12, idx_13] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 2]
    L[idx_12, idx_32] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[3, 1]
    L[idx_12, idx_10] = -1j * E_RWA_minus * sim_oqs.system.dipole_op[2, 0]
    L[idx_21, idx_21] = np.conj(term)
    L[idx_21, idx_20] = np.conj(L[idx_12, idx_02])
    L[idx_21, idx_31] = np.conj(L[idx_12, idx_13])
    L[idx_21, idx_23] = np.conj(L[idx_12, idx_32])
    L[idx_21, idx_01] = np.conj(L[idx_12, idx_10])

    term = -1j * (
        sim_oqs.system.omega_ij(3, 1) - omega_laser
    ) - sim_oqs.time_dep_paper_Gamma_ij(3, 1, t)
    L[idx_31, idx_31] = term
    L[idx_31, idx_11] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 1]
    L[idx_31, idx_21] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 2]
    L[idx_31, idx_30] = -1j * E_RWA_minus * sim_oqs.system.dipole_op[1, 0]
    L[idx_13, idx_13] = np.conj(term)
    L[idx_13, idx_11] = np.conj(L[idx_31, idx_11])
    L[idx_13, idx_12] = np.conj(L[idx_31, idx_21])
    L[idx_13, idx_03] = np.conj(L[idx_31, idx_30])

    term = -1j * (
        sim_oqs.system.omega_ij(3, 2) - omega_laser
    ) - sim_oqs.time_dep_paper_Gamma_ij(3, 2, t)
    L[idx_32, idx_32] = term
    L[idx_32, idx_22] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 2]
    L[idx_32, idx_12] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 1]
    L[idx_32, idx_30] = -1j * E_RWA_minus * sim_oqs.system.dipole_op[2, 0]
    L[idx_23, idx_23] = np.conj(term)
    L[idx_23, idx_22] = np.conj(L[idx_32, idx_22])
    L[idx_23, idx_21] = np.conj(L[idx_32, idx_12])
    L[idx_23, idx_03] = np.conj(L[idx_32, idx_30])

    # 4) Populations
    L[idx_00, idx_01] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_00, idx_02] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_00, idx_10] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[1, 0]
    L[idx_00, idx_20] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[2, 0]

    L[idx_11, idx_11] = -sim_oqs.time_dep_paper_Gamma_ij(1, 1, t)
    L[idx_11, idx_22] = sim_oqs.time_dep_paper_gamma_ij(1, 2, t)
    L[idx_11, idx_01] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[1, 0]
    L[idx_11, idx_13] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 1]
    L[idx_11, idx_31] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[3, 1]
    L[idx_11, idx_10] = -1j * E_RWA_minus * sim_oqs.system.dipole_op[1, 0]

    L[idx_22, idx_22] = -sim_oqs.time_dep_paper_Gamma_ij(2, 2, t)
    L[idx_22, idx_11] = sim_oqs.time_dep_paper_gamma_ij(2, 1, t)
    L[idx_22, idx_02] = 1j * E_RWA_plus * sim_oqs.system.dipole_op[2, 0]
    L[idx_22, idx_23] = -1j * E_RWA_plus * sim_oqs.system.dipole_op[3, 2]
    L[idx_22, idx_32] = 1j * E_RWA_minus * sim_oqs.system.dipole_op[3, 2]
    L[idx_22, idx_20] = -1j * E_RWA_minus * sim_oqs.system.dipole_op[2, 0]

    # Check for NaN/inf values before computing L[idx_33, :]
    L_sum = L[idx_00, :] + L[idx_11, :] + L[idx_22, :]
    # Replace NaN/inf with zeros to avoid propagation
    L_sum = np.nan_to_num(L_sum, nan=0.0, posinf=0.0, neginf=0.0)
    L[idx_33, :] = -L_sum

    return Qobj(L, dims=[[[size], [size]], [[size], [size]]])
