"""Bath-coupling helpers shared by the simulation layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qutip import BosonicEnvironment, Qobj

from .atomic_system import AtomicSystem, pair_to_index
from ..config.defaults import DEFAULTS

ATOMIC_DEFAULTS = DEFAULTS["atomic"]
DEPH_RATE_FS = ATOMIC_DEFAULTS["deph_rate_fs"]
DOWN_RATE_FS = ATOMIC_DEFAULTS["down_rate_fs"]
UP_RATE_FS = ATOMIC_DEFAULTS["up_rate_fs"]


def _dephasing_projector(system: AtomicSystem, index: int) -> Qobj:
    return system.deph_op_i(index)


def redfield_decay_channels(system: AtomicSystem, bath: BosonicEnvironment) -> list[tuple[Qobj, BosonicEnvironment]]:
    """Return Bloch-Redfield coupling operators for the current system."""
    channels: list[tuple[Qobj, BosonicEnvironment]] = []

    def add_channel(operator: Qobj) -> None:
        channels.append([system.to_eigenbasis(operator), bath])

    for i_atom in range(1, system.n_atoms + 1):
        add_channel(_dephasing_projector(system, i_atom))
        if system.n_atoms != 2:
            lowering = system.basis[0] * system.basis[i_atom].dag()
            add_channel(lowering + lowering.dag())

    if system.max_excitation != 2:
        return channels

    for i_atom in range(1, system.n_atoms):
        for j_atom in range(i_atom + 1, system.n_atoms + 1):
            idx = pair_to_index(i_atom, j_atom, system.n_atoms)
            add_channel(_dephasing_projector(system, idx))
            add_channel(_dephasing_projector(system, idx))

    if system.n_atoms == 2:
        return channels

    for i_atom in range(1, system.n_atoms):
        for j_atom in range(i_atom + 1, system.n_atoms + 1):
            idx = pair_to_index(i_atom, j_atom, system.n_atoms)
            op_ij_i = system.basis[i_atom] * system.basis[idx].dag()
            op_ij_j = system.basis[j_atom] * system.basis[idx].dag()
            add_channel(op_ij_i + op_ij_i.dag())
            add_channel(op_ij_j + op_ij_j.dag())
    return channels


def lindblad_decay_channels(system: AtomicSystem) -> list[Qobj]:
    """Return Lindblad collapse operators for the current system."""
    channels: list[Qobj] = []

    def add_channel(operator: Qobj, rate: float) -> None:
        channels.append(system.to_eigenbasis(operator) * np.sqrt(rate))

    for i_atom in range(1, system.n_atoms + 1):
        add_channel(system.deph_op_i(i_atom), DEPH_RATE_FS)
        if system.n_atoms != 2:
            lower = system.basis[0] * system.basis[i_atom].dag()
            add_channel(lower, DOWN_RATE_FS)
            if UP_RATE_FS > 0:
                add_channel(system.basis[i_atom] * system.basis[0].dag(), UP_RATE_FS)

    if system.max_excitation != 2:
        return channels

    for i_atom in range(1, system.n_atoms):
        for j_atom in range(i_atom + 1, system.n_atoms + 1):
            idx = pair_to_index(i_atom, j_atom, system.n_atoms)
            deph_op_ij = system.deph_op_i(idx)
            add_channel(deph_op_ij, DEPH_RATE_FS)
            add_channel(deph_op_ij, DEPH_RATE_FS)

    if system.n_atoms == 2:
        return channels

    for i_atom in range(1, system.n_atoms):
        for j_atom in range(i_atom + 1, system.n_atoms + 1):
            idx = pair_to_index(i_atom, j_atom, system.n_atoms)
            add_channel(system.basis[i_atom] * system.basis[idx].dag(), DOWN_RATE_FS)
            add_channel(system.basis[j_atom] * system.basis[idx].dag(), DOWN_RATE_FS)
    return channels


@dataclass
class BathCoupling:
    system: AtomicSystem
    bath: BosonicEnvironment

    @property
    def theta(self) -> float:
        if self.system.n_atoms != 2:
            raise ValueError("theta is only defined for n_atoms == 2")
        detuning = self.system.frequencies_fs[0] - self.system.frequencies_fs[1]
        return 0.5 * np.arctan2(2 * self.system.coupling_fs, detuning)

    def paper_gamma_ab(self, a: int, b: int) -> float:
        return np.sin(2 * self.theta) ** 2 * self.bath.power_spectrum(self.system.omega_ij(a, b))

    def paper_Gamma_ab(self, a: int, b: int) -> float:
        S_0 = self.bath.power_spectrum(0)
        sin2 = np.sin(2 * self.theta) ** 2
        cos2 = np.cos(2 * self.theta) ** 2
        gamma_21 = self.paper_gamma_ab(2, 1)
        gamma_12 = self.paper_gamma_ab(1, 2)
        gamma_sum = gamma_21 + gamma_12
        Gamma_ab = 2 * cos2 * S_0
        Gamma_a0 = (1 - 0.5 * sin2) * S_0
        Gamma_abar_0 = 2 * S_0
        Gamma_abar_a = Gamma_a0

        if a == 1:
            if b == 0:
                return Gamma_a0 + 0.5 * gamma_21
            if b == 1:
                return gamma_21
            if b == 2:
                return Gamma_ab + 0.5 * gamma_sum
        elif a == 2:
            if b == 0:
                return Gamma_a0 + 0.5 * gamma_12
            if b == 1:
                return Gamma_ab + 0.5 * gamma_sum
            if b == 2:
                return gamma_12
        elif a == 3:
            if b == 0:
                return Gamma_abar_0
            if b == 1:
                return Gamma_abar_a + 0.5 * gamma_21
            if b == 2:
                return Gamma_abar_a + 0.5 * gamma_12
        raise ValueError(f"paper_Gamma_ab: invalid dimer index pair (a={a}, b={b})")

    def summary(self) -> str:
        return "\n".join(
            [
                "=== BathCoupling Summary ===",
                "System Parameters:",
                str(self.system),
                "Bath Parameters:",
                str(self.bath),
            ]
        )

    def __str__(self) -> str:
        return self.summary()


__all__ = ["BathCoupling", "lindblad_decay_channels", "redfield_decay_channels"]