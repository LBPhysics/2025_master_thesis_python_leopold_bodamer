"""Bath-coupling helpers shared by the simulation layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import copy
from qutip import BosonicEnvironment, Qobj

from .atomic_system import AtomicSystem, pair_to_index
from ..config.defaults import DEFAULTS

ATOMIC_DEFAULTS = DEFAULTS["atomic"]
DEPH_RATE_FS = ATOMIC_DEFAULTS["deph_rate_fs"]
DOWN_RATE_FS = ATOMIC_DEFAULTS["down_rate_fs"]
UP_RATE_FS = ATOMIC_DEFAULTS["up_rate_fs"]


def _dephasing_projector(system: AtomicSystem, index: int) -> Qobj:
    return system.deph_op_i(index)


def redfield_decay_channels(
    system: AtomicSystem, bath: BosonicEnvironment
) -> list[tuple[Qobj, BosonicEnvironment]]:
    """Return paper-aligned Bloch-Redfield coupling operators.

    For the current max_excitation<=2 truncation:
        A_i = |i><i| + sum_{j != i} |ij><ij|
    and
        H_SB = sum_i F_i A_i

    Each site channel gets its own bath instance to reflect the paper's
    assumption of uncorrelated local baths.
    """
    channels: list[tuple[Qobj, BosonicEnvironment]] = []

    BR_NORM = 2.0
    pref = np.sqrt(BR_NORM)

    def add_channel(operator: Qobj) -> None:
        bath_i = copy.deepcopy(bath)
        channels.append((pref * system.to_eigenbasis(operator), bath_i))

    site_ops = {
        i_atom: _dephasing_projector(system, i_atom) for i_atom in range(1, system.n_atoms + 1)
    }

    if system.max_excitation >= 2:
        for i_atom in range(1, system.n_atoms):
            for j_atom in range(i_atom + 1, system.n_atoms + 1):
                idx = pair_to_index(i_atom, j_atom, system.n_atoms)
                pair_proj = _dephasing_projector(system, idx)
                site_ops[i_atom] += pair_proj
                site_ops[j_atom] += pair_proj

    for i_atom in range(1, system.n_atoms + 1):
        add_channel(site_ops[i_atom])

    # Keep monomer thermalization
    if system.n_atoms == 1:
        lowering = system.basis[0] * system.basis[1].dag()
        add_channel(lowering + lowering.dag())

    return channels


def lindblad_decay_channels(system: AtomicSystem) -> list[Qobj]:
    """Return Lindblad collapse operators aligned with the paper/BR structure.

    For the current max_excitation<=2 truncation, use one combined operator
    per site
        A_i = |i><i| + sum_{j != i} |ij><ij|
    and transform it to the eigenbasis.

    For dimers, we do not add explicit ground-state up/down channels so the
    structure stays parallel to the paper/BR dimer model.
    """
    channels: list[Qobj] = []

    def add_channel(operator: Qobj, rate: float) -> None:
        if rate <= 0:
            return
        channels.append(system.to_eigenbasis(operator) * np.sqrt(rate))

    # Build one combined bath operator per site
    site_ops = {i_atom: system.deph_op_i(i_atom) for i_atom in range(1, system.n_atoms + 1)}

    if system.max_excitation >= 2:
        for i_atom in range(1, system.n_atoms):
            for j_atom in range(i_atom + 1, system.n_atoms + 1):
                idx = pair_to_index(i_atom, j_atom, system.n_atoms)
                pair_proj = system.deph_op_i(idx)
                site_ops[i_atom] += pair_proj
                site_ops[j_atom] += pair_proj

    for i_atom in range(1, system.n_atoms + 1):
        add_channel(site_ops[i_atom], DEPH_RATE_FS)

    # Keep monomer thermalization
    if system.n_atoms == 1:
        lower = system.basis[0] * system.basis[1].dag()
        add_channel(lower, DOWN_RATE_FS)
        if UP_RATE_FS > 0:
            add_channel(system.basis[1] * system.basis[0].dag(), UP_RATE_FS)

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

    def paper_gamma_ab_fs(self, a: int, b: int) -> float:
        """Paper uses Ω_ab for transfer b -> a with ω_ab = E_a - E_b.
        QuTiP power_spectrum uses the opposite sign convention for transition rates,
        so we must evaluate at ω_ba = -ω_ab."""
        return np.sin(2 * self.theta) ** 2 * self.bath.power_spectrum(self.system.omega_ij(b, a))

    def paper_Gamma_ab_fs(self, a: int, b: int) -> float:
        S_0 = self.bath.power_spectrum(0)
        sin2 = np.sin(2 * self.theta) ** 2
        cos2 = np.cos(2 * self.theta) ** 2
        gamma_21 = self.paper_gamma_ab_fs(2, 1)
        gamma_12 = self.paper_gamma_ab_fs(1, 2)
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
        raise ValueError(f"paper_Gamma_ab_fs: invalid dimer index pair (a={a}, b={b})")

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
