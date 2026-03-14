"""Atomic system model and operator construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from qutip import Qobj, basis as qt_basis, ket2dm

from ...utils.constants import HBAR, convert_cm_to_fs, convert_fs_to_cm
from .basis import excitation_number_from_index, pair_to_index
from .geometry import cylindrical_positions, isotropic_coupling_matrix


@dataclass
class AtomicSystem:
    n_atoms: int = 1
    n_chains: int = 1
    frequencies_cm: list[float] = field(default_factory=lambda: [16000.0])
    dip_moments: list[float] = field(default_factory=lambda: [1.0])
    coupling_cm: float = 0.0
    delta_inhomogen_cm: float = 0.0
    max_excitation: int = 1

    def __post_init__(self) -> None:
        self.frequencies_cm_history = [self.frequencies_cm.copy()]
        self._sync_units()
        self._build_basis()
        self._setup_geometry()

    def _sync_units(self) -> None:
        self.frequencies_fs = np.asarray(convert_cm_to_fs(self.frequencies_cm), dtype=float)
        self.coupling_fs = float(convert_cm_to_fs(self.coupling_cm))
        self.delta_inhomogen_fs = float(convert_cm_to_fs(self.delta_inhomogen_cm))

    def _build_basis(self) -> None:
        self.basis = [qt_basis(self.dimension, index) for index in range(self.dimension)]

    def _setup_geometry(self) -> None:
        self.n_rings = self.n_atoms // self.n_chains
        self._positions = cylindrical_positions(self.n_atoms, self.n_chains)
        self._coupling_matrix_fs = isotropic_coupling_matrix(
            self._positions,
            self.dip_moments,
            self.coupling_fs,
        )

    def reset_cache(self) -> None:
        for key in ("eigenstates", "eigenbasis_transform"):
            self.__dict__.pop(key, None)

    def update_frequencies_cm(self, new_freqs: list[float]) -> None:
        if len(new_freqs) != self.n_atoms:
            raise ValueError(f"Expected {self.n_atoms} frequencies, got {len(new_freqs)}")
        self.frequencies_cm_history.append(new_freqs.copy())
        self.frequencies_cm = new_freqs.copy()
        self._sync_units()
        self.reset_cache()

    def update_delta_inhomogen_cm(self, new_delta_inhomogen_cm: float) -> None:
        self.delta_inhomogen_cm = float(new_delta_inhomogen_cm)
        self.delta_inhomogen_fs = float(convert_cm_to_fs(self.delta_inhomogen_cm))
        self.reset_cache()

    @property
    def dimension(self) -> int:
        if self.max_excitation == 1:
            return 1 + self.n_atoms
        if self.max_excitation == 2:
            return 1 + self.n_atoms + math.comb(self.n_atoms, 2)
        raise ValueError(f"Unsupported max_excitation: {self.max_excitation}")

    @property
    def hamiltonian(self) -> Qobj:
        hamiltonian = 0.0
        for i_site in range(1, self.n_atoms + 1):
            omega_i = float(self.frequencies_fs[i_site - 1])
            hamiltonian += HBAR * omega_i * ket2dm(self.basis[i_site])

        if self.max_excitation == 2 and self.n_atoms >= 2:
            for i_site in range(1, self.n_atoms):
                for j_site in range(i_site + 1, self.n_atoms + 1):
                    idx_d = pair_to_index(i_site, j_site, self.n_atoms)
                    omega_sum = float(self.frequencies_fs[i_site - 1] + self.frequencies_fs[j_site - 1])
                    hamiltonian += HBAR * omega_sum * ket2dm(self.basis[idx_d])

        return hamiltonian + self.coupling_op

    def idx_ground(self) -> int:
        return 0

    def idx_single(self, i: int) -> int:
        if not 1 <= i <= self.n_atoms:
            raise ValueError(f"single index i must be in [1,{self.n_atoms}], got {i}")
        return i

    def idx_double(self, i: int, j: int) -> int:
        if not (1 <= i <= self.n_atoms and 1 <= j <= self.n_atoms and i != j):
            raise ValueError(f"double indices must be distinct in [1,{self.n_atoms}], got ({i},{j})")
        a, b = (i, j) if i < j else (j, i)
        return pair_to_index(a, b, self.n_atoms)

    @cached_property
    def eigenstates(self) -> tuple[np.ndarray, np.ndarray]:
        return self.hamiltonian.eigenstates()

    @cached_property
    def eigenbasis_transform(self) -> Qobj:
        _, basis_states = self.eigenstates
        return Qobj(
            np.column_stack([state.full() for state in basis_states]),
            dims=self.hamiltonian.dims,
        )

    def to_eigenbasis(self, operator: Qobj) -> Qobj:
        return self.eigenbasis_transform.dag() * operator * self.eigenbasis_transform

    def to_site_basis(self, operator: Qobj) -> Qobj:
        return self.eigenbasis_transform * operator * self.eigenbasis_transform.dag()

    def ground_state_dm(self) -> Qobj:
        return ket2dm(self.basis[self.idx_ground()])

    @property
    def lowering_op(self) -> Qobj:
        lowering_op = 0.0
        for i_site in range(1, self.n_atoms + 1):
            lowering_op += self.dip_moments[i_site - 1] * (self.basis[0] * self.basis[i_site].dag())
        if self.max_excitation == 2:
            for i_site in range(1, self.n_atoms):
                for j_site in range(i_site + 1, self.n_atoms + 1):
                    idx = pair_to_index(i_site, j_site, self.n_atoms)
                    lowering_op += self.dip_moments[i_site - 1] * (self.basis[j_site] * self.basis[idx].dag())
                    lowering_op += self.dip_moments[j_site - 1] * (self.basis[i_site] * self.basis[idx].dag())
        return lowering_op

    @property
    def dipole_op(self) -> Qobj:
        return self.lowering_op + self.lowering_op.dag()

    @property
    def number_op(self) -> Qobj:
        number_op = 0.0
        for index in range(self.dimension):
            number_op += excitation_number_from_index(index, self.n_atoms) * ket2dm(self.basis[index])
        return number_op

    @property
    def coupling_op(self) -> Qobj:
        if self.n_atoms <= 1:
            return 0
        coupling_op = ket2dm(self.basis[0]) * 0.0
        for i_site in range(1, self.n_atoms + 1):
            for j_site in range(i_site + 1, self.n_atoms + 1):
                coupling_ij = self._coupling_matrix_fs[i_site - 1, j_site - 1]
                if np.isclose(coupling_ij, 0.0):
                    continue
                ket_i = self.basis[i_site]
                ket_j = self.basis[j_site]
                coupling_op += HBAR * coupling_ij * (ket_i * ket_j.dag() + ket_j * ket_i.dag())
        if self.max_excitation == 2 and self.n_atoms >= 3:
            for i_site in range(1, self.n_atoms):
                for j_site in range(i_site + 1, self.n_atoms + 1):
                    idx_ij = pair_to_index(i_site, j_site, self.n_atoms)
                    ket_ij = self.basis[idx_ij]
                    for k_site in range(1, self.n_atoms + 1):
                        if k_site in {i_site, j_site}:
                            continue
                        coupling_jk = self._coupling_matrix_fs[j_site - 1, k_site - 1]
                        if not np.isclose(coupling_jk, 0.0):
                            idx_ik = pair_to_index(min(i_site, k_site), max(i_site, k_site), self.n_atoms)
                            if idx_ij < idx_ik:
                                ket_ik = self.basis[idx_ik]
                                coupling_op += HBAR * coupling_jk * (ket_ij * ket_ik.dag() + ket_ik * ket_ij.dag())
                        coupling_ik = self._coupling_matrix_fs[i_site - 1, k_site - 1]
                        if not np.isclose(coupling_ik, 0.0):
                            idx_kj = pair_to_index(min(k_site, j_site), max(k_site, j_site), self.n_atoms)
                            if idx_ij < idx_kj:
                                ket_kj = self.basis[idx_kj]
                                coupling_op += HBAR * coupling_ik * (ket_ij * ket_kj.dag() + ket_kj * ket_ij.dag())
        return coupling_op

    @property
    def coupling_matrix_cm(self) -> np.ndarray:
        return convert_fs_to_cm(self._coupling_matrix_fs)

    def deph_op_i(self, i: int) -> Qobj:
        if i == 0:
            raise ValueError("indexing ground state -> use i elem 1,...,N+1")
        return ket2dm(self.basis[i])

    def omega_ij(self, i: int, j: int) -> float:
        return self.eigenstates[0][i] - self.eigenstates[0][j]

    def summary(self) -> str:
        lines = ["=== AtomicSystem Summary ===", "", "# The system with:", f"    {'n_atoms':<20}: {self.n_atoms}"]
        if len(self.frequencies_cm_history) > 1:
            all_freqs = np.array(self.frequencies_cm_history)
            lines.extend(
                [
                    "\n# Frequencies (cm^-1):",
                    f"    Current: {self.frequencies_cm}",
                    f"    Min: {float(all_freqs.min()):.2f}",
                    f"    Max: {float(all_freqs.max()):.2f}",
                ]
            )
        else:
            lines.extend(["\n# Frequencies (cm^-1):", f"    {self.frequencies_cm}"])
        lines.append("\n# Dipole Moments:")
        for i_site in range(self.n_atoms):
            lines.append(f"    Atom {i_site}: mu = {self.dip_moments[i_site]}")
        lines.append("\n# Coupling / Inhomogeneity:")
        if self.n_atoms == 2:
            lines.append(f"    {'coupling':<20}: {self.coupling_cm} cm^-1")
            lines.append(f"    {'delta':<20}: {self.delta_inhomogen_cm} cm^-1")
        elif self.n_atoms > 2:
            lines.append(f"    {'n_rings':<20}: {self.n_rings} (n_chains = {self.n_chains})")
            lines.append(f"    {'positions shape':<20}: {self._positions.shape}")
            lines.append(f"    {'coupling matrix (cm^-1)':<20}:")
            lines.append(str(self.coupling_matrix_cm))
        lines.append(f"\n    {'System Hamiltonian (undiagonalized)':<20}:")
        lines.append(str(self.hamiltonian))
        lines.append("\n# Dipole operator (dipole_op):")
        lines.append(str(self.dipole_op))
        lines.append("\n=== End of Summary ===")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        if len(self.frequencies_cm_history) > 1:
            all_freqs = np.array(self.frequencies_cm_history)
            freq_info = {
                "current": self.frequencies_cm,
                "min": float(all_freqs.min()),
                "max": float(all_freqs.max()),
            }
        else:
            freq_info = self.frequencies_cm
        payload = {
            "n_atoms": self.n_atoms,
            "frequencies_cm": freq_info,
            "dip_moments": self.dip_moments,
        }
        if self.n_chains != 1:
            payload["n_chains"] = self.n_chains
        if self.n_rings != 1:
            payload["n_rings"] = self.n_rings
        if self.delta_inhomogen_cm != 0.0:
            payload["delta_inhomogen_cm"] = self.delta_inhomogen_cm
        if self.coupling_cm != 0.0:
            payload["coupling_cm"] = self.coupling_cm
        return payload


__all__ = ["AtomicSystem"]
