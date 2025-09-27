# this file defines the AtomicSystem class

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from functools import cached_property
from qutip import basis as qt_basis, ket2dm, Qobj

from ...utils.constants import HBAR, convert_cm_to_fs, convert_fs_to_cm


@dataclass
class AtomicSystem:
    # VITAL ATTRIBUTES
    n_atoms: int = 1
    n_chains: int = 1
    frequencies_cm: List[float] = field(default_factory=lambda: [16000.0])  # in cm^-1
    dip_moments: List[float] = field(default_factory=lambda: [1.0])
    coupling_cm: float = 0.0
    delta_inhomogen_cm: float = 0.0

    # 1 = existing behaviour (ground + single exc manifold); 2 adds double manifold
    max_excitation: int = 1

    psi_ini: Optional[Qobj] = None  # initial state, default is ground state

    def __post_init__(self):
        # build basis
        self._build_basis()

        # spectroscopy always starts in the ground state
        self.psi_ini = ket2dm(self.basis[0])

        # store the initial frequencies in history
        self.frequencies_cm_history = [self.frequencies_cm.copy()]

        # internal fs^-1 storage (single source of truth for dynamics)
        self.frequencies_fs = np.asarray(
            convert_cm_to_fs(self.frequencies_cm), dtype=float
        )
        self.coupling_fs = convert_cm_to_fs(self.coupling_cm)
        self.delta_inhomogen_fs = convert_cm_to_fs(self.delta_inhomogen_cm)

        # always set cylindrical positions and compute isotropic couplings
        self.n_rings = self.n_atoms // self.n_chains
        self._setup_geometry_and_couplings()

    # CHECKED
    def update_frequencies_cm(self, new_freqs: List[float]):
        if len(new_freqs) != self.n_atoms:
            raise ValueError(
                f"Expected {self.n_atoms} frequencies, got {len(new_freqs)}"
            )

        # Save current freqs before updating
        self.frequencies_cm_history.append(new_freqs.copy())
        self.frequencies_cm = new_freqs.copy()
        # keep fs cache in sync
        self.frequencies_fs = np.asarray(
            convert_cm_to_fs(self.frequencies_cm), dtype=float
        )
        # cached spectrum/operators depend on frequencies
        self.reset_cache()

    # CHECKED
    def update_delta_inhomogen_cm(self, new_delta_inhomogen_cm: float) -> None:
        """Update inhomogeneous broadening (cm^-1)."""
        self.delta_inhomogen_cm = new_delta_inhomogen_cm
        self.delta_inhomogen_fs = float(convert_cm_to_fs(self.delta_inhomogen_cm))
        self.reset_cache()

    def reset_cache(self) -> None:
        """Invalidate cached spectral quantities affected by parameter changes."""
        # delete cached_properties to force recompute on parameter changes
        for key in (
            "eigenstates",
            "eigenbasis_transform",
            "hamiltonian",
            "coupling_op",
        ):
            if key in self.__dict__:
                del self.__dict__[key]

    # === CORE PARAMETERS ===
    # CHECKED
    @cached_property
    def dimension(self):
        """Dimension of the Hilbert space (ground + single + double excitations)."""
        N = self.n_atoms
        exc = self.max_excitation
        if exc == 1:
            dim = 1 + N
        elif exc == 2:
            n_pairs = math.comb(N, 2)
            dim = 1 + N + n_pairs
        return dim

    # === QUANTUM OPERATORS ===
    def _build_basis(self):
        """Construct basis depending on n_atoms and max_excitation.
        - Ground state index: 0
        - Single excitations: |i> mapped to index i (1..N) when max_excitation>=1
        - Double excitations: |i,j> (i<j) mapped to indices N + p where
              p runs from 1..N_pairs in lexicographic order over (i,j).
        """
        dim = self.dimension
        self.basis = [qt_basis(dim, i) for i in range(dim)]

    @cached_property
    def hamiltonian(self) -> Qobj:
        """System Hamiltonian in the truncated exciton basis (ground + singles [+ doubles]).

        H = H_onsite + H_coupling
        - On-site part adds ħ ω_i on |i><i| (singles) and ħ(ω_i+ω_j) on |i,j><i,j| (doubles).
        - Coupling part is the number-conserving dipole-dipole hopping (see coupling_op).

        Frequencies are taken from the internal fs^-1 cache (self.frequencies_fs).
        """
        N = self.n_atoms
        if N == 0:
            return 0

        # Start with a zero operator of correct dimension
        H = ket2dm(self.basis[0]) * 0.0

        # On-site energies for single-excitation manifold
        for i_site in range(1, N + 1):
            omega_i = float(self.frequencies_fs[i_site - 1])
            H += HBAR * omega_i * ket2dm(self.basis[i_site])

        # On-site energies for double-excitation manifold (if enabled)
        if self.max_excitation == 2 and N >= 2:
            for i_site in range(1, N):
                for j_site in range(i_site + 1, N + 1):
                    # basis index (0-based) of |i,j>
                    idx_d = pair_to_index(i_site, j_site, N) - 1
                    omega_sum = float(
                        self.frequencies_fs[i_site - 1]
                        + self.frequencies_fs[j_site - 1]
                    )
                    H += HBAR * omega_sum * ket2dm(self.basis[idx_d])

        # Add inter-site coupling (Hermitian by construction)
        H += self.coupling_op

        return H

    # === INDEX HELPERS (1-based site indices) ===
    def idx_ground(self) -> int:
        """Index of ground state |g>."""
        return 0

    def idx_single(self, i: int) -> int:
        """Index of single-excited state |i>, with 1 <= i <= N (1-based)."""
        N = self.n_atoms
        if not (1 <= i <= N):
            raise ValueError(f"single index i must be in [1,{N}], got {i}")
        return i

    def idx_double(self, i: int, j: int) -> int:
        """Index of double-excited state |i,j> (unordered pair), 1 <= i != j <= N.
        Returns 0-based basis index suitable for self.basis[idx]."""
        N = self.n_atoms
        if not (1 <= i <= N and 1 <= j <= N and i != j):
            raise ValueError(
                f"double indices must be distinct in [1,{N}], got (i,j)=({i},{j})"
            )
        a, b = (i, j) if i < j else (j, i)
        return pair_to_index(a, b, N) - 1

    @cached_property
    def eigenstates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues & eigenstates (cached)."""
        return self.hamiltonian.eigenstates()

    @cached_property
    def eigenbasis_transform(self) -> Qobj:
        """Unitary matrix U whose columns are eigenstates (site → eigen basis)."""
        _, basis_states = self.eigenstates
        u = Qobj(
            np.column_stack([e.full() for e in basis_states]),
            dims=self.hamiltonian.dims,
        )
        return u

    def to_eigenbasis(self, operator: Qobj) -> Qobj:
        """Transform an operator into the eigenbasis of the Hamiltonian using cached U."""
        return self.eigenbasis_transform.dag() * operator * self.eigenbasis_transform

    @property
    def lowering_op(self) -> Qobj:
        """return the lowering operator in the canonical basis
        ~ positive frequency part of the dipole operator mu^(+)"""
        lowering_op = 0
        # Single-excitation lowering operator: sum_i μ_i |0><i|
        for i in range(1, self.n_atoms + 1):
            mu_i = self.dip_moments[i - 1]
            lowering_op += mu_i * (self.basis[0] * self.basis[i].dag())

        # max_excitation == 2: add terms connecting double -> single manifolds
        # |j><i,j| and |j><j,i| but only one ordering stored (i<j)
        if self.max_excitation == 2:
            N = self.n_atoms
            for i in range(1, N):
                for j in range(i + 1, N + 1):
                    idx = pair_to_index(i, j, N)
                    mu_i = self.dip_moments[i - 1]
                    mu_j = self.dip_moments[j - 1]
                    # Annihilating excitation on site i from |i,j> leaves |j>, and vice versa
                    lowering_op += mu_i * (self.basis[j] * self.basis[idx - 1].dag())
                    lowering_op += mu_j * (self.basis[i] * self.basis[idx - 1].dag())

        return lowering_op

    @property  # CHECKED
    def dipole_op(self) -> Qobj:
        """return the dipole operator in the canonical basis"""
        dip_op = self.lowering_op + self.lowering_op.dag()
        return dip_op

    @property
    def number_op(self) -> Qobj:
        """
        Total excitation number operator (in the canonical basis).

        Definition in the canonical/site basis:
            N = sum_i |i><i| + 2 * sum_{i<j} |i,j><i,j|

        - For max_excitation == 1, only the single-excitation projectors are included.
        """
        N_op = 0
        dim = self.dimension

        for i in range(dim):
            N_op += self.excitation_number_from_index(i) * ket2dm(self.basis[i])

        return N_op

    @cached_property
    def coupling_op(self) -> Qobj:
        """Inter-site coupling operator (cached) using isotropic dipole couplings ~1/r^3.

        Uses the coupling matrix computed from cylindrical geometry.
        Single-manifold couplings: |i><j| + |j><i| with strength ħ J_ij.
        NOTE: If max_excitation == 2, also couples double states that share one site.
        """
        N = self.n_atoms
        if N <= 1:
            return 0

        J = self._coupling_matrix_fs  # fs^-1, symmetric with zero diagonal

        # Start with a zero operator of correct dimensions
        HJ = ket2dm(self.basis[0]) * 0.0

        # Single-excitation manifold: sum_{i<j} ħ J_ij (|i><j| + h.c.)
        for i in range(1, N + 1):
            ki = self.basis[i]
            for j in range(i + 1, N + 1):
                Jij = J[i - 1, j - 1]
                if np.isclose(Jij, 0.0):
                    continue
                kj = self.basis[j]
                HJ += (
                    HBAR * Jij * (ki * kj.dag() + kj * ki.dag())
                )  # NOTE coupling has to be real

        # Double-excitation manifold: connect |i,j> <-> |i,k| with J_{j,k} and |i,j> <-> |k,j| with J_{i,k}
        if self.max_excitation == 2 and N >= 3:
            for i in range(1, N):
                for j in range(i + 1, N + 1):
                    idx_ij = pair_to_index(i, j, N) - 1
                    ket_ij = self.basis[idx_ij]

                    for k in range(1, N + 1):
                        if k == i or k == j:
                            continue

                        # Hop j -> k while keeping i (amplitude J_{j,k})
                        Jij_val = J[j - 1, k - 1]
                        if not np.isclose(Jij_val, 0.0):
                            a, b = (min(i, k), max(i, k))
                            idx_ik = pair_to_index(a, b, N) - 1
                            if idx_ij < idx_ik:  # add each undirected connection once
                                ket_ik = self.basis[idx_ik]
                                HJ += (
                                    HBAR
                                    * Jij_val
                                    * (ket_ij * ket_ik.dag() + ket_ik * ket_ij.dag())
                                )

                        # Hop i -> k while keeping j (amplitude J_{i,k})
                        Iik_val = J[i - 1, k - 1]
                        if not np.isclose(Iik_val, 0.0):
                            a2, b2 = (min(k, j), max(k, j))
                            idx_kj = pair_to_index(a2, b2, N) - 1
                            if idx_ij < idx_kj:
                                ket_kj = self.basis[idx_kj]
                                HJ += (
                                    HBAR
                                    * Iik_val
                                    * (ket_ij * ket_kj.dag() + ket_kj * ket_ij.dag())
                                )

        return HJ

    # === GEOMETRY AND POSITIONS ===
    @property
    def coupling_matrix_cm(self) -> np.ndarray:
        """Return coupling matrix in cm^-1 for display."""
        return convert_fs_to_cm(self._coupling_matrix_fs)

    def _setup_geometry_and_couplings(self):
        """Set up cylindrical geometry and compute isotropic couplings."""
        self._set_cylindrical_positions()
        self._compute_isotropic_couplings()

    def _set_cylindrical_positions(
        self, distance: float = 1.0
    ):  # TODO this could be parameterized
        """Set cylindrical atom positions."""
        n_rings = self.n_rings
        n_chains = self.n_chains

        # Ring centers in xy-plane
        if n_chains == 1:
            # Linear chain
            positions = np.array([[0.0, 0.0, z * distance] for z in range(n_rings)])
        else:
            # Multiple chains in cylinder
            dphi = 2.0 * np.pi / n_chains
            radius = distance / (2.0 * np.sin(np.pi / n_chains))
            ring_centers = np.array(
                [
                    [radius * np.cos(k * dphi), radius * np.sin(k * dphi), 0.0]
                    for k in range(n_chains)
                ]
            )

            positions = np.array(
                [
                    ring_centers[c] + np.array([0.0, 0.0, z * distance])
                    for c in range(n_chains)
                    for z in range(n_rings)
                ]
            )

        self._positions = positions

    def _compute_isotropic_couplings(
        self, power: float = 3.0
    ) -> np.ndarray:  # TODO extend to vectorized dipoles
        """Compute isotropic J_ij = coupling * μ_i * μ_j / r^power."""
        N = self.n_atoms
        J = np.zeros((N, N), dtype=float)
        base_coupling_fs = self.coupling_fs
        if N == 2:
            J[0, 1] = base_coupling_fs
            J[1, 0] = base_coupling_fs
            self._coupling_matrix_fs = J
            # Invalidate operators that depend on couplings
            if "coupling_op" in self.__dict__:
                del self.__dict__["coupling_op"]
            if "hamiltonian" in self.__dict__:
                del self.__dict__["hamiltonian"]
            return
        pos = self._positions

        for i in range(N):
            for j in range(i + 1, N):
                r_vec = pos[j] - pos[i]
                r = float(np.linalg.norm(r_vec))
                if r == 0:
                    raise ValueError("Duplicate positions encountered (zero distance).")

                # Isotropic coupling with dipole product
                coupling_ij = (
                    base_coupling_fs
                    * self.dip_moments[i]
                    * self.dip_moments[j]
                    / (r**power)
                )
                J[i, j] = coupling_ij
                J[j, i] = coupling_ij

        self._coupling_matrix_fs = J
        # Invalidate operators that depend on couplings
        if "coupling_op" in self.__dict__:
            del self.__dict__["coupling_op"]
        if "hamiltonian" in self.__dict__:
            del self.__dict__["hamiltonian"]

    def excitation_number_from_index(self, idx: int) -> int:
        if idx == 0:
            return 0
        elif 1 <= idx <= self.n_atoms:
            return 1
        else:
            return 2

    def deph_op_i(self, i: int) -> Qobj:
        """Return site i population operator in the eigenbasis (|i><i|).
        Also works for double states |i,j> -> returns |i,j><i,j|."""
        if i == 0:
            raise ValueError("indexing ground state -> use i elem 1,...,N+1")
        op = ket2dm(self.basis[i])
        return op

    def omega_ij(self, i: int, j: int) -> float:
        """Return energy difference (frequency) between eigenstates i and j in fs^-1."""
        return self.eigenstates[0][i] - self.eigenstates[0][j]

    def summary(self):
        lines = [
            "=== AtomicSystem Summary ===",
            "",
            "# The system with:",
            f"    {'n_atoms':<20}: {self.n_atoms}",
        ]
        lines.append("\n# Frequencies and Dipole Moments:")
        for i in range(self.n_atoms):
            lines.append(
                f"    Atom {i}: ω = {self.frequencies_cm[i]} cm^-1, μ = {self.dip_moments[i]}"
            )
        lines.append("\n# Coupling / Inhomogeneity:")
        if self.n_atoms == 2:
            lines.append(f"    {'coupling':<20}: {self.coupling_cm} cm^-1")
            lines.append(f"    {'delta':<20}: {self.delta_inhomogen_cm} cm^-1")
        elif self.n_atoms > 2 and self.n_rings is not None:
            lines.append(
                f"    {'n_rings':<20}: {self.n_rings} (n_chains = {self.n_chains})"
            )
            lines.append(f"    {'positions shape':<20}: {self._positions.shape}")
            lines.append(f"    {'coupling matrix (cm^-1)':<20}:")
            lines.append(str(self.coupling_matrix_cm))
        lines.append(f"\n    {'psi_ini':<20}:<")
        lines.append(str(self.psi_ini))
        lines.append(f"\n    {'System Hamiltonian (undiagonalized)':<20}:")
        lines.append(str(self.hamiltonian))
        lines.append("\n# Dipole operator (dipole_op):")
        lines.append(str(self.dipole_op))
        lines.append("\n=== End of Summary ===")
        return "\n".join(lines)

    def __str__(self) -> str:
        # Return string representation without side effects (used by print())
        return self.summary()

    def to_dict(self):
        d = {
            "n_atoms": self.n_atoms,
            "frequencies_cm": self.frequencies_cm,
            "dip_moments": self.dip_moments,
            "delta_inhomogen_cm": self.delta_inhomogen_cm,
            "coupling_cm": self.coupling_cm,
        }
        return d


def pair_to_index(i: int, j: int, n: int) -> int:
    """0-based canonical index for |i,j> with i<j, given n atoms.
    ground=0, singles=1..n, doubles=(n+1)..
    Returns the index+1 of the basis corresponding to the double excitation |i,j>.
    """
    assert 1 <= i < j <= n
    return 1 + n + math.comb(j - 1, 2) + i


def index_to_pair(k: int, n: int) -> Tuple[int, int]:
    """Inverse map: from index k (in double block == corresponds to state basis[k-1]) to (atoms i,j excited)."""
    pair_rank = k - (1 + n)  # rank inside double block (0-based)

    # find j with C(j-1,2) < r <= C(j,2)
    j = 2
    while math.comb(j, 2) < pair_rank:
        j += 1
    i = pair_rank - math.comb(j - 1, 2)
    return i, j


''' 
Not used 


    def update_coupling_cm(self, new_coupling_cm: float) -> None:
        """Update base coupling (cm^-1) and refresh internal fs cache/coupling matrix."""
        self.coupling_cm = new_coupling_cm
        self.coupling_fs = float(convert_cm_to_fs(self.coupling_cm))
        # Coupling affects J matrix and H; recompute couplings and reset caches
        self._compute_isotropic_couplings()
        self.reset_cache()


'''
