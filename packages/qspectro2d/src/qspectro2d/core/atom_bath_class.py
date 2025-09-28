# TODO potentially Replace hard‑coded deph_rate and down_rate with rates computed from bath.power_spectrum(self.system.omega_ij(...)), and add thermal raising where applicable
from dataclasses import dataclass
import numpy as np
from qutip import BosonicEnvironment

from .atomic_system import AtomicSystem


@dataclass
class AtomBathCoupling:
    system: AtomicSystem
    bath: BosonicEnvironment

    @property
    def br_decay_channels(self):
        """Generate the a_ops list for Bloch-Redfield solver.
        Includes:
          - Pure dephasing projectors for all excitation states
          - Radiative decay channels (lowering operators) from singles to ground
          - (If max_excitation == 2) lowering from double states to singles

        NOTE: BR formalism usually builds rates from bath correlation functions; here we
        provide operator structures only. Coupling strengths are encoded in the bath object.
        """
        sys = self.system
        br_ops: list[list] = []
        n_atoms = sys.n_atoms

        # Helper: add dephasing projector for basis index k
        def add_projector(k: int):
            """Return site k population operator in the site basis (|k><k|)."""
            if k == 0:
                raise ValueError("indexing ground state -> use k elem 1,...,N+1")
            deph_op = sys.deph_op_i(k)
            br_ops.append([sys.to_eigenbasis(deph_op), self.bath])

        for i_atom in range(1, n_atoms + 1):
            # singles manifold projectors
            add_projector(i_atom)

            # Radiative-like decay (lowering + raising) operators from singles to ground: |0><i| + |i><0|
            Li = sys.basis[0] * sys.basis[i_atom].dag()
            decay_op = Li + Li.dag()
            br_ops.append([sys.to_eigenbasis(decay_op), self.bath])

        # Double manifold projectors only if available
        max_exc = sys.max_excitation
        if max_exc == 2:
            for i_atom in range(1, n_atoms):
                for j_atom in range(i_atom + 1, n_atoms + 1):
                    from qspectro2d.core.atomic_system.system_class import pair_to_index

                    idx = pair_to_index(i_atom, j_atom, n_atoms)
                    add_projector(idx - 1)
                    add_projector(
                        idx - 1
                    )  # each double excited state gets one contribution from each site

                    # Double -> single lowering (if double manifold present)
                    # |i> corresponds to basis index i, etc.
                    Li = sys.basis[i_atom] * sys.basis[idx - 1].dag()
                    op_ij_i = Li + Li.dag()
                    br_ops.append([sys.to_eigenbasis(op_ij_i), self.bath])

                    Lj = sys.basis[j_atom] * sys.basis[idx - 1].dag()
                    op_ij_j = Lj + Lj.dag()
                    br_ops.append([sys.to_eigenbasis(op_ij_j), self.bath])

        return br_ops

    @property
    def me_decay_channels(self):
        """Generate c_ops for Lindblad solver respecting max_excitation.

        Strategy:
          - Pure dephasing: projectors on each populated basis state (singles; doubles if present)
          - Population relaxation/excitation: per-site lowering/raising (|0><i|, |i><0|) with rates from bath
            (double-manifold relaxation: |i,j><i| and |i,j><j| with rates from bath)
        """
        sys = self.system
        n_atoms = sys.n_atoms
        c_ops = []

        # Dephasing rate (assumed identical structure for all single excitations)
        deph_rate = 1 / 100
        down_rate = 1 / 300

        for i_atom in range(1, n_atoms + 1):
            # singles dephasing
            deph_op = sys.deph_op_i(i_atom)
            c_ops.append(sys.to_eigenbasis(deph_op) * np.sqrt(deph_rate))

            # Radiative-like single-site relaxation (singles -> ground) and thermal excitation
            L_down = sys.basis[0] * sys.basis[i_atom].dag()  # |0><i|
            c_ops.append(sys.to_eigenbasis(L_down) * np.sqrt(down_rate))

        # Double-state dephasing if manifold present
        if sys.max_excitation == 2:
            for i in range(1, n_atoms):
                for j in range(i + 1, n_atoms + 1):
                    from qspectro2d.core.atomic_system.system_class import pair_to_index

                    idx = pair_to_index(i, j, n_atoms)
                    deph_op_ij = sys.deph_op_i(idx - 1)
                    c_ops.append(sys.to_eigenbasis(deph_op_ij) * np.sqrt(deph_rate))
                    c_ops.append(
                        sys.to_eigenbasis(deph_op_ij) * np.sqrt(deph_rate)
                    )  # each double excited state gets one contribution from each site

                    # Double -> single lowering
                    L_idx_i = sys.basis[i] * sys.basis[idx - 1].dag()  # |i><i,j|
                    L_idx_j = sys.basis[j] * sys.basis[idx - 1].dag()  # |j><i,j|
                    c_ops.append(sys.to_eigenbasis(L_idx_i) * np.sqrt(down_rate))
                    c_ops.append(sys.to_eigenbasis(L_idx_j) * np.sqrt(down_rate))

        return c_ops

    @property
    def theta(self) -> float:
        """Return dimer mixing angle θ (radians) for n_atoms == 2.

        Definition (standard exciton / coupled two-level system):

            tan(2θ) = 2J / Δ

        where
            J  = coupling (fs^-1)
            Δ  = ω_1 - ω_2 (fs^-1)  (bare transition frequency detuning)

        We compute:
            θ = 0.5 * arctan2(2J, Δ)

        Range: θ ∈ (-π/4, π/4]; magnitude governs the degree of state mixing.
        The previous implementation used arctan(J/Δ), which differs by a
        factor-of-two in the argument and is non-standard. Coefficients used
        in `lowering_op` retain their algebraic definitions (sinθ, cosθ), so this
        correction yields physically consistent mixing when J ~ Δ.
        """
        if self.system.n_atoms != 2:
            raise ValueError("theta is only defined for n_atoms == 2")
        detuning = self.system.frequencies_fs[0] - self.system.frequencies_fs[1]  # Δ
        return 0.5 * np.arctan2(2 * self.system.coupling_fs, detuning)

    # only for the paper solver
    def paper_gamma_ij(self, i: int, j: int) -> float:
        """
        Calculate the population relaxation rates. for the dimer system, analogous to the gamma_ij in the paper.

        Parameters:
            i (int): Index of the first state.
            j (int): Index of the second state.

        Returns:
            float: Population relaxation rate.
        """
        w_ij = self.system.omega_ij(i, j)
        """
        from qspectro2d.core.bath_system.bath_fcts import (
            power_spectrum_func_paper,
            extract_bath_parameters,
        )

        args = extract_bath_parameters(self.bath)
        
        args["alpha"] = args["alpha"] * args["wc"]  # rescale coupling for paper eqs
        P_wij = power_spectrum_func_paper(w_ij, **args)
        """
        P_wij = self.bath.power_spectrum(w_ij)
        return np.sin(2 * self.theta) ** 2 * P_wij

    def paper_Gamma_ij(self, i: int, j: int) -> float:
        """
        Calculate the pure dephasing rates. for the dimer system, analogous to the gamma_ij in the paper.

        Parameters:
            i (int): Index of the first state.
            j (int): Index of the second state.

        Returns:
            float: Pure dephasing rate.
        """
        # Pure dephasing rates helper
        """
        from qspectro2d.core.bath_system.bath_fcts import (
            power_spectrum_func_paper,
            extract_bath_parameters,
        )

        args = extract_bath_parameters(self.bath)
        args["alpha"] = args["alpha"] * args["wc"]  # rescale coupling for paper eqs
        P_0 = power_spectrum_func_paper(0, **args)
        """
        P_0 = self.bath.power_spectrum(0)
        Gamma_t_ab = 2 * np.cos(2 * self.theta) ** 2 * P_0  # tilde
        Gamma_t_a0 = (1 - 0.5 * np.sin(2 * self.theta) ** 2) * P_0
        Gamma_11 = self.paper_gamma_ij(2, 1)
        Gamma_22 = self.paper_gamma_ij(1, 2)
        Gamma_abar_0 = 2 * P_0
        Gamma_abar_a = Gamma_abar_0  # holds for dimer
        if i == 1:
            if j == 0:
                return Gamma_t_a0 + 0.5 * self.paper_gamma_ij(2, i)
            elif j == 1:
                return Gamma_11
            elif j == 2:
                return Gamma_t_ab + 0.5 * (
                    self.paper_gamma_ij(i, j) + self.paper_gamma_ij(j, i)
                )
        if i == 2:
            if j == 0:
                return Gamma_t_a0 + 0.5 * self.paper_gamma_ij(1, i)
            elif j == 1:
                return Gamma_t_ab + 0.5 * (
                    self.paper_gamma_ij(i, j) + self.paper_gamma_ij(j, i)
                )
            elif j == 2:
                return Gamma_22
        elif i == 3:
            if j == 0:
                return Gamma_abar_0
            elif j == 1:
                return Gamma_abar_a + 0.5 * (self.paper_gamma_ij(2, j))
            elif j == 2:
                return Gamma_abar_a + 0.5 * (self.paper_gamma_ij(1, j))
        else:
            raise ValueError("Invalid indices for i and j.")

    def summary(self) -> str:
        lines = [
            "=== AtomBathCoupling Summary ===",
            "System Parameters:",
            str(self.system),
            "Bath Parameters:",
            str(self.bath),
            "Decay Channels:",
            f"  Bloch-Redfield decay channels: {len(self.br_decay_channels)}",
            f"  Lindblad decay channels: {len(self.me_decay_channels)}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
