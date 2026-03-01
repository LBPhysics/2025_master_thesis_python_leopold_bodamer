from dataclasses import dataclass
import numpy as np
from qutip import BosonicEnvironment, Qobj

from .atomic_system import AtomicSystem
from ..config.atomic_system import DEPH_RATE_FS, DOWN_RATE_FS, UP_RATE_FS


@dataclass
class AtomBathCoupling:
    system: AtomicSystem
    bath: BosonicEnvironment

    @property
    def br_decay_channels(self) -> list[tuple[Qobj, BosonicEnvironment]]:
        """Generate the a_ops list for Bloch-Redfield solver.
        Includes:
          - Pure dephasing projectors for all excitation states
          - Radiative decay channels (lowering operators) from singles to ground
          - (If max_excitation == 2) lowering from double states to singles

        NOTE: redfield formalism usually builds rates from bath correlation functions; here we
        provide operator structures only. Coupling strengths are encoded in the bath object.
        """
        sys = self.system
        br_ops: list[tuple] = []
        n_atoms = sys.n_atoms

        # Helper: add dephasing projector for basis index k
        def projector(k: int):
            """Return site k population operator in the site basis (|k><k|)."""
            deph_op = sys.deph_op_i(k)
            return deph_op

        def add_op(op: Qobj):
            """add a system operator that couples to the bath"""
            br_ops.append([sys.to_eigenbasis(op), self.bath])

        for i_atom in range(1, n_atoms + 1):
            # singles manifold projectors
            add_op(projector(i_atom))

            # Radiative-like decay (lowering + raising) operators from singles to ground: |0><i| + |i><0|
            if n_atoms != 2:  # TO match the paper, no radiative decay in the dimer
                Li = sys.basis[0] * sys.basis[i_atom].dag()
                decay_op = Li + Li.dag()
                add_op(decay_op)

        # Double manifold projectors only if available
        max_exc = sys.max_excitation
        if max_exc == 2:
            for i_atom in range(1, n_atoms):
                for j_atom in range(i_atom + 1, n_atoms + 1):
                    from qspectro2d.core.atomic_system.system_class import pair_to_index

                    idx = pair_to_index(i_atom, j_atom, n_atoms)
                    add_op(projector(idx))
                    add_op(
                        projector(idx)
                    )  # each double excited state gets one contribution from each site
            if n_atoms == 2:  # to match the paper
                return br_ops
            for i_atom in range(1, n_atoms):  # otherwise add decay channels
                for j_atom in range(i_atom + 1, n_atoms + 1):
                    idx = pair_to_index(i_atom, j_atom, n_atoms)

                    # Double -> single lowering (if double manifold present)
                    # |i> corresponds to basis index i, etc.
                    Li = sys.basis[i_atom] * sys.basis[idx].dag()
                    op_ij_i = Li + Li.dag()
                    add_op(op_ij_i)
                    Lj = sys.basis[j_atom] * sys.basis[idx].dag()
                    op_ij_j = Lj + Lj.dag()
                    add_op(op_ij_j)

        return br_ops

    @property
    def me_decay_channels(self) -> list[Qobj]:
        """Generate hard coded rates-c_ops for Lindblad solver respecting max_excitation.

        - Pure dephasing: projectors on each populated basis state (singles; doubles if present)
        - Population relaxation/excitation: per-site lowering/raising (|0><i|, |i><0|)
          (double-manifold relaxation: |i,j><i| and |i,j><j|)
        """
        sys = self.system
        n_atoms = sys.n_atoms
        c_ops = []
        add_op = lambda op, rate: c_ops.append(sys.to_eigenbasis(op) * np.sqrt(rate))

        for i_atom in range(1, n_atoms + 1):
            # singles dephasing
            deph_op = sys.deph_op_i(i_atom)
            add_op(deph_op, DEPH_RATE_FS)

            # Radiative-like single-site relaxation (singles -> ground) and thermal excitation
            if n_atoms != 2:  # TO match the paper, no radiative decay in the dimer
                L_down = sys.basis[0] * sys.basis[i_atom].dag()  # |0><i|
                add_op(L_down, DOWN_RATE_FS)
                if UP_RATE_FS > 0:
                    L_up = sys.basis[i_atom] * sys.basis[0].dag()  # |i><0|
                    add_op(L_up, UP_RATE_FS)

        # Double-state dephasing if manifold present
        max_exc = sys.max_excitation
        if max_exc == 2:
            for i_atom in range(1, n_atoms):
                for j_atom in range(i_atom + 1, n_atoms + 1):
                    from qspectro2d.core.atomic_system.system_class import pair_to_index

                    idx = pair_to_index(i_atom, j_atom, n_atoms)
                    deph_op_ij = sys.deph_op_i(idx)
                    add_op(deph_op_ij, DEPH_RATE_FS)
                    add_op(
                        deph_op_ij, DEPH_RATE_FS
                    )  # each double excited state gets one contribution from each site
                    # TODO check wheather or not: twice add_op(deph_op_ij, DEPH_RATE_FS) does the same as add_op(deph_op_ij + deph_op_ij, DEPH_RATE_FS) and add_op(deph_op_ij - deph_op_ij, DEPH_RATE_FS) <- (anti-)correlation??
            if n_atoms == 2:  # to match the paper
                return c_ops
            for i_atom in range(1, n_atoms):  # otherwise add decay channels
                for j_atom in range(i_atom + 1, n_atoms + 1):
                    idx = pair_to_index(i_atom, j_atom, n_atoms)

                    # Double -> single lowering
                    L_idx_i = sys.basis[i_atom] * sys.basis[idx].dag()  # |i><i,j|
                    L_idx_j = sys.basis[j_atom] * sys.basis[idx].dag()  # |j><i,j|
                    add_op(L_idx_i, DOWN_RATE_FS)
                    add_op(L_idx_j, DOWN_RATE_FS)

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

    # --- Paper solver rates (JCP 124, 234504/234505) ---
    # Look appendix C in JCP 124, 234504   
    # ω_{αβ} = E_α − E_β and S is the bath power spectrum.
    def paper_gamma_ab(self, a: int, b: int) -> float:
        """Population transfer rate γ_{αβ} for the dimer eigenbasis.
        Eq. (C1) γ_{αβ} = sin²(2θ) · S(ω_{αβ}), αβ ∈ {1,2} (single-exciton states)
        """
        w_ab = self.system.omega_ij(a, b)
        S_wab = self.bath.power_spectrum(w_ab)
        sin2_2theta = np.sin(2 * self.theta) ** 2
        return sin2_2theta * S_wab

    def paper_Gamma_ab(self, a: int, b: int) -> float:
        """Total dephasing / population decay rate Γ_{αβ} for the paper Liouvillian.
        Eq. (C2-C6)
        Combines pure dephasing (∝ S(0)) with population-transfer contributions
        as given in the Redfield tensor.
        Parameters
        ----------
        a, b : int
            Eigenbasis indices of the density-matrix element ρ_{αβ}.

        Returns
        -------
        float
            Total dephasing rate Γ_{αβ} [fs⁻¹].

        Raises
        ------
        ValueError
            If (a, b) is not a valid dimer eigenbasis index pair.
        """
        S_0 = self.bath.power_spectrum(0)
        sin2 = np.sin(2 * self.theta) ** 2
        cos2 = np.cos(2 * self.theta) ** 2

        # Pure-dephasing building blocks (Appendix A, Table I notation)
        Gamma_ab = 2 * cos2 * S_0                # Γ̃_{αβ}  (exciton coherence)
        Gamma_a0 = (1 - 0.5 * sin2) * S_0      # Γ̃_{α0}  (optical coherence)
        Gamma_abar_0 = 2 * S_0                   # Γ̃_{ā,0} (double ↔ ground)
        Gamma_abar_a = Gamma_a0                # Γ̃_{ā,α} (holds for dimer)

        # Pre-compute the two independent population-transfer rates
        gamma_21 = self.paper_gamma_ab(2, 1)     # γ_{β→α}
        gamma_12 = self.paper_gamma_ab(1, 2)     # γ_{α→β}
        gamma_sum = gamma_21 + gamma_12

        if a == 1:
            if b == 0:          # ρ_{α,g} optical coherence
                return Gamma_a0 + 0.5 * gamma_21
            elif b == 1:        # ρ_{α,α} population
                return gamma_21
            elif b == 2:        # ρ_{α,β} exciton coherence
                return Gamma_ab + 0.5 * gamma_sum
        elif a == 2:
            if b == 0:          # ρ_{β,g} optical coherence
                return Gamma_a0 + 0.5 * gamma_12
            elif b == 1:        # ρ_{β,α} exciton coherence
                return Gamma_ab + 0.5 * gamma_sum
            elif b == 2:        # ρ_{β,β} population
                return gamma_12
        elif a == 3:
            if b == 0:          # ρ_{ā,g} double ↔ ground coherence
                return Gamma_abar_0
            elif b == 1:        # ρ_{ā,α}
                return Gamma_abar_a + 0.5 * gamma_21
            elif b == 2:        # ρ_{ā,β}
                return Gamma_abar_a + 0.5 * gamma_12

        raise ValueError(
            f"paper_Gamma_ab: invalid dimer index pair (α={a}, β={b}). "
            f"Expected α ∈ {{1,2,3}}, β ∈ {{0,1,2}}."
        )

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

