"""
Atomic system defaults for qspectro2d.
"""

# === ATOMIC SYSTEM DEFAULTS ===
N_ATOMS = 1
N_CHAINS = 1  # defaults to linear chain (single chain layout)
FREQUENCIES_CM = [16000.0] * N_ATOMS  # Number of frequency components in the system
DIP_MOMENTS = [1.0] * N_ATOMS  # Dipole moments for each atom
COUPLING_CM = 0.0  # Coupling strength [cm⁻¹]
DELTA_INHOMOGEN_CM = 0.0  # Inhomogeneous broadening [cm⁻¹]
MAX_EXCITATION = 1  # 1 -> ground+single manifold, 2 -> add double-excitation manifold
N_INHOMOGEN = 1  # 1 == no inhomogeneous broadening
