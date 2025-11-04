"""
Signal processing and phase cycling defaults for qspectro2d.
"""

import numpy as np

# === signal processing / phase cycling ===
N_PHASES = 4  # Number of phase cycles for the simulation

DPHI = 2 * np.pi / N_PHASES

PHASE_CYCLING_PHASES = DPHI * np.arange(N_PHASES)

SIGNAL_TYPES = ["rephasing", "nonrephasing"]  # Default signal == photon echo to simulate

COMPONENT_MAP: dict[str, tuple[int, int]] = {
    "average": (0, 0),  # special case for just averaging all phases
    "rephasing": (-1, 1),  # photon echo is extracted here
    "nonrephasing": (1, -1),
    "doublequantum": (1, 1),
}  # represents the (k1, k2) phase factors for each signal type [k3 doenst matter]

# last pulse is 10% of the first two to ensure probing character
RELATIVE_E0S = [1.0, 1.0, 0.1]

# Validation thresholds for physics checks
NEGATIVE_EIGVAL_THRESHOLD = -1e-3
TRACE_TOLERANCE = 1e-6
