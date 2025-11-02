"""
Laser system defaults for qspectro2d.
"""

from .atomic_system import N_ATOMS

# === LASER SYSTEM DEFAULTS ===
PULSE_FWHM_FS = 15.0 if N_ATOMS == 1 else 5.0  # Pulse FWHM in fs
BASE_AMPLITUDE = 0.01  # -> such that for 1 atom the |exe| < 1%
ENVELOPE_TYPE = "gaussian"  # Type of pulse envelope # gaussian or cos2
CARRIER_FREQ_CM = 16000.0  # np.mean(FREQUENCIES_CM)  # Carrier frequency of the laser
RWA_SL = True
