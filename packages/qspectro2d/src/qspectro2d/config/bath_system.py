"""
Bath system defaults for qspectro2d.
"""

# === BATH SYSTEM DEFAULTS ===
BATH_TYPE = "ohmic"  # TODO at the moment only ohmic baths are supported
BATH_CUTOFF = 1e2  # * frequencies[0]  # Cutoff frequency in cm⁻¹
BATH_TEMP = 1e-3  # * frequencies[0] / BOLTZMANN
BATH_COUPLING = 1e-4  # * frequencies[0]
