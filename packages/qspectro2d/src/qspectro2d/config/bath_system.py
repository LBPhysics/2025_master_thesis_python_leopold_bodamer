"""
Bath system defaults for qspectro2d.
"""

# === BATH SYSTEM DEFAULTS ===
BATH_TYPE = "ohmic"  # TODO at the moment only ohmic and Drude-Lorentzian baths are supported
# Bath parameters are specified as dimensionless multiples of ω0̄.
# The loader converts them to internal fs^-1 units via ω0̄ = mean(convert_cm_to_fs(atomic.frequencies_cm)).
BATH_CUTOFF = 1e2
BATH_TEMP = 1e-2
BATH_COUPLING = 1e-4
