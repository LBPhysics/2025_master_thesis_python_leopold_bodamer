"""Central HEOM defaults shared across the spectroscopy pipeline.

The project only supports a single configuration style for the HEOM solver.
These constants capture the default values expected when optional fields are
omitted from the YAML/CLI configuration described in the README.
"""

HEOM_DEFAULT_MAX_DEPTH = 2
HEOM_DEFAULT_METHOD = "sd"
HEOM_DEFAULT_W_MIN = 1e-3
HEOM_DEFAULT_W_MAX_FACTOR = 6.0
HEOM_DEFAULT_N_POINTS = 400
HEOM_DEFAULT_N_EXP = 8
HEOM_DEFAULT_INCLUDE_DOUBLE = False
