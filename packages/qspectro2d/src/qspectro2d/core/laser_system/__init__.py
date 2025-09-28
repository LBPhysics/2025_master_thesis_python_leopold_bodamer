"""
Laser system module for qspectro2d package.

This module provides functions and classes for defining and manipulating laser pulses,
including their electric field profiles and temporal shapes.

"""

# LASER SYSTEM FUNCTIONS AND CLASSES

from .laser_fcts import e_pulses, epsilon_pulses, pulse_envelopes, single_pulse_envelope
from .laser_class import (
    LaserPulseSequence,
    LaserPulse,
)


# PUBLIC API

__all__ = [
    # functions
    "e_pulses",
    "epsilon_pulses",
    "pulse_envelopes",
    "single_pulse_envelope",
    # classes
    "LaserPulseSequence",
    "LaserPulse",
]


# VERSION INFO

__version__ = "1.0.0"
__author__ = "Leopold Bodamer"
__email__ = ""
