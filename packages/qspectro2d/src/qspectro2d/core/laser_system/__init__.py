"""
Laser system module for qspectro2d package.

This module provides functions and classes for defining and manipulating laser pulses,
including their electric field profiles and temporal shapes.

"""

"""Laser pulse models and field helpers."""

from .fields import e_pulses, epsilon_pulses, pulse_envelopes, single_pulse_envelope
from .laser import LaserPulse, LaserPulseSequence

__all__ = [
    "LaserPulse",
    "LaserPulseSequence",
    "e_pulses",
    "epsilon_pulses",
    "pulse_envelopes",
    "single_pulse_envelope",
]
