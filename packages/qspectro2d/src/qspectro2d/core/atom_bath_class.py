# Compatibility shim -- real implementation lives in bath_coupling.py
from .bath_coupling import BathCoupling, lindblad_decay_channels, redfield_decay_channels

# Backward-compat alias: old code imported AtomBathCoupling from here
AtomBathCoupling = BathCoupling

__all__ = ["AtomBathCoupling", "BathCoupling", "lindblad_decay_channels", "redfield_decay_channels"]
