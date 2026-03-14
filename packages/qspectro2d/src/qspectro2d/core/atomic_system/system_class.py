# Compatibility shim
from .system import AtomicSystem
from .basis import pair_to_index, excitation_number_from_index

__all__ = ['AtomicSystem', 'pair_to_index', 'excitation_number_from_index']
