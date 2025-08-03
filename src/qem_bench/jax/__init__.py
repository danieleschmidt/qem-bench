"""JAX-accelerated quantum simulation backend."""

from .simulator import JAXSimulator
from .circuits import JAXCircuit, create_jax_circuit
from .gates import (
    PauliX, PauliY, PauliZ,
    Hadamard, Phase, CX, CZ,
    RX, RY, RZ
)
from .states import (
    zero_state, 
    plus_state,
    create_ghz_state,
    create_bell_state
)
from .observables import (
    PauliObservable,
    ZObservable,
    create_pauli_observable
)

__all__ = [
    "JAXSimulator",
    "JAXCircuit",
    "create_jax_circuit", 
    "PauliX", "PauliY", "PauliZ",
    "Hadamard", "Phase", "CX", "CZ",
    "RX", "RY", "RZ",
    "zero_state", "plus_state",
    "create_ghz_state", "create_bell_state",
    "PauliObservable", "ZObservable", 
    "create_pauli_observable"
]