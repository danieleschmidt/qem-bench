"""Noise models for quantum error mitigation benchmarking."""

import numpy as np


# Pauli matrices
_I = np.array([[1, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class NoiseModel:
    """Collection of standard noise channels for 1-qubit density matrices."""

    @staticmethod
    def depolarizing(rho: np.ndarray, p: float) -> np.ndarray:
        """Apply depolarizing noise to a 1-qubit density matrix.

        The depolarizing channel is:
            E(rho) = (1 - p) * rho + (p/4) * (X@rho@X + Y@rho@Y + Z@rho@Z + I@rho@I)

        Args:
            rho: 2x2 numpy array representing the density matrix.
            p: Error probability in [0, 1].

        Returns:
            Noisy density matrix after applying depolarizing channel.
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        rho = np.asarray(rho, dtype=complex)
        kraus_sum = (
            _X @ rho @ _X
            + _Y @ rho @ _Y
            + _Z @ rho @ _Z
            + _I @ rho @ _I
        )
        return (1.0 - p) * rho + (p / 4.0) * kraus_sum

    @staticmethod
    def bit_flip(rho: np.ndarray, p: float) -> np.ndarray:
        """Apply bit-flip noise to a 1-qubit density matrix.

        The bit-flip channel is:
            E(rho) = (1 - p) * rho + p * X@rho@X

        Args:
            rho: 2x2 numpy array representing the density matrix.
            p: Error probability in [0, 1].

        Returns:
            Noisy density matrix after applying bit-flip channel.
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        rho = np.asarray(rho, dtype=complex)
        return (1.0 - p) * rho + p * (_X @ rho @ _X)

    @staticmethod
    def phase_flip(rho: np.ndarray, p: float) -> np.ndarray:
        """Apply phase-flip noise to a 1-qubit density matrix.

        The phase-flip channel is:
            E(rho) = (1 - p) * rho + p * Z@rho@Z

        Args:
            rho: 2x2 numpy array representing the density matrix.
            p: Error probability in [0, 1].

        Returns:
            Noisy density matrix after applying phase-flip channel.
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        rho = np.asarray(rho, dtype=complex)
        return (1.0 - p) * rho + p * (_Z @ rho @ _Z)
