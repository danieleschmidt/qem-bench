"""Probabilistic Error Cancellation (PEC) quasi-probability stub."""

import numpy as np
from typing import Callable, List, Tuple, Any


class ProbabilisticErrorCancellation:
    """Probabilistic Error Cancellation via quasi-probability decomposition (stub).

    This is a simplified Monte Carlo PEC stub demonstrating the quasi-probability
    framework. In full PEC, the ideal gate is decomposed into a quasi-probability
    distribution over implementable noisy operations.

    Args:
        noise_model_fn: Callable that applies noise to a circuit/state.
        gamma: Sampling overhead factor (default 1.5). The variance overhead is gamma^2.
    """

    def __init__(self, noise_model_fn: Callable = None, gamma: float = 1.5):
        self.noise_model_fn = noise_model_fn
        self.gamma = gamma

    def estimate(
        self,
        ideal_circuit_fn: Callable,
        noisy_circuit_fn: Callable,
        n_samples: int = 100,
    ) -> float:
        """Monte Carlo quasi-probability estimate of the ideal expectation value.

        Each sample draws a random sign (+1 or -1) with probability proportional to
        the quasi-probability weights, calls the noisy circuit function, and accumulates
        a weighted average. The gamma factor accounts for the 1-norm overhead.

        Args:
            ideal_circuit_fn: Callable returning the ideal expectation value (unused in
                sampling but available for comparison).
            noisy_circuit_fn: Callable returning a noisy expectation value sample.
            n_samples: Number of Monte Carlo samples.

        Returns:
            Estimated expectation value with error cancellation applied.
        """
        # Quasi-probability weights: w+ = (1 + 1/gamma)/2, w- = (1 - 1/gamma)/2
        # Signs: +1 with prob |w+|/norm, -1 with prob |w-|/norm
        # The gamma factor is the 1-norm of quasi-probs
        p_plus = (1.0 + 1.0 / self.gamma) / 2.0
        p_minus = 1.0 - p_plus

        total = 0.0
        for _ in range(n_samples):
            # Sample sign according to quasi-probability weights
            if np.random.random() < p_plus:
                sign = +1
            else:
                sign = -1

            sample = noisy_circuit_fn()
            total += sign * sample

        # Normalize: multiply by gamma (the 1-norm of quasi-probability distribution)
        return self.gamma * total / n_samples

    def quasi_probs(self, p: float) -> List[Tuple[str, float]]:
        """Quasi-probability decomposition for the bit-flip noise model.

        For bit-flip noise with error rate p, decomposes the ideal identity operation
        into a quasi-probability distribution over implementable operations.

        Args:
            p: Bit-flip error probability in [0, 1].

        Returns:
            List of (circuit_variant, coefficient) pairs where coefficient can be
            negative (quasi-probability).
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0, 1], got {p}")

        # For bit-flip channel E(rho) = (1-p)*rho + p*X@rho@X,
        # the inverse quasi-probability representation is:
        #   I = c_I * E + c_X * (X * E * X)
        # Solving: c_I = 1/(1-2p), c_X = -p/(1-2p)
        # (valid for p < 0.5)
        if abs(1.0 - p) < 1e-10:
            # At p=1, fully flipped; return trivial
            return [("identity", 0.0), ("bit_flip", 1.0)]

        # Decompose: I = c_I * E_p + c_X * X_gate
        # where E_p(rho) = (1-p)*rho + p*X@rho@X  (noisy channel)
        #   and X_gate(rho) = X@rho@X              (perfect X gate)
        # Solving: c_I*(1-p) = 1, c_I*p + c_X = 0
        # => c_I = 1/(1-p), c_X = -p/(1-p)   [sum = 1]
        c_identity = 1.0 / (1.0 - p)
        c_bit_flip = -p / (1.0 - p)

        return [
            ("identity", c_identity),
            ("bit_flip", c_bit_flip),
        ]
