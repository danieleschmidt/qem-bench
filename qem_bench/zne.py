"""Zero-Noise Extrapolation (ZNE) for quantum error mitigation."""

import numpy as np
from typing import List


class ZeroNoiseExtrapolation:
    """Zero-Noise Extrapolation using Richardson or linear extrapolation.

    Args:
        noise_levels: List of noise scale factors (e.g., [1, 2, 3]).
    """

    def __init__(self, noise_levels: List[float] = None):
        self.noise_levels = noise_levels if noise_levels is not None else [1, 2, 3]

    def extrapolate(self, expectation_values: List[float]) -> float:
        """Richardson extrapolation to zero noise.

        For n points, solves the Vandermonde system to find coefficients c_i such that:
            result = sum_i c_i * E_i
        where the polynomial extrapolates to noise_level = 0.

        Args:
            expectation_values: List of expectation values at each noise level.

        Returns:
            Extrapolated expectation value at zero noise.
        """
        lambdas = np.array(self.noise_levels[: len(expectation_values)], dtype=float)
        evals = np.array(expectation_values, dtype=float)

        if len(evals) == 1:
            return float(evals[0])

        # Build Vandermonde-like system: sum_i c_i * lambda_i^k = delta_{k,0}
        # i.e., we want sum_i c_i * f(lambda_i) to extrapolate a polynomial to lambda=0
        # Vandermonde matrix V[k, i] = lambda_i^k
        n = len(lambdas)
        V = np.vander(lambdas, n, increasing=True).T  # shape (n, n)

        # Right-hand side: e_0 = [1, 0, 0, ...] (extrapolate constant term)
        rhs = np.zeros(n)
        rhs[0] = 1.0

        # Solve V @ c = rhs for coefficients
        coeffs = np.linalg.solve(V, rhs)

        return float(np.dot(coeffs, evals))

    @staticmethod
    def linear_extrapolation(
        noise_levels: List[float], expectation_values: List[float]
    ) -> float:
        """Linear extrapolation to zero noise using numpy polyfit.

        Fits a line to (noise_level, expectation_value) pairs and evaluates at 0.

        Args:
            noise_levels: List of noise scale factors.
            expectation_values: List of expectation values at each noise level.

        Returns:
            Extrapolated expectation value at zero noise.
        """
        lambdas = np.array(noise_levels, dtype=float)
        evals = np.array(expectation_values, dtype=float)

        if len(evals) == 1:
            return float(evals[0])

        # Fit degree-1 polynomial and evaluate at 0
        coeffs = np.polyfit(lambdas, evals, 1)
        poly = np.poly1d(coeffs)
        return float(poly(0.0))
