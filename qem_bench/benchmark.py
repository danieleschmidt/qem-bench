"""BenchmarkRunner for quantum error mitigation comparison."""

import json
import numpy as np
from typing import Any, Dict

from .noise import NoiseModel
from .zne import ZeroNoiseExtrapolation
from .pec import ProbabilisticErrorCancellation


# Pauli Z matrix for expectation value computation
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Ideal |+> state: rho = |+><+| = [[0.5, 0.5],[0.5, 0.5]]
_RHO_PLUS = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)


def _z_expectation(rho: np.ndarray) -> float:
    """Compute <Z> = Tr(Z @ rho) for a 1-qubit density matrix."""
    return float(np.real(np.trace(_Z @ rho)))


class BenchmarkRunner:
    """Run QEM benchmarks comparing raw, ZNE-mitigated, and PEC-mitigated results.

    Uses a canonical 1-qubit example: ideal state |+> with Z measurement
    (ideal expectation = 0). Noise is applied and mitigation methods are compared.
    """

    def run(
        self,
        noise_model: NoiseModel,
        noise_params: Dict[str, Any],
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """Run the benchmark.

        Args:
            noise_model: A NoiseModel instance.
            noise_params: Dict of noise parameters (must include 'p' for error prob).
            n_trials: Number of trials to average over.

        Returns:
            Dict with keys: raw_expectation, mitigated_zne, mitigated_pec, noise_params.
        """
        p = noise_params.get("p", 0.01)

        # --- Raw expectation (depolarizing noise at scale 1) ---
        raw_values = []
        for _ in range(n_trials):
            noisy_rho = noise_model.depolarizing(_RHO_PLUS, p)
            raw_values.append(_z_expectation(noisy_rho))
        raw_expectation = float(np.mean(raw_values))

        # --- ZNE: evaluate at noise levels [1, 2, 3] ---
        noise_levels = [1, 2, 3]
        zne = ZeroNoiseExtrapolation(noise_levels=noise_levels)
        zne_evals = []
        for scale in noise_levels:
            scaled_p = min(scale * p, 1.0)
            trial_vals = []
            for _ in range(n_trials):
                noisy_rho = noise_model.depolarizing(_RHO_PLUS, scaled_p)
                trial_vals.append(_z_expectation(noisy_rho))
            zne_evals.append(float(np.mean(trial_vals)))

        mitigated_zne = zne.extrapolate(zne_evals)

        # --- PEC: quasi-probability Monte Carlo ---
        def noisy_circuit_fn():
            noisy_rho = noise_model.bit_flip(_RHO_PLUS, p)
            return _z_expectation(noisy_rho)

        def ideal_circuit_fn():
            return _z_expectation(_RHO_PLUS)

        pec = ProbabilisticErrorCancellation(
            noise_model_fn=noise_model.bit_flip,
            gamma=1.0 / (1.0 - 2.0 * p) if p < 0.5 else 1.5,
        )
        mitigated_pec = pec.estimate(
            ideal_circuit_fn=ideal_circuit_fn,
            noisy_circuit_fn=noisy_circuit_fn,
            n_samples=200,
        )

        return {
            "raw_expectation": raw_expectation,
            "mitigated_zne": mitigated_zne,
            "mitigated_pec": mitigated_pec,
            "noise_params": noise_params,
        }

    def save_report(self, results: Dict[str, Any], path: str) -> None:
        """Save benchmark results to a JSON file.

        Args:
            results: Results dict from run().
            path: File path to write JSON output.
        """
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
