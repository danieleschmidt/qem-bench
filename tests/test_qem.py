"""Tests for qem-bench: noise models, ZNE, PEC, and benchmark runner."""

import json
import os
import tempfile

import numpy as np
import pytest

from qem_bench.noise import NoiseModel
from qem_bench.zne import ZeroNoiseExtrapolation
from qem_bench.pec import ProbabilisticErrorCancellation
from qem_bench.benchmark import BenchmarkRunner

# --- Common fixtures ---

# |0><0| density matrix
RHO_0 = np.array([[1, 0], [0, 0]], dtype=complex)

# |1><1| density matrix
RHO_1 = np.array([[0, 0], [0, 1]], dtype=complex)

# |+><+| density matrix
RHO_PLUS = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ===== Noise model tests =====


def test_depolarizing_trace_preserved():
    """Depolarizing channel must preserve trace == 1."""
    rho = RHO_PLUS.copy()
    result = NoiseModel.depolarizing(rho, p=0.1)
    assert abs(np.trace(result) - 1.0) < 1e-10, f"Trace not 1: {np.trace(result)}"


def test_depolarizing_hermitian():
    """Depolarizing channel output must be Hermitian."""
    rho = RHO_PLUS.copy()
    result = NoiseModel.depolarizing(rho, p=0.3)
    assert np.allclose(result, result.conj().T), "Result is not Hermitian"


def test_bit_flip_identity_at_p0():
    """Bit-flip at p=0 should return the original state unchanged."""
    rho = RHO_0.copy()
    result = NoiseModel.bit_flip(rho, p=0.0)
    assert np.allclose(result, rho), "bit_flip at p=0 should be identity"


def test_bit_flip_at_p1():
    """Bit-flip at p=1 should fully flip |0> to |1> and vice versa."""
    result_0 = NoiseModel.bit_flip(RHO_0, p=1.0)
    assert np.allclose(result_0, RHO_1), f"Expected |1><1|, got {result_0}"

    result_1 = NoiseModel.bit_flip(RHO_1, p=1.0)
    assert np.allclose(result_1, RHO_0), f"Expected |0><0|, got {result_1}"


def test_phase_flip_identity_at_p0():
    """Phase-flip at p=0 should return the original state unchanged."""
    rho = RHO_PLUS.copy()
    result = NoiseModel.phase_flip(rho, p=0.0)
    assert np.allclose(result, rho), "phase_flip at p=0 should be identity"


def test_depolarizing_maximally_mixed_at_p1():
    """Depolarizing at p=1 should give maximally mixed state I/2."""
    # At p=1: result = 0 * rho + 1/4 * (X@rho@X + Y@rho@Y + Z@rho@Z + I@rho@I)
    # For any single-qubit state, this gives I/2
    result = NoiseModel.depolarizing(RHO_0, p=1.0)
    expected = np.eye(2, dtype=complex) / 2.0
    assert np.allclose(result, expected, atol=1e-10), f"Expected I/2, got {result}"


# ===== ZNE tests =====


def test_zne_linear_extrapolation():
    """Linear extrapolation should recover the zero-noise value from a linear model."""
    # Model: E(lambda) = 0.8 - 0.1 * lambda => E(0) = 0.8
    noise_levels = [1, 2, 3]
    expectations = [0.7, 0.6, 0.5]
    result = ZeroNoiseExtrapolation.linear_extrapolation(noise_levels, expectations)
    assert abs(result - 0.8) < 1e-8, f"Expected 0.8, got {result}"


def test_zne_richardson_at_zero_noise():
    """Richardson extrapolation should recover exact zero-noise value for polynomial."""
    # For a quadratic: E(lambda) = 1 - 0.1*lambda - 0.05*lambda^2, E(0) = 1.0
    noise_levels = [1, 2, 3]
    expectations = [1.0 - 0.1 * l - 0.05 * l ** 2 for l in noise_levels]
    zne = ZeroNoiseExtrapolation(noise_levels=noise_levels)
    result = zne.extrapolate(expectations)
    assert abs(result - 1.0) < 1e-8, f"Expected 1.0, got {result}"


def test_zne_single_point_returns_value():
    """With a single noise level, extrapolation should return that value."""
    zne = ZeroNoiseExtrapolation(noise_levels=[1])
    result = zne.extrapolate([0.42])
    assert abs(result - 0.42) < 1e-10, f"Expected 0.42, got {result}"


def test_zne_linear_single_point():
    """Linear extrapolation with a single point should return that value."""
    result = ZeroNoiseExtrapolation.linear_extrapolation([2], [0.7])
    assert abs(result - 0.7) < 1e-10, f"Expected 0.7, got {result}"


# ===== PEC tests =====


def test_pec_estimate_runs():
    """PEC estimate should run without errors and return a float."""
    pec = ProbabilisticErrorCancellation(gamma=1.5)

    def noisy():
        return 0.0 + np.random.normal(0, 0.01)

    def ideal():
        return 0.0

    result = pec.estimate(ideal, noisy, n_samples=50)
    assert isinstance(result, float), f"Expected float, got {type(result)}"


def test_pec_quasi_probs_sum_near_one():
    """Quasi-prob coefficients should sum to 1 (they form a resolution of identity)."""
    pec = ProbabilisticErrorCancellation()
    probs = pec.quasi_probs(p=0.1)
    total = sum(c for _, c in probs)
    assert abs(total - 1.0) < 1e-10, f"Quasi-probs sum to {total}, expected 1"


# ===== BenchmarkRunner tests =====


def test_benchmark_runner_keys():
    """BenchmarkRunner.run() must return dict with required keys."""
    runner = BenchmarkRunner()
    nm = NoiseModel()
    results = runner.run(nm, {"p": 0.01}, n_trials=3)
    for key in ("raw_expectation", "mitigated_zne", "mitigated_pec", "noise_params"):
        assert key in results, f"Missing key: {key}"


def test_benchmark_report_json():
    """save_report should write valid JSON with the expected keys."""
    runner = BenchmarkRunner()
    nm = NoiseModel()
    results = runner.run(nm, {"p": 0.01}, n_trials=3)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        runner.save_report(results, path)
        with open(path) as f:
            loaded = json.load(f)
        for key in ("raw_expectation", "mitigated_zne", "mitigated_pec", "noise_params"):
            assert key in loaded, f"Missing key in JSON: {key}"
    finally:
        os.unlink(path)
