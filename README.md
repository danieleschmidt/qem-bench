# qem-bench

Quantum Error Mitigation benchmarking toolkit. Implements Zero-Noise Extrapolation (ZNE) with Richardson extrapolation and Probabilistic Error Cancellation (PEC) stubs for benchmarking against raw noisy expectation values.

## Install

```bash
pip install numpy
```

## Usage

```python
from qem_bench.benchmark import BenchmarkRunner
from qem_bench.noise import NoiseModel

runner = BenchmarkRunner()
results = runner.run(NoiseModel(), {'p': 0.01})
```

## Modules

- **`qem_bench.noise`** — `NoiseModel` with `depolarizing`, `bit_flip`, and `phase_flip` channels
- **`qem_bench.zne`** — `ZeroNoiseExtrapolation` with Richardson and linear extrapolation
- **`qem_bench.pec`** — `ProbabilisticErrorCancellation` quasi-probability stub
- **`qem_bench.benchmark`** — `BenchmarkRunner` for end-to-end benchmarking

## Example Output

```python
{
  "raw_expectation": -0.005,
  "mitigated_zne": 0.0001,
  "mitigated_pec": -0.0003,
  "noise_params": {"p": 0.01}
}
```
