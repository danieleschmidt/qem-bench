# ADR-0002: JAX Backend Selection for Quantum Simulation

**Status:** Accepted  
**Date:** 2025-01-02  
**Deciders:** Core Development Team  

## Context

QEM-Bench requires a high-performance numerical computing backend for quantum circuit simulation, error mitigation calculations, and optimization tasks. The choice of backend significantly impacts performance, scalability, and development complexity.

Key requirements:
- GPU/TPU acceleration for large-scale simulations
- Automatic differentiation for gradient-based optimization
- Vectorization support for batch processing
- Integration with quantum computing libraries
- Active development and community support

## Decision

We will use JAX as the primary numerical computing backend for QEM-Bench.

## Rationale

### Alternatives Considered

1. **NumPy + SciPy**: 
   - ✅ Mature, stable, well-documented
   - ❌ No GPU acceleration, no automatic differentiation
   - ❌ Limited performance for large-scale simulations

2. **PyTorch**:
   - ✅ GPU acceleration, automatic differentiation
   - ✅ Large community, good documentation
   - ❌ Primarily designed for machine learning
   - ❌ Less suitable for quantum computing primitives

3. **TensorFlow**:
   - ✅ GPU/TPU support, mature ecosystem
   - ✅ TensorFlow Quantum integration available
   - ❌ More complex API, larger overhead
   - ❌ Less functional programming paradigm

4. **JAX** (Selected):
   - ✅ NumPy-compatible API with GPU/TPU acceleration
   - ✅ Automatic differentiation and vectorization
   - ✅ Functional programming paradigm suits quantum operations
   - ✅ JIT compilation for performance optimization
   - ✅ Growing adoption in scientific computing
   - ✅ Excellent for gradient-based error mitigation optimization

### Key Advantages of JAX

- **Performance**: JIT compilation and XLA optimization
- **Scalability**: Seamless CPU/GPU/TPU deployment
- **Research-Friendly**: Functional paradigm matches quantum operations
- **Differentiation**: Essential for variational error mitigation
- **Vectorization**: Efficient batch processing of quantum circuits
- **Composability**: Easy to combine transformations (jit, vmap, grad)

## Consequences

### Positive
- Significant performance improvements for large-scale simulations
- Enable GPU/TPU acceleration out of the box
- Support for gradient-based error mitigation optimization
- Efficient vectorized operations for batch processing
- Future-proof choice with growing scientific computing adoption
- Cleaner functional programming interface for quantum operations

### Negative
- Learning curve for developers unfamiliar with JAX
- Smaller ecosystem compared to PyTorch/TensorFlow
- Potential debugging complexity with JIT compilation
- Dependency on Google's XLA compiler
- Some quantum libraries may have limited JAX integration

### Neutral
- JAX version pinned to >=0.4.0 for stability
- Optional dependency: users can choose CPU-only or GPU-enabled installation
- Fallback NumPy implementations for unsupported operations
- Interoperability layer for converting to/from other frameworks
- Documentation includes JAX performance optimization guides

## Implementation Notes

### Core Components Using JAX
- Quantum state vector simulation
- Unitary matrix operations and decompositions
- Error mitigation calculations (ZNE, PEC, VD)
- Noise model simulation and sampling
- Optimization routines for adaptive mitigation
- Batch processing of benchmark circuits

### Performance Optimizations
- JIT compilation for hot paths
- Vectorization with `jax.vmap` for batch operations
- Device memory management for large simulations
- Gradient computation for variational methods
- Parallel processing across multiple devices

### Compatibility Strategy
- Abstract backend interface to support multiple frameworks
- NumPy fallback for operations not requiring acceleration
- Conversion utilities for interfacing with Qiskit/Cirq circuits
- Optional dependencies to avoid forcing JAX installation

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX for Scientific Computing](https://github.com/google/jax)
- [Performance Comparison: JAX vs NumPy vs PyTorch](https://blog.tensorflow.org/2020/03/jax-vs-tensorflow-benchmarks.html)
- [Quantum Machine Learning with JAX](https://arxiv.org/abs/2103.12469)