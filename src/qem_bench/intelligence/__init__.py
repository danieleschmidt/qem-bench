"""
Intelligent Quantum Enhancement Module

Provides AI-powered optimization and intelligent adaptation for quantum error mitigation:
- Machine learning-driven parameter optimization
- Adaptive quantum circuit generation
- Intelligent noise characterization and prediction
- Self-optimizing quantum protocols
- Neural quantum state estimation
"""

# Core intelligent optimization
from .intelligent_optimizer import (
    IntelligentQuantumOptimizer, OptimizationStrategy, LearningMetrics,
    AdaptiveBayesianOptimizer, GradientFreeOptimizer, HybridOptimizer
)

# Neural quantum estimation
from .neural_quantum_estimator import (
    NeuralQuantumStateEstimator, QuantumNeuralNetwork, StateEstimationMetrics,
    VariationalQuantumEstimator, DeepQuantumEstimator
)

# Adaptive circuit generation
from .adaptive_circuit_generator import (
    AdaptiveCircuitGenerator, CircuitGenerationStrategy, CircuitOptimizer,
    QuantumCircuitEvolution, CircuitComplexityAnalyzer
)

# Intelligent noise modeling
from .intelligent_noise_modeler import (
    IntelligentNoiseModeler, NoiseModelingStrategy, AdaptiveNoisePredictor,
    TemporalNoiseAnalyzer, CrossTalkIntelligence
)

# Self-optimizing protocols
from .self_optimizing_protocols import (
    SelfOptimizingQEM, ProtocolEvolution, PerformanceFeedbackLoop,
    AdaptiveProtocolSelection, MetaOptimizer
)

__all__ = [
    # Intelligent optimization
    "IntelligentQuantumOptimizer",
    "OptimizationStrategy", 
    "LearningMetrics",
    "AdaptiveBayesianOptimizer",
    "GradientFreeOptimizer",
    "HybridOptimizer",
    
    # Neural quantum estimation
    "NeuralQuantumStateEstimator",
    "QuantumNeuralNetwork",
    "StateEstimationMetrics",
    "VariationalQuantumEstimator", 
    "DeepQuantumEstimator",
    
    # Adaptive circuit generation
    "AdaptiveCircuitGenerator",
    "CircuitGenerationStrategy",
    "CircuitOptimizer",
    "QuantumCircuitEvolution",
    "CircuitComplexityAnalyzer",
    
    # Intelligent noise modeling
    "IntelligentNoiseModeler",
    "NoiseModelingStrategy", 
    "AdaptiveNoisePredictor",
    "TemporalNoiseAnalyzer",
    "CrossTalkIntelligence",
    
    # Self-optimizing protocols
    "SelfOptimizingQEM",
    "ProtocolEvolution",
    "PerformanceFeedbackLoop",
    "AdaptiveProtocolSelection",
    "MetaOptimizer"
]