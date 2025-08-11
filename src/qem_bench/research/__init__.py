"""
Advanced Research Module for QEM-Bench

Cutting-edge research capabilities for quantum error mitigation, including:
- Machine learning-powered QEM optimization
- Quantum-classical hybrid algorithms  
- Real-time adaptive error mitigation
- Novel QEM technique discovery
- Advanced experimental frameworks
"""

# Machine learning QEM optimization
from .ml_qem import (
    MLQEMOptimizer, QEMNeuralNetwork, AdaptiveParameterLearner,
    QEMReinforcementLearner, EnsembleQEMOptimizer
)

# Quantum-classical hybrid algorithms
from .hybrid_qem import (
    HybridQEMFramework, QuantumClassicalOptimizer,
    VariationalQEMOptimizer, HybridZNEOptimizer
)

# Real-time adaptive mitigation
from .adaptive_qem import (
    RealTimeQEMAdapter, DeviceDriftPredictor, AdaptiveNoisePredictor,
    DynamicMitigationSelector, ContextAwareMitigation
)

# Novel technique discovery
from .discovery import (
    QEMTechniqueDiscoverer, EvolutionaryQEMSearch,
    AutomaticQEMGenerator, NovelMitigationSynthesis
)

# Advanced experimental framework
from .experimental import (
    ResearchExperimentFramework, StatisticalSignificanceTester,
    MultiVariateQEMAnalysis, CausalInferenceEngine
)

# Research utilities
from .utils import (
    ResearchDataCollector, ExperimentReproducer,
    PublicationDataPreparer, BenchmarkValidator
)

__all__ = [
    # ML QEM
    'MLQEMOptimizer', 'QEMNeuralNetwork', 'AdaptiveParameterLearner',
    'QEMReinforcementLearner', 'EnsembleQEMOptimizer',
    
    # Hybrid algorithms
    'HybridQEMFramework', 'QuantumClassicalOptimizer',
    'VariationalQEMOptimizer', 'HybridZNEOptimizer',
    
    # Adaptive mitigation
    'RealTimeQEMAdapter', 'DeviceDriftPredictor', 'AdaptiveNoisePredictor',
    'DynamicMitigationSelector', 'ContextAwareMitigation',
    
    # Discovery
    'QEMTechniqueDiscoverer', 'EvolutionaryQEMSearch',
    'AutomaticQEMGenerator', 'NovelMitigationSynthesis',
    
    # Experimental framework
    'ResearchExperimentFramework', 'StatisticalSignificanceTester',
    'MultiVariateQEMAnalysis', 'CausalInferenceEngine',
    
    # Utilities
    'ResearchDataCollector', 'ExperimentReproducer',
    'PublicationDataPreparer', 'BenchmarkValidator'
]