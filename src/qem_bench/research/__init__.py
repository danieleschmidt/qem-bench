"""
Advanced Research Module for QEM-Bench

Cutting-edge research capabilities for quantum error mitigation, including:
- Machine learning-powered QEM optimization
- Quantum-classical hybrid algorithms  
- Real-time adaptive error mitigation
- Novel QEM technique discovery
- Advanced experimental frameworks
- Novel quantum syndrome correlation learning
- Cross-platform error model transfer learning
- Causal inference for adaptive QEM
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

# Novel Research Implementations (Generation 4+)

# Quantum-Enhanced Error Syndrome Correlation Learning
from .quantum_syndrome_learning import (
    QuantumSyndromeEncoder, QuantumCorrelationPredictor,
    QuantumSyndromeLearningFramework, ErrorSyndromeData,
    QuantumFeatureMap, create_research_benchmark,
    run_research_validation
)

# Cross-Platform Error Model Transfer Learning
from .cross_platform_transfer import (
    UniversalErrorRepresentation, CrossPlatformTransferLearning,
    PlatformCharacteristics, ErrorModelFingerprint,
    TransferLearningDataset, PlatformType,
    create_platform_characteristics, generate_synthetic_error_data,
    create_transfer_learning_benchmark, run_cross_platform_validation
)

# Real-Time Adaptive QEM with Causal Inference
from .causal_adaptive_qem import (
    CausalInferenceEngine, RealTimeAdaptiveQEM,
    CausalEvent, ErrorBurst, CausalGraph,
    CausalEventType, create_causal_qem_benchmark,
    run_causal_adaptive_validation
)

# Integrated Research Validation Framework
from .integrated_validation import (
    IntegratedResearchValidation, ResearchValidationResults,
    ComparativeStudy, run_comprehensive_research_validation
)

# Generation 4: Advanced Research Modules

# Quantum Coherence Preservation
from .quantum_coherence_preservation import (
    DynamicalDecouplingProtocol, AdaptiveCoherencePreservation,
    QuantumErrorSuppression, CoherenceResearchFramework,
    create_coherence_preservation_system
)

# Quantum Advantage Detection  
from .quantum_advantage_detection import (
    RandomCircuitSamplingAdvantage, VariationalQuantumAdvantage,
    QuantumMachineLearningAdvantage, CompositeQuantumAdvantageFramework,
    create_quantum_advantage_detector
)

# Quantum Neural Architecture Search
from .quantum_neural_architecture_search import (
    QuantumArchitectureGenome, QuantumNASConfig,
    QuantumCircuitSimulator, QuantumNeuralArchitectureSearch,
    create_quantum_nas
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
    'PublicationDataPreparer', 'BenchmarkValidator',
    
    # Novel Research Implementations
    # Quantum Syndrome Learning
    'QuantumSyndromeEncoder', 'QuantumCorrelationPredictor',
    'QuantumSyndromeLearningFramework', 'ErrorSyndromeData',
    'QuantumFeatureMap', 'create_research_benchmark',
    'run_research_validation',
    
    # Cross-Platform Transfer Learning
    'UniversalErrorRepresentation', 'CrossPlatformTransferLearning',
    'PlatformCharacteristics', 'ErrorModelFingerprint',
    'TransferLearningDataset', 'PlatformType',
    'create_platform_characteristics', 'generate_synthetic_error_data',
    'create_transfer_learning_benchmark', 'run_cross_platform_validation',
    
    # Causal Adaptive QEM
    'CausalInferenceEngine', 'RealTimeAdaptiveQEM',
    'CausalEvent', 'ErrorBurst', 'CausalGraph',
    'CausalEventType', 'create_causal_qem_benchmark',
    'run_causal_adaptive_validation',
    
    # Integrated Validation
    'IntegratedResearchValidation', 'ResearchValidationResults',
    'ComparativeStudy', 'run_comprehensive_research_validation',
    
    # Generation 4: Advanced Research Modules
    # Quantum Coherence Preservation
    'DynamicalDecouplingProtocol', 'AdaptiveCoherencePreservation', 
    'QuantumErrorSuppression', 'CoherenceResearchFramework',
    'create_coherence_preservation_system',
    
    # Quantum Advantage Detection
    'RandomCircuitSamplingAdvantage', 'VariationalQuantumAdvantage',
    'QuantumMachineLearningAdvantage', 'CompositeQuantumAdvantageFramework',
    'create_quantum_advantage_detector',
    
    # Quantum Neural Architecture Search
    'QuantumArchitectureGenome', 'QuantumNASConfig',
    'QuantumCircuitSimulator', 'QuantumNeuralArchitectureSearch',
    'create_quantum_nas'
]