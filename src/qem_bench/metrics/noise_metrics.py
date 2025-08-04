"""Noise model analysis and metrics for QEM-Bench."""

import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class NoiseAnalysis:
    """Comprehensive analysis of a noise model."""
    noise_model_id: str
    timestamp: float
    
    # Basic properties
    num_qubits: int
    noise_type: str  # e.g., "composite", "depolarizing", "amplitude_damping"
    
    # Error rates
    single_qubit_error_rates: Dict[int, float] = field(default_factory=dict)
    two_qubit_error_rates: Dict[Tuple[int, int], float] = field(default_factory=dict)
    readout_error_rates: Dict[int, float] = field(default_factory=dict)
    
    # Statistical properties
    avg_single_qubit_error: float = 0.0
    avg_two_qubit_error: float = 0.0
    avg_readout_error: float = 0.0
    max_error_rate: float = 0.0
    min_error_rate: float = 0.0
    
    # Coherence properties
    t1_times: Dict[int, float] = field(default_factory=dict)  # Relaxation times
    t2_times: Dict[int, float] = field(default_factory=dict)  # Dephasing times
    
    # Crosstalk analysis
    crosstalk_matrix: Optional[np.ndarray] = None
    crosstalk_strength: float = 0.0
    
    # Correlation and structure
    error_correlations: Dict[Tuple[int, int], float] = field(default_factory=dict)
    noise_structure: str = "unknown"  # "independent", "correlated", "structured"
    
    # Quality metrics
    fidelity_estimates: Dict[str, float] = field(default_factory=dict)
    diamond_norm: Optional[float] = None
    process_fidelity: Optional[float] = None
    
    # Performance impact
    mitigation_difficulty: float = 0.0  # 0=easy to mitigate, 1=very difficult
    recommended_methods: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def overall_error_rate(self) -> float:
        """Weighted average of all error rates."""
        total_weighted_error = 0.0
        total_weight = 0.0
        
        # Single-qubit errors (weight = 1)
        for error_rate in self.single_qubit_error_rates.values():
            total_weighted_error += error_rate
            total_weight += 1
        
        # Two-qubit errors (weight = 2, as they're typically more significant)
        for error_rate in self.two_qubit_error_rates.values():
            total_weighted_error += error_rate * 2
            total_weight += 2
        
        # Readout errors (weight = 0.5, as they can sometimes be corrected)
        for error_rate in self.readout_error_rates.values():
            total_weighted_error += error_rate * 0.5
            total_weight += 0.5
        
        return total_weighted_error / total_weight if total_weight > 0 else 0.0
    
    @property
    def coherence_quality(self) -> float:
        """Overall coherence quality metric (0=poor, 1=excellent)."""
        if not self.t1_times and not self.t2_times:
            return 0.5  # Unknown
        
        # Normalize based on typical values (in microseconds)
        typical_t1 = 100.0  # 100 μs
        typical_t2 = 50.0   # 50 μs
        
        t1_quality = 0.0
        t2_quality = 0.0
        
        if self.t1_times:
            avg_t1 = np.mean(list(self.t1_times.values()))
            t1_quality = min(1.0, avg_t1 / typical_t1)
        
        if self.t2_times:
            avg_t2 = np.mean(list(self.t2_times.values()))
            t2_quality = min(1.0, avg_t2 / typical_t2)
        
        # Combined quality
        if self.t1_times and self.t2_times:
            return (t1_quality + t2_quality) / 2
        elif self.t1_times:
            return t1_quality
        else:
            return t2_quality


class NoiseMetrics:
    """
    Noise model analyzer for extracting metrics and properties.
    
    Analyzes noise models to extract error rates, coherence properties,
    correlation patterns, and provide recommendations for error mitigation
    strategies.
    
    Example:
        >>> analyzer = NoiseMetrics()
        >>> 
        >>> # Analyze a noise model
        >>> analysis = analyzer.analyze_noise_model(noise_model, "device_noise")
        >>> print(f"Overall error rate: {analysis.overall_error_rate:.4f}")
        >>> print(f"Recommended methods: {analysis.recommended_methods}")
        >>> 
        >>> # Compare noise models
        >>> comparison = analyzer.compare_noise_models(analysis1, analysis2)
        >>> print(f"Noise increase: {comparison['error_rate_ratio']:.2f}x")
    """
    
    def __init__(self):
        # Thresholds for quality assessment
        self.error_rate_thresholds = {
            'excellent': 0.001,
            'good': 0.005,
            'fair': 0.01,
            'poor': 0.05
        }
        
        # Analysis history
        self._analysis_history: List[NoiseAnalysis] = []
    
    def analyze_noise_model(self, noise_model: Any, 
                           noise_model_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> NoiseAnalysis:
        """
        Analyze a noise model and extract comprehensive metrics.
        
        Args:
            noise_model: Noise model to analyze
            noise_model_id: Unique identifier for the noise model
            metadata: Additional metadata
        
        Returns:
            NoiseAnalysis object with detailed metrics
        """
        if noise_model_id is None:
            noise_model_id = f"noise_{int(time.time())}_{id(noise_model)}"
        
        analysis = NoiseAnalysis(
            noise_model_id=noise_model_id,
            timestamp=time.time(),
            num_qubits=self._get_num_qubits(noise_model),
            noise_type=self._get_noise_type(noise_model),
            metadata=metadata or {}
        )
        
        # Extract error rates
        analysis.single_qubit_error_rates = self._extract_single_qubit_errors(noise_model)
        analysis.two_qubit_error_rates = self._extract_two_qubit_errors(noise_model)
        analysis.readout_error_rates = self._extract_readout_errors(noise_model)
        
        # Calculate statistical properties
        analysis = self._calculate_error_statistics(analysis)
        
        # Extract coherence properties
        analysis.t1_times = self._extract_t1_times(noise_model)
        analysis.t2_times = self._extract_t2_times(noise_model)
        
        # Analyze correlations and structure
        analysis.error_correlations = self._analyze_error_correlations(noise_model)
        analysis.noise_structure = self._classify_noise_structure(analysis)
        
        # Calculate crosstalk
        analysis.crosstalk_matrix = self._extract_crosstalk_matrix(noise_model)
        analysis.crosstalk_strength = self._calculate_crosstalk_strength(analysis.crosstalk_matrix)
        
        # Quality metrics
        analysis.fidelity_estimates = self._estimate_fidelities(noise_model, analysis)
        analysis.diamond_norm = self._calculate_diamond_norm(noise_model)
        analysis.process_fidelity = self._calculate_process_fidelity(noise_model)
        
        # Mitigation recommendations
        analysis.mitigation_difficulty = self._assess_mitigation_difficulty(analysis)
        analysis.recommended_methods = self._recommend_mitigation_methods(analysis)
        
        # Store in history
        self._analysis_history.append(analysis)
        
        return analysis
    
    def _get_num_qubits(self, noise_model: Any) -> int:
        """Extract number of qubits from noise model."""
        if hasattr(noise_model, 'num_qubits'):
            return noise_model.num_qubits
        elif hasattr(noise_model, 'n_qubits'):
            return noise_model.n_qubits
        elif hasattr(noise_model, '_num_qubits'):
            return noise_model._num_qubits
        else:
            # Try to infer from error structure
            try:
                if hasattr(noise_model, 'errors'):
                    max_qubit = 0
                    for error in noise_model.errors:
                        if hasattr(error, 'qubits'):
                            max_qubit = max(max_qubit, max(error.qubits))
                    return max_qubit + 1
            except:
                pass
            
            logger.warning("Could not determine number of qubits in noise model")
            return 1
    
    def _get_noise_type(self, noise_model: Any) -> str:
        """Identify the type of noise model."""
        class_name = noise_model.__class__.__name__.lower()
        
        if 'depolarizing' in class_name:
            return 'depolarizing'
        elif 'amplitude' in class_name and 'damping' in class_name:
            return 'amplitude_damping'
        elif 'phase' in class_name and 'damping' in class_name:
            return 'phase_damping'
        elif 'composite' in class_name:
            return 'composite'
        elif 'coherent' in class_name:
            return 'coherent'
        elif 'readout' in class_name:
            return 'readout'
        elif 'crosstalk' in class_name:
            return 'crosstalk'
        else:
            return 'unknown'
    
    def _extract_single_qubit_errors(self, noise_model: Any) -> Dict[int, float]:
        """Extract single-qubit error rates."""
        error_rates = {}
        
        try:
            # Try different noise model interfaces
            if hasattr(noise_model, 'single_qubit_errors'):
                for qubit, error_rate in noise_model.single_qubit_errors.items():
                    error_rates[int(qubit)] = float(error_rate)
            elif hasattr(noise_model, 'depolarizing_1q'):
                for qubit, rate in noise_model.depolarizing_1q.items():
                    error_rates[int(qubit)] = float(rate)
            elif hasattr(noise_model, 'gate_errors'):
                # Extract from gate error data
                for gate_error in noise_model.gate_errors:
                    if hasattr(gate_error, 'qubits') and len(gate_error.qubits) == 1:
                        qubit = gate_error.qubits[0]
                        error_rate = getattr(gate_error, 'error_rate', getattr(gate_error, 'value', 0.0))
                        error_rates[int(qubit)] = float(error_rate)
            
            # If we have a known noise model type, try specific extraction
            noise_type = self._get_noise_type(noise_model)
            if noise_type == 'depolarizing' and hasattr(noise_model, 'p'):
                # Single parameter depolarizing model
                num_qubits = self._get_num_qubits(noise_model)
                for i in range(num_qubits):
                    error_rates[i] = float(noise_model.p)
            
        except Exception as e:
            logger.debug(f"Error extracting single-qubit errors: {e}")
        
        return error_rates
    
    def _extract_two_qubit_errors(self, noise_model: Any) -> Dict[Tuple[int, int], float]:
        """Extract two-qubit error rates."""
        error_rates = {}
        
        try:
            if hasattr(noise_model, 'two_qubit_errors'):
                for qubit_pair, error_rate in noise_model.two_qubit_errors.items():
                    if isinstance(qubit_pair, (list, tuple)) and len(qubit_pair) == 2:
                        key = tuple(sorted([int(qubit_pair[0]), int(qubit_pair[1])]))
                        error_rates[key] = float(error_rate)
            elif hasattr(noise_model, 'depolarizing_2q'):
                for qubit_pair, rate in noise_model.depolarizing_2q.items():
                    if isinstance(qubit_pair, (list, tuple)) and len(qubit_pair) == 2:
                        key = tuple(sorted([int(qubit_pair[0]), int(qubit_pair[1])]))
                        error_rates[key] = float(rate)
            elif hasattr(noise_model, 'gate_errors'):
                for gate_error in noise_model.gate_errors:
                    if hasattr(gate_error, 'qubits') and len(gate_error.qubits) == 2:
                        qubits = sorted([int(q) for q in gate_error.qubits])
                        key = tuple(qubits)
                        error_rate = getattr(gate_error, 'error_rate', getattr(gate_error, 'value', 0.0))
                        error_rates[key] = float(error_rate)
        
        except Exception as e:
            logger.debug(f"Error extracting two-qubit errors: {e}")
        
        return error_rates
    
    def _extract_readout_errors(self, noise_model: Any) -> Dict[int, float]:
        """Extract readout error rates."""
        error_rates = {}
        
        try:
            if hasattr(noise_model, 'readout_errors'):
                for qubit, error_rate in noise_model.readout_errors.items():
                    error_rates[int(qubit)] = float(error_rate)
            elif hasattr(noise_model, 'measurement_errors'):
                for qubit, error_rate in noise_model.measurement_errors.items():
                    error_rates[int(qubit)] = float(error_rate)
            elif hasattr(noise_model, 'readout_error'):
                # Single global readout error
                num_qubits = self._get_num_qubits(noise_model)
                for i in range(num_qubits):
                    error_rates[i] = float(noise_model.readout_error)
        
        except Exception as e:
            logger.debug(f"Error extracting readout errors: {e}")
        
        return error_rates
    
    def _calculate_error_statistics(self, analysis: NoiseAnalysis) -> NoiseAnalysis:
        """Calculate statistical properties of error rates."""
        all_errors = []
        
        # Collect all error rates
        all_errors.extend(analysis.single_qubit_error_rates.values())
        all_errors.extend(analysis.two_qubit_error_rates.values())
        all_errors.extend(analysis.readout_error_rates.values())
        
        if all_errors:
            analysis.max_error_rate = max(all_errors)
            analysis.min_error_rate = min(all_errors)
        
        # Calculate averages
        if analysis.single_qubit_error_rates:
            analysis.avg_single_qubit_error = np.mean(list(analysis.single_qubit_error_rates.values()))
        
        if analysis.two_qubit_error_rates:
            analysis.avg_two_qubit_error = np.mean(list(analysis.two_qubit_error_rates.values()))
        
        if analysis.readout_error_rates:
            analysis.avg_readout_error = np.mean(list(analysis.readout_error_rates.values()))
        
        return analysis
    
    def _extract_t1_times(self, noise_model: Any) -> Dict[int, float]:
        """Extract T1 relaxation times."""
        t1_times = {}
        
        try:
            if hasattr(noise_model, 't1_times'):
                for qubit, t1 in noise_model.t1_times.items():
                    t1_times[int(qubit)] = float(t1)
            elif hasattr(noise_model, 'relaxation_times'):
                for qubit, t1 in noise_model.relaxation_times.items():
                    t1_times[int(qubit)] = float(t1)
            elif hasattr(noise_model, 'coherence_times'):
                coherence_data = noise_model.coherence_times
                if isinstance(coherence_data, dict):
                    for qubit, times in coherence_data.items():
                        if isinstance(times, dict) and 't1' in times:
                            t1_times[int(qubit)] = float(times['t1'])
                        elif isinstance(times, (list, tuple)) and len(times) >= 1:
                            t1_times[int(qubit)] = float(times[0])  # Assume first is T1
        
        except Exception as e:
            logger.debug(f"Error extracting T1 times: {e}")
        
        return t1_times
    
    def _extract_t2_times(self, noise_model: Any) -> Dict[int, float]:
        """Extract T2 dephasing times."""
        t2_times = {}
        
        try:
            if hasattr(noise_model, 't2_times'):
                for qubit, t2 in noise_model.t2_times.items():
                    t2_times[int(qubit)] = float(t2)
            elif hasattr(noise_model, 'dephasing_times'):
                for qubit, t2 in noise_model.dephasing_times.items():
                    t2_times[int(qubit)] = float(t2)
            elif hasattr(noise_model, 'coherence_times'):
                coherence_data = noise_model.coherence_times
                if isinstance(coherence_data, dict):
                    for qubit, times in coherence_data.items():
                        if isinstance(times, dict) and 't2' in times:
                            t2_times[int(qubit)] = float(times['t2'])
                        elif isinstance(times, (list, tuple)) and len(times) >= 2:
                            t2_times[int(qubit)] = float(times[1])  # Assume second is T2
        
        except Exception as e:
            logger.debug(f"Error extracting T2 times: {e}")
        
        return t2_times
    
    def _analyze_error_correlations(self, noise_model: Any) -> Dict[Tuple[int, int], float]:
        """Analyze correlations between qubit errors."""
        correlations = {}
        
        try:
            if hasattr(noise_model, 'correlation_matrix'):
                matrix = noise_model.correlation_matrix
                if isinstance(matrix, np.ndarray):
                    for i in range(matrix.shape[0]):
                        for j in range(i+1, matrix.shape[1]):
                            correlations[(i, j)] = float(matrix[i, j])
            elif hasattr(noise_model, 'crosstalk_matrix'):
                # Use crosstalk as a proxy for correlation
                matrix = noise_model.crosstalk_matrix
                if isinstance(matrix, np.ndarray):
                    for i in range(matrix.shape[0]):
                        for j in range(i+1, matrix.shape[1]):
                            correlations[(i, j)] = float(matrix[i, j])
        
        except Exception as e:
            logger.debug(f"Error analyzing error correlations: {e}")
        
        return correlations
    
    def _classify_noise_structure(self, analysis: NoiseAnalysis) -> str:
        """Classify the structure of the noise model."""
        # Check for correlations
        if analysis.error_correlations:
            max_correlation = max(abs(corr) for corr in analysis.error_correlations.values())
            if max_correlation > 0.5:
                return "highly_correlated"
            elif max_correlation > 0.1:
                return "weakly_correlated"
        
        # Check for spatial structure
        if analysis.crosstalk_strength > 0.1:
            return "spatially_structured"
        
        # Check for uniform vs non-uniform errors
        single_errors = list(analysis.single_qubit_error_rates.values())
        two_qubit_errors = list(analysis.two_qubit_error_rates.values())
        
        if single_errors:
            single_std = np.std(single_errors) / np.mean(single_errors) if np.mean(single_errors) > 0 else 0
            if single_std < 0.1:  # Low relative standard deviation
                return "uniform"
            elif single_std > 0.5:
                return "highly_nonuniform"
        
        return "independent"
    
    def _extract_crosstalk_matrix(self, noise_model: Any) -> Optional[np.ndarray]:
        """Extract crosstalk matrix if available."""
        try:
            if hasattr(noise_model, 'crosstalk_matrix'):
                matrix = noise_model.crosstalk_matrix
                if isinstance(matrix, np.ndarray):
                    return matrix
                elif isinstance(matrix, (list, tuple)):
                    return np.array(matrix)
            elif hasattr(noise_model, 'coupling_matrix'):
                matrix = noise_model.coupling_matrix
                if isinstance(matrix, np.ndarray):
                    return matrix
        except Exception as e:
            logger.debug(f"Error extracting crosstalk matrix: {e}")
        
        return None
    
    def _calculate_crosstalk_strength(self, crosstalk_matrix: Optional[np.ndarray]) -> float:
        """Calculate overall crosstalk strength."""
        if crosstalk_matrix is None:
            return 0.0
        
        try:
            # Calculate off-diagonal elements (crosstalk)
            n = crosstalk_matrix.shape[0]
            off_diagonal = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        off_diagonal.append(abs(crosstalk_matrix[i, j]))
            
            return np.mean(off_diagonal) if off_diagonal else 0.0
        
        except Exception:
            return 0.0
    
    def _estimate_fidelities(self, noise_model: Any, analysis: NoiseAnalysis) -> Dict[str, float]:
        """Estimate various fidelity measures."""
        fidelities = {}
        
        try:
            # Gate fidelity estimates (1 - error_rate approximation)
            if analysis.avg_single_qubit_error > 0:
                fidelities['avg_single_qubit_fidelity'] = 1.0 - analysis.avg_single_qubit_error
            
            if analysis.avg_two_qubit_error > 0:
                fidelities['avg_two_qubit_fidelity'] = 1.0 - analysis.avg_two_qubit_error
            
            if analysis.avg_readout_error > 0:
                fidelities['avg_readout_fidelity'] = 1.0 - analysis.avg_readout_error
            
            # Overall state fidelity estimate
            overall_error = analysis.overall_error_rate
            if overall_error > 0:
                fidelities['estimated_state_fidelity'] = 1.0 - overall_error
        
        except Exception as e:
            logger.debug(f"Error estimating fidelities: {e}")
        
        return fidelities
    
    def _calculate_diamond_norm(self, noise_model: Any) -> Optional[float]:
        """Calculate diamond norm if possible."""
        # This is a placeholder - diamond norm calculation is complex
        # and would require quantum information theory implementations
        try:
            if hasattr(noise_model, 'diamond_norm'):
                return float(noise_model.diamond_norm)
        except Exception:
            pass
        
        return None
    
    def _calculate_process_fidelity(self, noise_model: Any) -> Optional[float]:
        """Calculate process fidelity if possible."""
        try:
            if hasattr(noise_model, 'process_fidelity'):
                return float(noise_model.process_fidelity)
            elif hasattr(noise_model, 'fidelity'):
                return float(noise_model.fidelity)
        except Exception:
            pass
        
        return None
    
    def _assess_mitigation_difficulty(self, analysis: NoiseAnalysis) -> float:
        """Assess how difficult this noise will be to mitigate (0=easy, 1=very hard)."""
        difficulty = 0.0
        
        # High error rates increase difficulty
        overall_error = analysis.overall_error_rate
        if overall_error > 0.1:
            difficulty += 0.4
        elif overall_error > 0.05:
            difficulty += 0.3
        elif overall_error > 0.01:
            difficulty += 0.2
        elif overall_error > 0.005:
            difficulty += 0.1
        
        # Correlated noise is harder to mitigate
        if analysis.noise_structure in ['highly_correlated', 'spatially_structured']:
            difficulty += 0.3
        elif analysis.noise_structure == 'weakly_correlated':
            difficulty += 0.15
        
        # High crosstalk increases difficulty
        if analysis.crosstalk_strength > 0.1:
            difficulty += 0.2
        elif analysis.crosstalk_strength > 0.05:
            difficulty += 0.1
        
        # Coherent errors are typically harder than incoherent
        if analysis.noise_type == 'coherent':
            difficulty += 0.1
        
        return min(1.0, difficulty)
    
    def _recommend_mitigation_methods(self, analysis: NoiseAnalysis) -> List[str]:
        """Recommend appropriate error mitigation methods."""
        recommendations = []
        
        # Based on error rates
        if analysis.overall_error_rate < 0.005:
            recommendations.append("zero_noise_extrapolation")
        
        if analysis.avg_two_qubit_error > 0.01:
            recommendations.append("virtual_distillation")
        
        # Based on noise structure
        if analysis.noise_structure == "independent":
            recommendations.extend(["zero_noise_extrapolation", "probabilistic_error_cancellation"])
        elif analysis.noise_structure in ["correlated", "spatially_structured"]:
            recommendations.append("clifford_data_regression")
        
        # Based on noise type
        if analysis.noise_type == "depolarizing":
            recommendations.extend(["zero_noise_extrapolation", "virtual_distillation"])
        elif analysis.noise_type == "coherent":
            recommendations.append("randomized_compiling")
        elif analysis.noise_type == "amplitude_damping":
            recommendations.append("error_suppression")
        
        # Based on readout errors
        if analysis.avg_readout_error > 0.02:
            recommendations.append("readout_error_mitigation")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for method in recommendations:
            if method not in seen:
                seen.add(method)
                unique_recommendations.append(method)
        
        return unique_recommendations[:5]  # Limit to top 5 recommendations
    
    def compare_noise_models(self, analysis1: NoiseAnalysis, 
                           analysis2: NoiseAnalysis) -> Dict[str, Any]:
        """
        Compare two noise model analyses.
        
        Args:
            analysis1: First noise analysis
            analysis2: Second noise analysis
        
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            'noise_model_ids': [analysis1.noise_model_id, analysis2.noise_model_id],
            'error_rate_comparison': {
                'overall_error_ratio': analysis2.overall_error_rate / analysis1.overall_error_rate if analysis1.overall_error_rate > 0 else float('inf'),
                'single_qubit_error_ratio': analysis2.avg_single_qubit_error / analysis1.avg_single_qubit_error if analysis1.avg_single_qubit_error > 0 else float('inf'),
                'two_qubit_error_ratio': analysis2.avg_two_qubit_error / analysis1.avg_two_qubit_error if analysis1.avg_two_qubit_error > 0 else float('inf'),
                'readout_error_ratio': analysis2.avg_readout_error / analysis1.avg_readout_error if analysis1.avg_readout_error > 0 else float('inf')
            },
            'coherence_comparison': {
                'coherence_quality_diff': analysis2.coherence_quality - analysis1.coherence_quality
            },
            'structure_comparison': {
                'crosstalk_strength_diff': analysis2.crosstalk_strength - analysis1.crosstalk_strength,
                'structure_change': f"{analysis1.noise_structure} -> {analysis2.noise_structure}"
            },
            'mitigation_comparison': {
                'difficulty_diff': analysis2.mitigation_difficulty - analysis1.mitigation_difficulty,
                'method_overlap': len(set(analysis1.recommended_methods) & set(analysis2.recommended_methods)),
                'unique_methods_1': list(set(analysis1.recommended_methods) - set(analysis2.recommended_methods)),
                'unique_methods_2': list(set(analysis2.recommended_methods) - set(analysis1.recommended_methods))
            }
        }
        
        return comparison
    
    def get_analysis_history(self, limit: Optional[int] = None) -> List[NoiseAnalysis]:
        """Get historical noise analyses."""
        if limit:
            return self._analysis_history[-limit:]
        return list(self._analysis_history)
    
    def clear_history(self):
        """Clear analysis history."""
        self._analysis_history.clear()
    
    def export_analysis(self, analysis: NoiseAnalysis, filepath: str):
        """Export noise analysis to a JSON file."""
        import json
        
        export_data = {
            'noise_model_id': analysis.noise_model_id,
            'timestamp': analysis.timestamp,
            'basic_properties': {
                'num_qubits': analysis.num_qubits,
                'noise_type': analysis.noise_type
            },
            'error_rates': {
                'single_qubit_errors': analysis.single_qubit_error_rates,
                'two_qubit_errors': {f"{k[0]}-{k[1]}": v for k, v in analysis.two_qubit_error_rates.items()},
                'readout_errors': analysis.readout_error_rates,
                'statistics': {
                    'avg_single_qubit_error': analysis.avg_single_qubit_error,
                    'avg_two_qubit_error': analysis.avg_two_qubit_error,
                    'avg_readout_error': analysis.avg_readout_error,
                    'overall_error_rate': analysis.overall_error_rate
                }
            },
            'coherence_properties': {
                't1_times': analysis.t1_times,
                't2_times': analysis.t2_times,
                'coherence_quality': analysis.coherence_quality
            },
            'structure_analysis': {
                'noise_structure': analysis.noise_structure,
                'crosstalk_strength': analysis.crosstalk_strength,
                'error_correlations': {f"{k[0]}-{k[1]}": v for k, v in analysis.error_correlations.items()}
            },
            'quality_metrics': {
                'fidelity_estimates': analysis.fidelity_estimates,
                'diamond_norm': analysis.diamond_norm,
                'process_fidelity': analysis.process_fidelity
            },
            'mitigation_assessment': {
                'mitigation_difficulty': analysis.mitigation_difficulty,
                'recommended_methods': analysis.recommended_methods
            },
            'metadata': analysis.metadata
        }
        
        # Handle numpy arrays in crosstalk matrix
        if analysis.crosstalk_matrix is not None:
            export_data['structure_analysis']['crosstalk_matrix'] = analysis.crosstalk_matrix.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported noise analysis to {filepath}")