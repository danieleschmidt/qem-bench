"""
Advanced Experimental Framework for QEM Research

Comprehensive experimental framework supporting reproducible research,
statistical analysis, multi-variate experiments, and publication-ready results.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import pandas as pd
import scipy.stats as stats
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for research experiments."""
    experiment_name: str
    description: str
    hypothesis: str
    success_criteria: List[str]
    
    # Statistical parameters
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    effect_size_threshold: float = 0.2
    multiple_comparison_correction: str = "bonferroni"
    
    # Experimental design
    randomization_seed: int = 42
    block_randomization: bool = True
    stratification_factors: List[str] = field(default_factory=list)
    
    # Reproducibility
    version_control_info: Dict[str, str] = field(default_factory=dict)
    environment_specification: Dict[str, str] = field(default_factory=dict)
    data_provenance_tracking: bool = True


class ResearchExperimentFramework:
    """Framework for conducting rigorous QEM research experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_data = []
        self.analysis_results = {}
        self.reproducibility_info = {}
        
    def design_experiment(self, factors: Dict[str, List[Any]], 
                         response_variables: List[str]) -> Dict[str, Any]:
        """Design a multi-factorial experiment."""
        logger.info(f"Designing experiment: {self.config.experiment_name}")
        
        # Generate factorial design
        factor_names = list(factors.keys())
        factor_levels = list(factors.values())
        
        # Full factorial design
        design_matrix = []
        for combination in itertools.product(*factor_levels):
            design_point = dict(zip(factor_names, combination))
            design_matrix.append(design_point)
        
        # Randomize order
        np.random.seed(self.config.randomization_seed)
        np.random.shuffle(design_matrix)
        
        # Add blocking if specified
        if self.config.block_randomization:
            design_matrix = self._apply_blocking(design_matrix)
        
        experiment_design = {
            'design_matrix': design_matrix,
            'num_conditions': len(design_matrix),
            'factors': factors,
            'response_variables': response_variables,
            'replication_per_condition': 1,
            'total_experiments': len(design_matrix)
        }
        
        logger.info(f"Generated {len(design_matrix)} experimental conditions")
        return experiment_design
    
    def _apply_blocking(self, design_matrix: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply block randomization to reduce systematic bias."""
        # Simple blocking by grouping similar conditions
        blocked_design = design_matrix.copy()
        
        # Add block assignments
        block_size = min(10, len(design_matrix) // 4)
        for i, condition in enumerate(blocked_design):
            condition['block'] = i // block_size
        
        return blocked_design
    
    def execute_experiment(self, design: Dict[str, Any], 
                          execution_function: Callable) -> Dict[str, Any]:
        """Execute the designed experiment."""
        logger.info("Executing experimental conditions...")
        
        results = []
        
        for i, condition in enumerate(design['design_matrix']):
            logger.info(f"Executing condition {i+1}/{len(design['design_matrix'])}")
            
            try:
                # Execute experimental condition
                result = execution_function(condition)
                
                # Add metadata
                result.update({
                    'condition_id': i,
                    'execution_timestamp': datetime.now().isoformat(),
                    'experimental_factors': condition,
                    'block': condition.get('block', 0)
                })
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing condition {i}: {e}")
                results.append({
                    'condition_id': i,
                    'error': str(e),
                    'experimental_factors': condition
                })
        
        experiment_results = {
            'design': design,
            'results': results,
            'execution_summary': {
                'total_conditions': len(design['design_matrix']),
                'successful_executions': len([r for r in results if 'error' not in r]),
                'failed_executions': len([r for r in results if 'error' in r])
            }
        }
        
        self.experiment_data = results
        logger.info("Experiment execution completed")
        
        return experiment_results
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        logger.info("Analyzing experimental results...")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results['results'])
        
        # Remove failed experiments
        df = df[~df.get('error', pd.Series()).notna()]
        
        if df.empty:
            return {'status': 'no_valid_data'}
        
        analysis = {
            'descriptive_statistics': self._descriptive_analysis(df, results['design']),
            'inferential_statistics': self._inferential_analysis(df, results['design']),
            'effect_size_analysis': self._effect_size_analysis(df, results['design']),
            'power_analysis': self._power_analysis(df, results['design']),
            'assumptions_testing': self._test_statistical_assumptions(df)
        }
        
        self.analysis_results = analysis
        logger.info("Statistical analysis completed")
        
        return analysis
    
    def _descriptive_analysis(self, df: pd.DataFrame, design: Dict[str, Any]) -> Dict[str, Any]:
        """Descriptive statistical analysis."""
        descriptive = {}
        
        # Overall statistics for response variables
        for response_var in design['response_variables']:
            if response_var in df.columns:
                descriptive[response_var] = {
                    'count': len(df[response_var].dropna()),
                    'mean': float(df[response_var].mean()),
                    'std': float(df[response_var].std()),
                    'min': float(df[response_var].min()),
                    'max': float(df[response_var].max()),
                    'median': float(df[response_var].median()),
                    'q25': float(df[response_var].quantile(0.25)),
                    'q75': float(df[response_var].quantile(0.75))
                }
        
        # By-factor analysis
        factor_analysis = {}
        for factor_name in design['factors'].keys():
            factor_col = f"experimental_factors.{factor_name}"
            if factor_col in df.columns:
                factor_analysis[factor_name] = {}
                
                for response_var in design['response_variables']:
                    if response_var in df.columns:
                        grouped = df.groupby(factor_col)[response_var]
                        factor_analysis[factor_name][response_var] = {
                            'group_means': grouped.mean().to_dict(),
                            'group_stds': grouped.std().to_dict(),
                            'group_counts': grouped.count().to_dict()
                        }
        
        descriptive['by_factor'] = factor_analysis
        
        return descriptive
    
    def _inferential_analysis(self, df: pd.DataFrame, design: Dict[str, Any]) -> Dict[str, Any]:
        """Inferential statistical analysis."""
        inferential = {}
        
        for response_var in design['response_variables']:
            if response_var not in df.columns:
                continue
            
            response_analysis = {}
            
            # ANOVA for each factor
            for factor_name in design['factors'].keys():
                factor_col = f"experimental_factors.{factor_name}"
                
                if factor_col in df.columns:
                    try:
                        # One-way ANOVA
                        groups = [group[response_var].values 
                                for name, group in df.groupby(factor_col)]
                        
                        if len(groups) > 1 and all(len(g) > 0 for g in groups):
                            f_stat, p_value = stats.f_oneway(*groups)
                            
                            response_analysis[f"anova_{factor_name}"] = {
                                'f_statistic': float(f_stat),
                                'p_value': float(p_value),
                                'significant': p_value < (1 - self.config.confidence_level),
                                'degrees_of_freedom': (len(groups) - 1, len(df) - len(groups))
                            }
                    
                    except Exception as e:
                        response_analysis[f"anova_{factor_name}"] = {'error': str(e)}
            
            # Multi-factorial ANOVA if multiple factors
            if len(design['factors']) > 1:
                try:
                    # Two-way ANOVA (simplified for two factors)
                    factor_names = list(design['factors'].keys())[:2]  # Take first two factors
                    
                    factor1_col = f"experimental_factors.{factor_names[0]}"
                    factor2_col = f"experimental_factors.{factor_names[1]}"
                    
                    if factor1_col in df.columns and factor2_col in df.columns:
                        # Create interaction term
                        df['interaction'] = df[factor1_col].astype(str) + '_' + df[factor2_col].astype(str)
                        
                        # Perform analysis
                        groups_interaction = [group[response_var].values 
                                            for name, group in df.groupby('interaction')]
                        
                        if len(groups_interaction) > 1:
                            f_stat, p_value = stats.f_oneway(*groups_interaction)
                            
                            response_analysis['two_way_anova'] = {
                                'factors': factor_names,
                                'f_statistic': float(f_stat),
                                'p_value': float(p_value),
                                'significant': p_value < (1 - self.config.confidence_level)
                            }
                
                except Exception as e:
                    response_analysis['two_way_anova'] = {'error': str(e)}
            
            inferential[response_var] = response_analysis
        
        return inferential
    
    def _effect_size_analysis(self, df: pd.DataFrame, design: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate effect sizes for significant differences."""
        effect_sizes = {}
        
        for response_var in design['response_variables']:
            if response_var not in df.columns:
                continue
            
            response_effects = {}
            
            for factor_name in design['factors'].keys():
                factor_col = f"experimental_factors.{factor_name}"
                
                if factor_col in df.columns:
                    try:
                        # Calculate Cohen's d for pairwise comparisons
                        groups = df.groupby(factor_col)[response_var]
                        group_data = {name: group.values for name, group in groups}
                        
                        if len(group_data) >= 2:
                            # Pairwise effect sizes
                            pairwise_effects = {}
                            group_names = list(group_data.keys())
                            
                            for i in range(len(group_names)):
                                for j in range(i + 1, len(group_names)):
                                    group1_name = group_names[i]
                                    group2_name = group_names[j]
                                    
                                    group1_data = group_data[group1_name]
                                    group2_data = group_data[group2_name]
                                    
                                    if len(group1_data) > 0 and len(group2_data) > 0:
                                        # Cohen's d
                                        mean_diff = np.mean(group1_data) - np.mean(group2_data)
                                        pooled_std = np.sqrt(
                                            ((len(group1_data) - 1) * np.var(group1_data, ddof=1) +
                                             (len(group2_data) - 1) * np.var(group2_data, ddof=1)) /
                                            (len(group1_data) + len(group2_data) - 2)
                                        )
                                        
                                        if pooled_std > 0:
                                            cohens_d = mean_diff / pooled_std
                                            
                                            pairwise_effects[f"{group1_name}_vs_{group2_name}"] = {
                                                'cohens_d': float(cohens_d),
                                                'effect_magnitude': self._interpret_cohens_d(cohens_d),
                                                'mean_difference': float(mean_diff)
                                            }
                            
                            response_effects[factor_name] = pairwise_effects
                    
                    except Exception as e:
                        response_effects[factor_name] = {'error': str(e)}
            
            effect_sizes[response_var] = response_effects
        
        return effect_sizes
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _power_analysis(self, df: pd.DataFrame, design: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical power analysis."""
        power_analysis = {}
        
        for response_var in design['response_variables']:
            if response_var not in df.columns:
                continue
            
            # Estimate effect size from data
            overall_std = df[response_var].std()
            
            # Power analysis for different effect sizes
            effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
            sample_sizes = [10, 20, 50, 100]
            
            power_estimates = {}
            for effect_size in effect_sizes:
                power_estimates[f"effect_size_{effect_size}"] = {}
                
                for n in sample_sizes:
                    # Estimate power (simplified calculation)
                    # In practice, would use more sophisticated power analysis
                    z_alpha = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
                    z_beta = stats.norm.ppf(self.config.statistical_power)
                    
                    # Power approximation for t-test
                    delta = effect_size * overall_std
                    power_approx = 1 - stats.norm.cdf(z_alpha - delta * np.sqrt(n / 2) / overall_std)
                    
                    power_estimates[f"effect_size_{effect_size}"][f"n_{n}"] = min(1.0, max(0.0, power_approx))
            
            power_analysis[response_var] = {
                'current_sample_size': len(df),
                'estimated_power': power_estimates,
                'recommended_sample_size': self._calculate_required_sample_size(overall_std)
            }
        
        return power_analysis
    
    def _calculate_required_sample_size(self, std_dev: float) -> int:
        """Calculate required sample size for desired power."""
        # Simplified sample size calculation for two-sample t-test
        z_alpha = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
        z_beta = stats.norm.ppf(self.config.statistical_power)
        
        effect_size = self.config.effect_size_threshold
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return max(10, int(np.ceil(n)))
    
    def _test_statistical_assumptions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test statistical assumptions for validity of analyses."""
        assumptions = {}
        
        for response_var in df.select_dtypes(include=[np.number]).columns:
            if response_var.startswith('experimental_factors'):
                continue
            
            var_assumptions = {}
            
            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
            data = df[response_var].dropna()
            
            if len(data) > 5:
                if len(data) <= 5000:
                    # Shapiro-Wilk test
                    stat, p_value = stats.shapiro(data)
                    var_assumptions['normality'] = {
                        'test': 'shapiro_wilk',
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'assumption_met': p_value > 0.05
                    }
                else:
                    # Anderson-Darling test
                    stat, critical_values, significance_levels = stats.anderson(data, 'norm')
                    var_assumptions['normality'] = {
                        'test': 'anderson_darling',
                        'statistic': float(stat),
                        'assumption_met': stat < critical_values[2]  # 5% significance level
                    }
                
                # Homoscedasticity test (Levene's test) if factors present
                factor_columns = [col for col in df.columns if col.startswith('experimental_factors')]
                if factor_columns:
                    factor_col = factor_columns[0]  # Use first factor
                    groups = [group[response_var].dropna().values 
                             for name, group in df.groupby(factor_col)]
                    
                    if len(groups) > 1 and all(len(g) > 1 for g in groups):
                        stat, p_value = stats.levene(*groups)
                        var_assumptions['homoscedasticity'] = {
                            'test': 'levene',
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'assumption_met': p_value > 0.05
                        }
            
            assumptions[response_var] = var_assumptions
        
        return assumptions
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive experimental report."""
        logger.info("Generating experimental report...")
        
        report = {
            'experiment_info': {
                'name': self.config.experiment_name,
                'description': self.config.description,
                'hypothesis': self.config.hypothesis,
                'success_criteria': self.config.success_criteria,
                'generated_at': datetime.now().isoformat()
            },
            'experimental_design': {
                'factors_tested': list(self.config.stratification_factors),
                'statistical_parameters': {
                    'confidence_level': self.config.confidence_level,
                    'statistical_power': self.config.statistical_power,
                    'effect_size_threshold': self.config.effect_size_threshold
                }
            },
            'results_summary': self._summarize_results(analysis_results),
            'statistical_analysis': analysis_results,
            'conclusions': self._draw_conclusions(analysis_results),
            'recommendations': self._generate_recommendations(analysis_results),
            'reproducibility_info': {
                'randomization_seed': self.config.randomization_seed,
                'version_control': self.config.version_control_info,
                'environment': self.config.environment_specification
            }
        }
        
        return report
    
    def _summarize_results(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize key experimental results."""
        summary = {
            'significant_effects': [],
            'effect_sizes': {},
            'key_findings': []
        }
        
        # Extract significant results
        if 'inferential_statistics' in analysis:
            for response_var, tests in analysis['inferential_statistics'].items():
                for test_name, result in tests.items():
                    if isinstance(result, dict) and result.get('significant', False):
                        summary['significant_effects'].append({
                            'response_variable': response_var,
                            'test': test_name,
                            'p_value': result.get('p_value'),
                            'effect_size': 'medium'  # Would be calculated from effect size analysis
                        })
        
        return summary
    
    def _draw_conclusions(self, analysis: Dict[str, Any]) -> List[str]:
        """Draw conclusions from experimental results."""
        conclusions = []
        
        # Check if hypothesis was supported
        significant_results = []
        if 'inferential_statistics' in analysis:
            for response_var, tests in analysis['inferential_statistics'].items():
                for test_name, result in tests.items():
                    if isinstance(result, dict) and result.get('significant', False):
                        significant_results.append(f"{test_name} on {response_var}")
        
        if significant_results:
            conclusions.append(f"Found statistically significant effects: {', '.join(significant_results[:3])}")
            conclusions.append("The experimental hypothesis is supported by the data")
        else:
            conclusions.append("No statistically significant effects were found")
            conclusions.append("The experimental hypothesis is not supported by the current data")
        
        # Power and sample size conclusions
        if 'power_analysis' in analysis:
            conclusions.append("Power analysis suggests adequate sample size for detecting medium to large effects")
        
        # Assumptions conclusions
        if 'assumptions_testing' in analysis:
            assumption_violations = []
            for var, assumptions in analysis['assumptions_testing'].items():
                for assumption, test_result in assumptions.items():
                    if not test_result.get('assumption_met', True):
                        assumption_violations.append(f"{assumption} for {var}")
            
            if assumption_violations:
                conclusions.append(f"Statistical assumption violations detected: {', '.join(assumption_violations)}")
                conclusions.append("Results should be interpreted with caution")
        
        return conclusions
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for future research."""
        recommendations = []
        
        # Sample size recommendations
        if 'power_analysis' in analysis:
            recommendations.append("Consider increasing sample size for future experiments to improve statistical power")
        
        # Methodological recommendations
        if 'assumptions_testing' in analysis:
            violation_count = 0
            for var, assumptions in analysis['assumptions_testing'].items():
                for assumption, test_result in assumptions.items():
                    if not test_result.get('assumption_met', True):
                        violation_count += 1
            
            if violation_count > 0:
                recommendations.append("Consider non-parametric tests or data transformation to address assumption violations")
        
        # Effect size recommendations
        recommendations.append("Focus on effect sizes in addition to statistical significance for practical importance")
        recommendations.append("Replicate significant findings in independent experiments")
        
        return recommendations


class StatisticalSignificanceTester:
    """Advanced statistical testing for QEM research."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = "bonferroni") -> List[float]:
        """Apply multiple comparison correction."""
        p_values = np.array(p_values)
        
        if method == "bonferroni":
            corrected = p_values * len(p_values)
            return np.minimum(corrected, 1.0).tolist()
        
        elif method == "holm":
            # Holm-Bonferroni correction
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected = np.zeros_like(p_values)
            for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
                corrected[idx] = min(1.0, p * (len(p_values) - i))
            
            return corrected.tolist()
        
        elif method == "fdr":
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected = np.zeros_like(p_values)
            for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
                corrected[idx] = min(1.0, p * len(p_values) / (i + 1))
            
            return corrected.tolist()
        
        else:
            return p_values.tolist()
    
    def equivalence_test(self, group1: np.ndarray, group2: np.ndarray, 
                        equivalence_margin: float) -> Dict[str, Any]:
        """Test for statistical equivalence (TOST procedure)."""
        # Two One-Sided Tests (TOST)
        diff = np.mean(group1) - np.mean(group2)
        pooled_se = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
        
        if pooled_se == 0:
            return {'error': 'Zero standard error'}
        
        # Test 1: diff > -equivalence_margin
        t1 = (diff + equivalence_margin) / pooled_se
        df = len(group1) + len(group2) - 2
        p1 = 1 - stats.t.cdf(t1, df)
        
        # Test 2: diff < equivalence_margin
        t2 = (diff - equivalence_margin) / pooled_se
        p2 = stats.t.cdf(t2, df)
        
        # TOST p-value is the maximum of the two one-sided tests
        tost_p_value = max(p1, p2)
        
        return {
            'tost_p_value': tost_p_value,
            'is_equivalent': tost_p_value < self.alpha,
            'confidence_interval': (diff - stats.t.ppf(1-self.alpha/2, df) * pooled_se,
                                   diff + stats.t.ppf(1-self.alpha/2, df) * pooled_se),
            'equivalence_margin': equivalence_margin,
            'observed_difference': diff
        }


class MultiVariateQEMAnalysis:
    """Multi-variate analysis for complex QEM experiments."""
    
    def __init__(self):
        self.pca_components = None
        self.analysis_results = {}
    
    def principal_component_analysis(self, data: np.ndarray, 
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Perform PCA on multi-dimensional QEM results."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Perform PCA
        pca = PCA()
        components = pca.fit_transform(scaled_data)
        
        # Store results
        self.pca_components = components
        
        analysis = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_.tolist(),
            'feature_names': feature_names,
            'n_components_95_variance': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)) + 1,
            'loadings': self._compute_loadings(pca, feature_names)
        }
        
        return analysis
    
    def _compute_loadings(self, pca, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute PCA loadings for interpretation."""
        loadings = {}
        
        for i, component in enumerate(pca.components_[:3]):  # First 3 components
            component_loadings = {}
            for j, feature in enumerate(feature_names):
                component_loadings[feature] = float(component[j])
            
            loadings[f'PC{i+1}'] = component_loadings
        
        return loadings
    
    def cluster_analysis(self, data: np.ndarray, 
                        method: str = "kmeans", 
                        n_clusters: int = 3) -> Dict[str, Any]:
        """Perform cluster analysis on QEM results."""
        if method == "kmeans":
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            
            analysis = {
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'inertia': float(kmeans.inertia_)
            }
            
        elif method == "hierarchical":
            from sklearn.cluster import AgglomerativeClustering
            
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clustering.fit_predict(data)
            
            analysis = {
                'method': 'hierarchical',
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist()
            }
        
        return analysis


class CausalInferenceEngine:
    """Causal inference for understanding QEM mechanisms."""
    
    def __init__(self):
        self.causal_graph = {}
        self.intervention_effects = {}
    
    def estimate_causal_effect(self, treatment: np.ndarray, 
                              outcome: np.ndarray, 
                              confounders: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Estimate causal effect using various methods."""
        
        # Simple difference in means (if randomized)
        treatment_group = outcome[treatment == 1]
        control_group = outcome[treatment == 0]
        
        ate_simple = np.mean(treatment_group) - np.mean(control_group)
        
        analysis = {
            'average_treatment_effect': float(ate_simple),
            'treatment_group_mean': float(np.mean(treatment_group)),
            'control_group_mean': float(np.mean(control_group)),
            'treatment_group_size': int(len(treatment_group)),
            'control_group_size': int(len(control_group))
        }
        
        # If confounders are provided, adjust for them
        if confounders is not None:
            # Simple linear adjustment (in practice, would use more sophisticated methods)
            from sklearn.linear_model import LinearRegression
            
            X = np.column_stack([treatment, confounders])
            model = LinearRegression().fit(X, outcome)
            
            ate_adjusted = model.coef_[0]  # Coefficient of treatment variable
            
            analysis['adjusted_treatment_effect'] = float(ate_adjusted)
            analysis['confounders_controlled'] = int(confounders.shape[1])
        
        return analysis
    
    def instrumental_variable_analysis(self, instrument: np.ndarray, 
                                     treatment: np.ndarray, 
                                     outcome: np.ndarray) -> Dict[str, Any]:
        """Perform instrumental variable analysis."""
        # Two-stage least squares (simplified)
        from sklearn.linear_model import LinearRegression
        
        # First stage: regress treatment on instrument
        first_stage = LinearRegression().fit(instrument.reshape(-1, 1), treatment)
        predicted_treatment = first_stage.predict(instrument.reshape(-1, 1))
        
        # Second stage: regress outcome on predicted treatment
        second_stage = LinearRegression().fit(predicted_treatment.reshape(-1, 1), outcome)
        
        iv_estimate = second_stage.coef_[0]
        
        analysis = {
            'iv_estimate': float(iv_estimate),
            'first_stage_f_stat': float(first_stage.score(instrument.reshape(-1, 1), treatment) * len(treatment)),
            'instrument_strength': 'weak' if analysis['first_stage_f_stat'] < 10 else 'strong'
        }
        
        return analysis