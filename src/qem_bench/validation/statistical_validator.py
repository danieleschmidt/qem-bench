"""
Statistical Validation Framework

Implements comprehensive statistical testing and validation methods for
quantum error mitigation research, ensuring rigorous hypothesis testing
and robust statistical inference.

Research Contributions:
- Comprehensive hypothesis testing suite for QEM claims
- Multiple testing correction with FDR control
- Bootstrap and permutation testing for non-parametric inference
- Effect size estimation with confidence intervals
- Power analysis for experimental design
- Bayesian statistical inference for uncertainty quantification
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from enum import Enum
import scipy.stats as stats
from scipy import special


class TestType(Enum):
    """Types of statistical tests"""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"
    BAYES_FACTOR = "bayes_factor"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    ANDERSON_DARLING = "anderson_darling"


class EffectSizeType(Enum):
    """Types of effect size measures"""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"
    CLIFF_DELTA = "cliff_delta"


@dataclass
class HypothesisTest:
    """Definition of a statistical hypothesis test"""
    
    null_hypothesis: str
    alternative_hypothesis: str
    test_type: TestType
    significance_level: float = 0.05
    one_tailed: bool = False
    effect_size_type: EffectSizeType = EffectSizeType.COHENS_D
    minimum_effect_size: float = 0.2  # Minimum meaningful effect
    
    def __post_init__(self):
        """Validate test parameters"""
        if not 0 < self.significance_level < 1:
            raise ValueError("Significance level must be between 0 and 1")
        
        if self.minimum_effect_size < 0:
            raise ValueError("Minimum effect size must be non-negative")


@dataclass
class EffectSize:
    """Effect size calculation result"""
    
    effect_size_type: EffectSizeType
    point_estimate: float
    confidence_interval: Tuple[float, float]
    interpretation: str  # "negligible", "small", "medium", "large"
    confidence_level: float = 0.95
    
    def is_meaningful(self, threshold: float = 0.2) -> bool:
        """Check if effect size is practically meaningful"""
        return abs(self.point_estimate) >= threshold


@dataclass
class TestResult:
    """Comprehensive statistical test result"""
    
    hypothesis_test: HypothesisTest
    test_statistic: float
    p_value: float
    adjusted_p_value: Optional[float] = None
    effect_size: Optional[EffectSize] = None
    power: Optional[float] = None
    
    # Additional statistics
    degrees_freedom: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Interpretation
    reject_null: bool = False
    statistical_significance: str = "not_significant"
    practical_significance: str = "unknown"
    
    # Sample information
    sample_sizes: Tuple[int, ...] = field(default_factory=tuple)
    sample_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Meta-information
    test_timestamp: datetime = field(default_factory=datetime.now)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Determine significance based on p-value"""
        self.reject_null = (self.adjusted_p_value or self.p_value) < self.hypothesis_test.significance_level
        
        if self.reject_null:
            self.statistical_significance = "significant"
        else:
            self.statistical_significance = "not_significant"
        
        # Determine practical significance
        if self.effect_size:
            if self.effect_size.is_meaningful():
                self.practical_significance = "meaningful"
            else:
                self.practical_significance = "negligible"


@dataclass
class PowerAnalysis:
    """Statistical power analysis results"""
    
    test_type: TestType
    effect_size: float
    sample_size: int
    significance_level: float
    power: float
    
    # Sample size calculations
    required_sample_size_80_power: int = 0
    required_sample_size_90_power: int = 0
    
    # Effect size detection
    detectable_effect_80_power: float = 0.0
    detectable_effect_90_power: float = 0.0


class StatisticalValidator:
    """
    Comprehensive statistical validation framework for QEM research
    
    This class implements rigorous statistical methods for validating
    quantum error mitigation claims:
    
    1. Hypothesis testing with multiple correction methods
    2. Effect size estimation with confidence intervals  
    3. Bootstrap and permutation testing for robust inference
    4. Power analysis for experimental design
    5. Bayesian inference for uncertainty quantification
    6. Cross-validation for machine learning components
    
    The framework ensures that research claims are statistically sound
    and reproducible according to best practices in scientific computing.
    """
    
    def __init__(
        self,
        default_significance_level: float = 0.05,
        multiple_testing_correction: str = "benjamini_hochberg",
        bootstrap_samples: int = 10000,
        permutation_samples: int = 10000,
        random_seed: Optional[int] = None
    ):
        self.default_significance_level = default_significance_level
        self.multiple_testing_correction = multiple_testing_correction
        self.bootstrap_samples = bootstrap_samples
        self.permutation_samples = permutation_samples
        
        # Set random seed for reproducibility
        self.random_seed = random_seed or 42
        np.random.seed(self.random_seed)
        
        # JAX compiled functions for performance
        self._bootstrap_statistic = jax.jit(self._compute_bootstrap_statistic)
        self._permutation_statistic = jax.jit(self._compute_permutation_statistic)
        self._bayesian_t_test = jax.jit(self._compute_bayesian_t_test)
        
        # Test history for multiple testing correction
        self.test_history: List[TestResult] = []
        
        # Research tracking
        self._research_metrics = {
            "tests_performed": 0,
            "significant_results": 0,
            "effect_sizes_computed": 0,
            "power_analyses_performed": 0,
            "bootstrap_tests": 0,
            "permutation_tests": 0
        }
    
    def validate_improvement(
        self,
        baseline_results: List[float],
        improved_results: List[float],
        hypothesis_test: Optional[HypothesisTest] = None,
        compute_effect_size: bool = True,
        compute_power: bool = True
    ) -> TestResult:
        """
        Validate that improved method significantly outperforms baseline
        
        Args:
            baseline_results: Performance results from baseline method
            improved_results: Performance results from improved method  
            hypothesis_test: Specification of hypothesis test to perform
            compute_effect_size: Whether to compute effect size
            compute_power: Whether to compute statistical power
            
        Returns:
            TestResult with comprehensive validation statistics
        """
        
        if hypothesis_test is None:
            hypothesis_test = HypothesisTest(
                null_hypothesis="Improved method performance <= Baseline performance",
                alternative_hypothesis="Improved method performance > Baseline performance",
                test_type=TestType.T_TEST,
                one_tailed=True
            )
        
        # Validate input data
        self._validate_input_data(baseline_results, improved_results)
        
        # Perform statistical test
        test_result = self._perform_hypothesis_test(
            baseline_results, improved_results, hypothesis_test
        )
        
        # Compute effect size if requested
        if compute_effect_size:
            test_result.effect_size = self._compute_effect_size(
                baseline_results, improved_results, hypothesis_test.effect_size_type
            )
            self._research_metrics["effect_sizes_computed"] += 1
        
        # Compute power if requested
        if compute_power and test_result.effect_size:
            test_result.power = self._compute_statistical_power(
                test_result.effect_size.point_estimate,
                len(baseline_results) + len(improved_results),
                hypothesis_test
            )
            self._research_metrics["power_analyses_performed"] += 1
        
        # Store result for multiple testing correction
        self.test_history.append(test_result)
        self._research_metrics["tests_performed"] += 1
        
        if test_result.reject_null:
            self._research_metrics["significant_results"] += 1
        
        return test_result
    
    def _validate_input_data(self, baseline_results: List[float], improved_results: List[float]):
        """Validate input data quality and assumptions"""
        
        if len(baseline_results) == 0 or len(improved_results) == 0:
            raise ValueError("Both baseline and improved results must be non-empty")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(baseline_results)) or np.any(np.isnan(improved_results)):
            raise ValueError("Results contain NaN values")
        
        if np.any(np.isinf(baseline_results)) or np.any(np.isinf(improved_results)):
            raise ValueError("Results contain infinite values")
        
        # Check sample size adequacy
        min_sample_size = 3
        if len(baseline_results) < min_sample_size or len(improved_results) < min_sample_size:
            warnings.warn(f"Small sample sizes may affect test validity (minimum recommended: {min_sample_size})")
        
        # Check for identical values (potential data issues)
        if len(set(baseline_results)) == 1:
            warnings.warn("Baseline results are identical - may indicate data collection issues")
        
        if len(set(improved_results)) == 1:
            warnings.warn("Improved results are identical - may indicate data collection issues")
    
    def _perform_hypothesis_test(
        self,
        baseline_results: List[float],
        improved_results: List[float], 
        hypothesis_test: HypothesisTest
    ) -> TestResult:
        """Perform the specified hypothesis test"""
        
        baseline_array = np.array(baseline_results)
        improved_array = np.array(improved_results)
        
        if hypothesis_test.test_type == TestType.T_TEST:
            return self._perform_t_test(baseline_array, improved_array, hypothesis_test)
        elif hypothesis_test.test_type == TestType.MANN_WHITNEY:
            return self._perform_mann_whitney_test(baseline_array, improved_array, hypothesis_test)
        elif hypothesis_test.test_type == TestType.WILCOXON:
            return self._perform_wilcoxon_test(baseline_array, improved_array, hypothesis_test)
        elif hypothesis_test.test_type == TestType.BOOTSTRAP:
            return self._perform_bootstrap_test(baseline_array, improved_array, hypothesis_test)
        elif hypothesis_test.test_type == TestType.PERMUTATION:
            return self._perform_permutation_test(baseline_array, improved_array, hypothesis_test)
        elif hypothesis_test.test_type == TestType.BAYES_FACTOR:
            return self._perform_bayesian_test(baseline_array, improved_array, hypothesis_test)
        else:
            raise ValueError(f"Unsupported test type: {hypothesis_test.test_type}")
    
    def _perform_t_test(
        self,
        baseline: np.ndarray,
        improved: np.ndarray,
        hypothesis_test: HypothesisTest
    ) -> TestResult:
        """Perform independent samples t-test"""
        
        # Check normality assumption
        baseline_normal = self._test_normality(baseline)
        improved_normal = self._test_normality(improved)
        
        warnings_list = []
        if not baseline_normal or not improved_normal:
            warnings_list.append("Data may not be normally distributed - consider non-parametric test")
        
        # Perform Welch's t-test (unequal variances)
        if hypothesis_test.one_tailed:
            alternative = 'greater' if np.mean(improved) > np.mean(baseline) else 'less'
        else:
            alternative = 'two-sided'
        
        t_stat, p_value = stats.ttest_ind(improved, baseline, equal_var=False, alternative=alternative)
        
        # Calculate degrees of freedom for Welch's t-test
        n1, n2 = len(improved), len(baseline)
        s1, s2 = np.var(improved, ddof=1), np.var(baseline, ddof=1)
        
        df = ((s1/n1 + s2/n2)**2) / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        
        # Confidence interval for mean difference
        mean_diff = np.mean(improved) - np.mean(baseline)
        se_diff = np.sqrt(s1/n1 + s2/n2)
        
        t_critical = stats.t.ppf(1 - hypothesis_test.significance_level/2, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        return TestResult(
            hypothesis_test=hypothesis_test,
            test_statistic=t_stat,
            p_value=p_value,
            degrees_freedom=df,
            confidence_interval=(ci_lower, ci_upper),
            sample_sizes=(len(baseline), len(improved)),
            sample_statistics={
                "baseline_mean": np.mean(baseline),
                "baseline_std": np.std(baseline, ddof=1),
                "improved_mean": np.mean(improved),
                "improved_std": np.std(improved, ddof=1),
                "mean_difference": mean_diff
            },
            warnings=warnings_list
        )
    
    def _perform_mann_whitney_test(
        self,
        baseline: np.ndarray,
        improved: np.ndarray,
        hypothesis_test: HypothesisTest
    ) -> TestResult:
        """Perform Mann-Whitney U test (non-parametric)"""
        
        if hypothesis_test.one_tailed:
            alternative = 'greater'
        else:
            alternative = 'two-sided'
        
        u_stat, p_value = stats.mannwhitneyu(
            improved, baseline, alternative=alternative
        )
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(improved), len(baseline)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)  # Rank-biserial correlation
        
        return TestResult(
            hypothesis_test=hypothesis_test,
            test_statistic=u_stat,
            p_value=p_value,
            sample_sizes=(len(baseline), len(improved)),
            sample_statistics={
                "baseline_median": np.median(baseline),
                "improved_median": np.median(improved),
                "rank_biserial_correlation": r_rb
            }
        )
    
    def _perform_wilcoxon_test(
        self,
        baseline: np.ndarray,
        improved: np.ndarray,
        hypothesis_test: HypothesisTest
    ) -> TestResult:
        """Perform Wilcoxon signed-rank test (paired samples)"""
        
        if len(baseline) != len(improved):
            raise ValueError("Wilcoxon test requires paired samples of equal length")
        
        differences = improved - baseline
        
        if hypothesis_test.one_tailed:
            alternative = 'greater' if np.mean(differences) > 0 else 'less'
        else:
            alternative = 'two-sided'
        
        w_stat, p_value = stats.wilcoxon(differences, alternative=alternative)
        
        return TestResult(
            hypothesis_test=hypothesis_test,
            test_statistic=w_stat,
            p_value=p_value,
            sample_sizes=(len(baseline), len(improved)),
            sample_statistics={
                "median_difference": np.median(differences),
                "mean_difference": np.mean(differences)
            }
        )
    
    def _perform_bootstrap_test(
        self,
        baseline: np.ndarray,
        improved: np.ndarray,
        hypothesis_test: HypothesisTest
    ) -> TestResult:
        """Perform bootstrap hypothesis test"""
        
        self._research_metrics["bootstrap_tests"] += 1
        
        # Observed test statistic (difference in means)
        observed_diff = np.mean(improved) - np.mean(baseline)
        
        # Bootstrap under null hypothesis (no difference)
        combined_data = np.concatenate([baseline, improved])
        n_baseline = len(baseline)
        
        bootstrap_diffs = []
        
        for _ in range(self.bootstrap_samples):
            # Resample combined data
            resampled = np.random.choice(combined_data, size=len(combined_data), replace=True)
            
            # Split into two groups of original sizes
            boot_baseline = resampled[:n_baseline]
            boot_improved = resampled[n_baseline:]
            
            # Calculate bootstrap statistic
            boot_diff = np.mean(boot_improved) - np.mean(boot_baseline)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value
        if hypothesis_test.one_tailed:
            p_value = np.mean(bootstrap_diffs >= observed_diff)
        else:
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # Bootstrap confidence interval for the difference
        alpha = hypothesis_test.significance_level
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        return TestResult(
            hypothesis_test=hypothesis_test,
            test_statistic=observed_diff,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            sample_sizes=(len(baseline), len(improved)),
            sample_statistics={
                "observed_difference": observed_diff,
                "bootstrap_samples": self.bootstrap_samples,
                "bootstrap_std": np.std(bootstrap_diffs)
            }
        )
    
    def _perform_permutation_test(
        self,
        baseline: np.ndarray,
        improved: np.ndarray,
        hypothesis_test: HypothesisTest
    ) -> TestResult:
        """Perform permutation test"""
        
        self._research_metrics["permutation_tests"] += 1
        
        # Observed test statistic
        observed_diff = np.mean(improved) - np.mean(baseline)
        
        # Combined data
        combined_data = np.concatenate([baseline, improved])
        n_baseline = len(baseline)
        n_total = len(combined_data)
        
        # Permutation distribution
        perm_diffs = []
        
        for _ in range(self.permutation_samples):
            # Random permutation
            permuted_indices = np.random.permutation(n_total)
            perm_baseline = combined_data[permuted_indices[:n_baseline]]
            perm_improved = combined_data[permuted_indices[n_baseline:]]
            
            perm_diff = np.mean(perm_improved) - np.mean(perm_baseline)
            perm_diffs.append(perm_diff)
        
        perm_diffs = np.array(perm_diffs)
        
        # Calculate p-value
        if hypothesis_test.one_tailed:
            p_value = np.mean(perm_diffs >= observed_diff)
        else:
            p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        return TestResult(
            hypothesis_test=hypothesis_test,
            test_statistic=observed_diff,
            p_value=p_value,
            sample_sizes=(len(baseline), len(improved)),
            sample_statistics={
                "observed_difference": observed_diff,
                "permutation_samples": self.permutation_samples,
                "permutation_std": np.std(perm_diffs)
            }
        )
    
    def _perform_bayesian_test(
        self,
        baseline: np.ndarray,
        improved: np.ndarray,
        hypothesis_test: HypothesisTest
    ) -> TestResult:
        """Perform Bayesian t-test with Bayes Factor"""
        
        # Calculate Bayes Factor using JAX for efficiency
        bayes_factor = self._compute_bayes_factor(baseline, improved)
        
        # Convert to p-value equivalent (approximate)
        # BF > 3 is often considered moderate evidence
        if bayes_factor > 10:
            p_value_equiv = 0.01
        elif bayes_factor > 3:
            p_value_equiv = 0.05
        elif bayes_factor > 1:
            p_value_equiv = 0.1
        else:
            p_value_equiv = 0.5
        
        return TestResult(
            hypothesis_test=hypothesis_test,
            test_statistic=bayes_factor,
            p_value=p_value_equiv,
            sample_sizes=(len(baseline), len(improved)),
            sample_statistics={
                "bayes_factor": bayes_factor,
                "evidence_strength": self._interpret_bayes_factor(bayes_factor)
            }
        )
    
    @jax.jit
    def _compute_bootstrap_statistic(
        self, 
        data1: jnp.ndarray, 
        data2: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> float:
        """Compute bootstrap statistic using JAX"""
        
        n1, n2 = len(data1), len(data2)
        combined = jnp.concatenate([data1, data2])
        
        # Bootstrap resample
        indices = jax.random.choice(key, len(combined), shape=(len(combined),), replace=True)
        resampled = combined[indices]
        
        boot_sample1 = resampled[:n1]
        boot_sample2 = resampled[n1:]
        
        return jnp.mean(boot_sample2) - jnp.mean(boot_sample1)
    
    @jax.jit
    def _compute_permutation_statistic(
        self,
        data1: jnp.ndarray,
        data2: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> float:
        """Compute permutation statistic using JAX"""
        
        combined = jnp.concatenate([data1, data2])
        indices = jax.random.permutation(key, len(combined))
        
        perm_sample1 = combined[indices[:len(data1)]]
        perm_sample2 = combined[indices[len(data1):]]
        
        return jnp.mean(perm_sample2) - jnp.mean(perm_sample1)
    
    @jax.jit
    def _compute_bayesian_t_test(
        self,
        data1: jnp.ndarray,
        data2: jnp.ndarray
    ) -> float:
        """Compute Bayesian t-test statistic using JAX"""
        
        # Simplified Bayesian t-test implementation
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = jnp.mean(data1), jnp.mean(data2)
        var1, var2 = jnp.var(data1, ddof=1), jnp.var(data2, ddof=1)
        
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        
        # Standard error
        se = jnp.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # t-statistic
        t = (mean2 - mean1) / se
        
        return t
    
    def _compute_bayes_factor(
        self, 
        baseline: np.ndarray, 
        improved: np.ndarray
    ) -> float:
        """Compute Bayes Factor for two-sample comparison"""
        
        # Simplified Bayes Factor calculation
        # In practice, would use more sophisticated methods
        
        n1, n2 = len(baseline), len(improved)
        mean1, mean2 = np.mean(baseline), np.mean(improved)
        var1, var2 = np.var(baseline, ddof=1), np.var(improved, ddof=1)
        
        # Effect size
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        effect_size = (mean2 - mean1) / pooled_std
        
        # Approximate Bayes Factor based on effect size and sample size
        # This is a simplified approximation
        bf_approx = np.exp(0.5 * effect_size**2 * (n1 * n2) / (n1 + n2))
        
        return float(bf_approx)
    
    def _interpret_bayes_factor(self, bf: float) -> str:
        """Interpret Bayes Factor strength of evidence"""
        
        if bf > 100:
            return "extreme_evidence"
        elif bf > 30:
            return "very_strong_evidence"
        elif bf > 10:
            return "strong_evidence"
        elif bf > 3:
            return "moderate_evidence"
        elif bf > 1:
            return "weak_evidence"
        else:
            return "evidence_against"
    
    def _test_normality(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """Test data for normality using Shapiro-Wilk test"""
        
        if len(data) < 3:
            return True  # Assume normal for very small samples
        
        if len(data) > 5000:
            # Use Anderson-Darling for large samples
            result = stats.anderson(data, dist='norm')
            return result.statistic < result.critical_values[2]  # 5% significance level
        else:
            # Use Shapiro-Wilk for smaller samples
            _, p_value = stats.shapiro(data)
            return p_value > alpha
    
    def _compute_effect_size(
        self,
        baseline: List[float],
        improved: List[float],
        effect_size_type: EffectSizeType
    ) -> EffectSize:
        """Compute effect size with confidence interval"""
        
        baseline_array = np.array(baseline)
        improved_array = np.array(improved)
        
        if effect_size_type == EffectSizeType.COHENS_D:
            effect_size_value, ci = self._compute_cohens_d(baseline_array, improved_array)
        elif effect_size_type == EffectSizeType.HEDGES_G:
            effect_size_value, ci = self._compute_hedges_g(baseline_array, improved_array)
        elif effect_size_type == EffectSizeType.GLASS_DELTA:
            effect_size_value, ci = self._compute_glass_delta(baseline_array, improved_array)
        elif effect_size_type == EffectSizeType.CLIFF_DELTA:
            effect_size_value, ci = self._compute_cliff_delta(baseline_array, improved_array)
        else:
            raise ValueError(f"Unsupported effect size type: {effect_size_type}")
        
        # Interpret effect size
        interpretation = self._interpret_effect_size(effect_size_value, effect_size_type)
        
        return EffectSize(
            effect_size_type=effect_size_type,
            point_estimate=effect_size_value,
            confidence_interval=ci,
            interpretation=interpretation
        )
    
    def _compute_cohens_d(self, baseline: np.ndarray, improved: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Compute Cohen's d with confidence interval"""
        
        n1, n2 = len(baseline), len(improved)
        mean1, mean2 = np.mean(baseline), np.mean(improved)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*np.var(baseline, ddof=1) + (n2-1)*np.var(improved, ddof=1)) / (n1+n2-2))
        
        # Cohen's d
        d = (mean2 - mean1) / pooled_std
        
        # Confidence interval using bootstrap
        bootstrap_ds = []
        for _ in range(1000):
            boot_baseline = np.random.choice(baseline, size=n1, replace=True)
            boot_improved = np.random.choice(improved, size=n2, replace=True)
            
            boot_pooled_std = np.sqrt(
                ((n1-1)*np.var(boot_baseline, ddof=1) + (n2-1)*np.var(boot_improved, ddof=1)) / (n1+n2-2)
            )
            boot_d = (np.mean(boot_improved) - np.mean(boot_baseline)) / boot_pooled_std
            bootstrap_ds.append(boot_d)
        
        ci_lower = np.percentile(bootstrap_ds, 2.5)
        ci_upper = np.percentile(bootstrap_ds, 97.5)
        
        return float(d), (ci_lower, ci_upper)
    
    def _compute_hedges_g(self, baseline: np.ndarray, improved: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Compute Hedges' g (bias-corrected Cohen's d)"""
        
        n1, n2 = len(baseline), len(improved)
        
        # First compute Cohen's d
        d, _ = self._compute_cohens_d(baseline, improved)
        
        # Bias correction factor
        df = n1 + n2 - 2
        correction_factor = 1 - (3 / (4 * df - 1))
        
        g = d * correction_factor
        
        # Approximate confidence interval (adjusted from Cohen's d CI)
        d_ci = self._compute_cohens_d(baseline, improved)[1]
        g_ci = (d_ci[0] * correction_factor, d_ci[1] * correction_factor)
        
        return float(g), g_ci
    
    def _compute_glass_delta(self, baseline: np.ndarray, improved: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Compute Glass's delta (standardized by control group SD)"""
        
        mean1, mean2 = np.mean(baseline), np.mean(improved)
        std1 = np.std(baseline, ddof=1)
        
        delta = (mean2 - mean1) / std1
        
        # Bootstrap confidence interval
        n1, n2 = len(baseline), len(improved)
        bootstrap_deltas = []
        
        for _ in range(1000):
            boot_baseline = np.random.choice(baseline, size=n1, replace=True)
            boot_improved = np.random.choice(improved, size=n2, replace=True)
            
            boot_delta = (np.mean(boot_improved) - np.mean(boot_baseline)) / np.std(boot_baseline, ddof=1)
            bootstrap_deltas.append(boot_delta)
        
        ci_lower = np.percentile(bootstrap_deltas, 2.5)
        ci_upper = np.percentile(bootstrap_deltas, 97.5)
        
        return float(delta), (ci_lower, ci_upper)
    
    def _compute_cliff_delta(self, baseline: np.ndarray, improved: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """Compute Cliff's delta (non-parametric effect size)"""
        
        n1, n2 = len(baseline), len(improved)
        
        # Count pairs where improved > baseline
        greater_count = 0
        less_count = 0
        
        for x in improved:
            for y in baseline:
                if x > y:
                    greater_count += 1
                elif x < y:
                    less_count += 1
        
        # Cliff's delta
        delta = (greater_count - less_count) / (n1 * n2)
        
        # Bootstrap confidence interval
        bootstrap_deltas = []
        
        for _ in range(1000):
            boot_baseline = np.random.choice(baseline, size=n1, replace=True)
            boot_improved = np.random.choice(improved, size=n2, replace=True)
            
            boot_greater = boot_less = 0
            for x in boot_improved:
                for y in boot_baseline:
                    if x > y:
                        boot_greater += 1
                    elif x < y:
                        boot_less += 1
            
            boot_delta = (boot_greater - boot_less) / (n1 * n2)
            bootstrap_deltas.append(boot_delta)
        
        ci_lower = np.percentile(bootstrap_deltas, 2.5)
        ci_upper = np.percentile(bootstrap_deltas, 97.5)
        
        return float(delta), (ci_lower, ci_upper)
    
    def _interpret_effect_size(self, effect_size: float, effect_type: EffectSizeType) -> str:
        """Interpret effect size magnitude"""
        
        abs_effect = abs(effect_size)
        
        if effect_type in [EffectSizeType.COHENS_D, EffectSizeType.HEDGES_G, EffectSizeType.GLASS_DELTA]:
            # Cohen's conventions
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        
        elif effect_type == EffectSizeType.CLIFF_DELTA:
            # Cliff's delta conventions
            if abs_effect < 0.147:
                return "negligible"
            elif abs_effect < 0.33:
                return "small"
            elif abs_effect < 0.474:
                return "medium"
            else:
                return "large"
        
        else:
            return "unknown"
    
    def _compute_statistical_power(
        self,
        effect_size: float,
        total_sample_size: int,
        hypothesis_test: HypothesisTest
    ) -> float:
        """Compute statistical power for given effect size and sample size"""
        
        # Simplified power calculation for t-test
        # In practice, would use more sophisticated methods
        
        alpha = hypothesis_test.significance_level
        
        # Approximate power calculation
        if hypothesis_test.test_type == TestType.T_TEST:
            # Non-centrality parameter
            ncp = effect_size * np.sqrt(total_sample_size / 4)  # Assumes equal group sizes
            
            # Critical t-value
            df = total_sample_size - 2
            if hypothesis_test.one_tailed:
                t_critical = stats.t.ppf(1 - alpha, df)
            else:
                t_critical = stats.t.ppf(1 - alpha/2, df)
            
            # Power calculation
            power = 1 - stats.t.cdf(t_critical - ncp, df)
            
            return min(1.0, max(0.0, power))
        
        else:
            # Default approximate power
            return 0.8 if abs(effect_size) > 0.5 else 0.5
    
    def apply_multiple_testing_correction(
        self, 
        method: str = None
    ) -> List[TestResult]:
        """Apply multiple testing correction to test history"""
        
        method = method or self.multiple_testing_correction
        
        if not self.test_history:
            return []
        
        p_values = [test.p_value for test in self.test_history]
        
        if method == "benjamini_hochberg":
            adjusted_p_values = self._benjamini_hochberg_correction(p_values)
        elif method == "bonferroni":
            adjusted_p_values = [min(1.0, p * len(p_values)) for p in p_values]
        elif method == "holm":
            adjusted_p_values = self._holm_correction(p_values)
        else:
            raise ValueError(f"Unsupported correction method: {method}")
        
        # Update test results with adjusted p-values
        for i, test_result in enumerate(self.test_history):
            test_result.adjusted_p_value = adjusted_p_values[i]
            # Re-evaluate significance
            test_result.reject_null = adjusted_p_values[i] < test_result.hypothesis_test.significance_level
            if test_result.reject_null:
                test_result.statistical_significance = "significant"
            else:
                test_result.statistical_significance = "not_significant"
        
        return self.test_history
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction"""
        
        n = len(p_values)
        indexed_p_values = [(p, i) for i, p in enumerate(p_values)]
        indexed_p_values.sort()
        
        adjusted_p_values = [0] * n
        
        for rank, (p_value, original_index) in enumerate(indexed_p_values):
            adjusted_p = min(1.0, p_value * n / (rank + 1))
            adjusted_p_values[original_index] = adjusted_p
        
        # Ensure monotonicity
        sorted_indices = [i for _, i in indexed_p_values]
        for i in range(n - 2, -1, -1):
            current_idx = sorted_indices[i]
            next_idx = sorted_indices[i + 1]
            adjusted_p_values[current_idx] = min(
                adjusted_p_values[current_idx],
                adjusted_p_values[next_idx]
            )
        
        return adjusted_p_values
    
    def _holm_correction(self, p_values: List[float]) -> List[float]:
        """Apply Holm step-down correction"""
        
        n = len(p_values)
        indexed_p_values = [(p, i) for i, p in enumerate(p_values)]
        indexed_p_values.sort()
        
        adjusted_p_values = [0] * n
        
        for rank, (p_value, original_index) in enumerate(indexed_p_values):
            adjusted_p = min(1.0, p_value * (n - rank))
            adjusted_p_values[original_index] = adjusted_p
        
        # Ensure monotonicity
        sorted_indices = [i for _, i in indexed_p_values]
        for i in range(1, n):
            current_idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            adjusted_p_values[current_idx] = max(
                adjusted_p_values[current_idx],
                adjusted_p_values[prev_idx]
            )
        
        return adjusted_p_values
    
    def perform_power_analysis(
        self,
        effect_sizes: List[float],
        sample_sizes: List[int],
        hypothesis_test: HypothesisTest
    ) -> PowerAnalysis:
        """Perform comprehensive power analysis"""
        
        # Calculate power for each combination
        powers = []
        for effect_size in effect_sizes:
            for sample_size in sample_sizes:
                power = self._compute_statistical_power(effect_size, sample_size, hypothesis_test)
                powers.append({
                    "effect_size": effect_size,
                    "sample_size": sample_size,
                    "power": power
                })
        
        # Find sample sizes needed for 80% and 90% power
        target_effect = effect_sizes[0] if effect_sizes else 0.5
        required_n_80 = self._find_required_sample_size(target_effect, 0.8, hypothesis_test)
        required_n_90 = self._find_required_sample_size(target_effect, 0.9, hypothesis_test)
        
        # Find detectable effect sizes
        target_n = sample_sizes[0] if sample_sizes else 50
        detectable_80 = self._find_detectable_effect_size(target_n, 0.8, hypothesis_test)
        detectable_90 = self._find_detectable_effect_size(target_n, 0.9, hypothesis_test)
        
        return PowerAnalysis(
            test_type=hypothesis_test.test_type,
            effect_size=target_effect,
            sample_size=target_n,
            significance_level=hypothesis_test.significance_level,
            power=powers[0]["power"] if powers else 0.5,
            required_sample_size_80_power=required_n_80,
            required_sample_size_90_power=required_n_90,
            detectable_effect_80_power=detectable_80,
            detectable_effect_90_power=detectable_90
        )
    
    def _find_required_sample_size(
        self,
        effect_size: float,
        target_power: float,
        hypothesis_test: HypothesisTest
    ) -> int:
        """Find required sample size for target power"""
        
        # Binary search for required sample size
        low, high = 4, 10000  # Minimum 4, maximum 10000
        
        while high - low > 1:
            mid = (low + high) // 2
            power = self._compute_statistical_power(effect_size, mid, hypothesis_test)
            
            if power < target_power:
                low = mid
            else:
                high = mid
        
        return high
    
    def _find_detectable_effect_size(
        self,
        sample_size: int,
        target_power: float,
        hypothesis_test: HypothesisTest
    ) -> float:
        """Find minimum detectable effect size for target power"""
        
        # Binary search for effect size
        low, high = 0.01, 3.0
        tolerance = 0.01
        
        while high - low > tolerance:
            mid = (low + high) / 2
            power = self._compute_statistical_power(mid, sample_size, hypothesis_test)
            
            if power < target_power:
                low = mid
            else:
                high = mid
        
        return high
    
    def generate_validation_report(
        self,
        test_results: List[TestResult] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        test_results = test_results or self.test_history
        
        if not test_results:
            return {"error": "No test results available"}
        
        # Summary statistics
        total_tests = len(test_results)
        significant_tests = sum(1 for test in test_results if test.reject_null)
        significant_rate = significant_tests / total_tests if total_tests > 0 else 0
        
        # Effect size analysis
        effect_sizes = [test.effect_size.point_estimate for test in test_results 
                       if test.effect_size is not None]
        
        effect_size_stats = {
            "count": len(effect_sizes),
            "mean": np.mean(effect_sizes) if effect_sizes else 0,
            "median": np.median(effect_sizes) if effect_sizes else 0,
            "std": np.std(effect_sizes) if effect_sizes else 0,
            "min": min(effect_sizes) if effect_sizes else 0,
            "max": max(effect_sizes) if effect_sizes else 0
        }
        
        # Power analysis
        powers = [test.power for test in test_results if test.power is not None]
        power_stats = {
            "count": len(powers),
            "mean": np.mean(powers) if powers else 0,
            "median": np.median(powers) if powers else 0,
            "below_80_percent": sum(1 for p in powers if p < 0.8) if powers else 0
        }
        
        # Test type distribution
        test_type_counts = {}
        for test in test_results:
            test_type = test.hypothesis_test.test_type.value
            test_type_counts[test_type] = test_type_counts.get(test_type, 0) + 1
        
        return {
            "summary": {
                "total_tests": total_tests,
                "significant_tests": significant_tests,
                "significance_rate": significant_rate,
                "multiple_testing_correction": self.multiple_testing_correction
            },
            "effect_sizes": effect_size_stats,
            "statistical_power": power_stats,
            "test_types": test_type_counts,
            "research_metrics": self._research_metrics,
            "recommendations": self._generate_recommendations(test_results)
        }
    
    def _generate_recommendations(self, test_results: List[TestResult]) -> List[str]:
        """Generate methodological recommendations based on results"""
        
        recommendations = []
        
        # Check sample sizes
        small_samples = sum(1 for test in test_results 
                           if any(n < 30 for n in test.sample_sizes))
        if small_samples > 0:
            recommendations.append(
                f"{small_samples} tests used small sample sizes (n<30). "
                "Consider larger samples or non-parametric tests."
            )
        
        # Check effect sizes
        effect_sizes = [test.effect_size.point_estimate for test in test_results 
                       if test.effect_size is not None]
        small_effects = sum(1 for es in effect_sizes if abs(es) < 0.2)
        if small_effects > len(effect_sizes) * 0.5 and effect_sizes:
            recommendations.append(
                "Many effect sizes are small (<0.2). Consider practical significance."
            )
        
        # Check multiple testing
        if len(test_results) > 1:
            unadjusted_significant = sum(1 for test in test_results if test.p_value < 0.05)
            adjusted_significant = sum(1 for test in test_results 
                                     if test.adjusted_p_value and test.adjusted_p_value < 0.05)
            if unadjusted_significant != adjusted_significant:
                recommendations.append(
                    f"Multiple testing correction changed {unadjusted_significant - adjusted_significant} "
                    "significant results. Report adjusted p-values."
                )
        
        # Check power
        powers = [test.power for test in test_results if test.power is not None]
        low_power = sum(1 for p in powers if p < 0.8)
        if low_power > len(powers) * 0.3 and powers:
            recommendations.append(
                f"{low_power} tests have low power (<80%). Consider larger samples."
            )
        
        return recommendations
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics for research tracking"""
        
        return {
            "framework_metrics": self._research_metrics,
            "test_history_size": len(self.test_history),
            "correction_method": self.multiple_testing_correction,
            "bootstrap_samples": self.bootstrap_samples,
            "permutation_samples": self.permutation_samples,
            "default_alpha": self.default_significance_level
        }