"""Result classes for Zero-Noise Extrapolation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ZNEResult:
    """
    Results from Zero-Noise Extrapolation.
    
    Contains both raw and mitigated expectation values, along with
    detailed information about the extrapolation process.
    
    Attributes:
        raw_value: Expectation value at noise factor 1.0
        mitigated_value: Extrapolated expectation value at zero noise
        noise_factors: List of noise scaling factors used
        expectation_values: Measured expectation values at each noise factor
        extrapolation_data: Detailed information about the extrapolation fit
        error_reduction: Fractional error reduction (if ideal value known)
        config: Configuration used for ZNE
    """
    
    raw_value: float
    mitigated_value: float
    noise_factors: List[float]
    expectation_values: List[float]
    extrapolation_data: Dict[str, Any]
    error_reduction: Optional[float] = None
    config: Optional[Any] = None
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result data after initialization."""
        if len(self.noise_factors) != len(self.expectation_values):
            raise ValueError("noise_factors and expectation_values must have same length")
        
        if not all(factor >= 1.0 for factor in self.noise_factors):
            raise ValueError("All noise factors must be ≥ 1.0")
    
    @property
    def improvement_factor(self) -> Optional[float]:
        """
        Calculate improvement factor: |raw_error| / |mitigated_error|.
        
        Returns None if ideal value is not known.
        """
        if self.error_reduction is None:
            return None
        
        if self.error_reduction >= 1.0:
            return float('inf')  # Perfect mitigation
        
        return 1.0 / (1.0 - self.error_reduction) if self.error_reduction != 1.0 else float('inf')
    
    @property
    def absolute_error_reduction(self) -> Optional[float]:
        """
        Calculate absolute error reduction.
        
        Returns None if ideal value is not known.
        """
        ideal_value = self.extrapolation_data.get("ideal_value")
        if ideal_value is None:
            return None
        
        raw_error = abs(self.raw_value - ideal_value)
        mitigated_error = abs(self.mitigated_value - ideal_value)
        
        return raw_error - mitigated_error
    
    @property
    def extrapolation_method(self) -> str:
        """Get the extrapolation method used."""
        return self.extrapolation_data.get("method", "unknown")
    
    @property
    def fit_quality(self) -> float:
        """Get the R² value of the extrapolation fit."""
        return self.extrapolation_data.get("r_squared", 0.0)
    
    def plot_extrapolation(
        self,
        show_fit: bool = True,
        show_confidence: bool = False,
        extrapolation_range: Optional[tuple] = None,
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the extrapolation curve and data points.
        
        Args:
            show_fit: Whether to show the fitted curve
            show_confidence: Whether to show confidence intervals
            extrapolation_range: Range for plotting extrapolation (min_lambda, max_lambda)
            figsize: Figure size (width, height)
            save_path: Path to save the plot
            title: Custom plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot data points
        ax.scatter(
            self.noise_factors, 
            self.expectation_values,
            color='blue',
            s=100,
            alpha=0.7,
            label='Measured values',
            zorder=5
        )
        
        # Plot extrapolated point
        ax.scatter(
            [0], 
            [self.mitigated_value],
            color='red',
            s=150,
            marker='*',
            label=f'Extrapolated value: {self.mitigated_value:.4f}',
            zorder=5
        )
        
        # Plot raw value
        ax.scatter(
            [1.0],
            [self.raw_value],
            color='orange',
            s=120,
            marker='s',
            label=f'Raw value: {self.raw_value:.4f}',
            zorder=5
        )
        
        # Plot fitted curve if available
        if show_fit and "fitted_curve" in self.extrapolation_data:
            if extrapolation_range is None:
                x_min = 0
                x_max = max(self.noise_factors) * 1.1
            else:
                x_min, x_max = extrapolation_range
            
            x_fit = np.linspace(x_min, x_max, 200)
            try:
                y_fit = self.extrapolation_data["fitted_curve"](x_fit)
                ax.plot(
                    x_fit, 
                    y_fit,
                    color='green',
                    linewidth=2,
                    alpha=0.8,
                    label=f'{self.extrapolation_method} fit (R²={self.fit_quality:.3f})',
                    zorder=3
                )
            except Exception as e:
                print(f"Warning: Could not plot fitted curve: {e}")
        
        # Plot confidence intervals if available
        if show_confidence and "confidence_interval" in self.extrapolation_data:
            ci = self.extrapolation_data["confidence_interval"]
            if "lower" in ci and "upper" in ci:
                ax.axhspan(
                    ci["lower"],
                    ci["upper"],
                    alpha=0.2,
                    color='red',
                    label=f'95% confidence interval'
                )
        
        # Formatting
        ax.set_xlabel('Noise Factor (λ)')
        ax.set_ylabel('Expectation Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if title is None:
            title = f'Zero-Noise Extrapolation ({self.extrapolation_method})'
        ax.set_title(title)
        
        # Set axis limits
        ax.set_xlim(-0.1, max(self.noise_factors) * 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def summary(self) -> str:
        """Generate a summary string of the ZNE results."""
        lines = [
            "=== Zero-Noise Extrapolation Results ===",
            f"Raw value (λ=1):      {self.raw_value:.6f}",
            f"Mitigated value (λ=0): {self.mitigated_value:.6f}",
            f"Improvement:           {abs(self.mitigated_value - self.raw_value):.6f}",
            f"",
            f"Extrapolation method:  {self.extrapolation_method}",
            f"Fit quality (R²):      {self.fit_quality:.4f}",
            f"Noise factors used:    {len(self.noise_factors)} points",
            f"Factor range:          {min(self.noise_factors):.1f} - {max(self.noise_factors):.1f}",
        ]
        
        if self.error_reduction is not None:
            lines.extend([
                f"",
                f"Error reduction:       {self.error_reduction:.1%}",
                f"Improvement factor:    {self.improvement_factor:.2f}x"
            ])
        
        # Add method-specific information
        if "coefficients" in self.extrapolation_data:
            lines.extend([
                f"",
                f"Fit coefficients:",
            ])
            for key, value in self.extrapolation_data["coefficients"].items():
                lines.append(f"  {key}: {value:.6f}")
        
        return "\\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "raw_value": self.raw_value,
            "mitigated_value": self.mitigated_value,
            "noise_factors": self.noise_factors,
            "expectation_values": self.expectation_values,
            "extrapolation_data": self.extrapolation_data,
            "error_reduction": self.error_reduction,
            "improvement_factor": self.improvement_factor,
            "absolute_error_reduction": self.absolute_error_reduction,
            "extrapolation_method": self.extrapolation_method,
            "fit_quality": self.fit_quality
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZNEResult":
        """Create ZNEResult from dictionary."""
        # Extract required fields
        required_fields = [
            "raw_value", "mitigated_value", "noise_factors", 
            "expectation_values", "extrapolation_data"
        ]
        
        kwargs = {field: data[field] for field in required_fields}
        
        # Add optional fields
        if "error_reduction" in data:
            kwargs["error_reduction"] = data["error_reduction"]
        if "config" in data:
            kwargs["config"] = data["config"]
        
        return cls(**kwargs)
    
    def save(self, filepath: str) -> None:
        """Save result to file (JSON format)."""
        import json
        
        # Convert to serializable format
        data = self.to_dict()
        
        # Handle numpy arrays
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Recursively convert numpy objects
        def clean_data(d):
            if isinstance(d, dict):
                return {k: clean_data(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_data(v) for v in d]
            else:
                return convert_numpy(d)
        
        clean_data_dict = clean_data(data)
        
        with open(filepath, 'w') as f:
            json.dump(clean_data_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "ZNEResult":
        """Load result from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


@dataclass
class ZNEBatchResult:
    """
    Results from batch Zero-Noise Extrapolation.
    
    Used when running ZNE on multiple circuits or observables.
    """
    
    results: List[ZNEResult]
    circuit_names: Optional[List[str]] = None
    observable_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate batch result data."""
        if self.circuit_names and len(self.circuit_names) != len(self.results):
            raise ValueError("circuit_names length must match results length")
        
        if self.observable_names and len(self.observable_names) != len(self.results):
            raise ValueError("observable_names length must match results length")
    
    @property
    def mean_error_reduction(self) -> Optional[float]:
        """Calculate mean error reduction across all results."""
        reductions = [r.error_reduction for r in self.results if r.error_reduction is not None]
        if not reductions:
            return None
        return np.mean(reductions)
    
    @property
    def mean_improvement_factor(self) -> Optional[float]:
        """Calculate mean improvement factor across all results."""
        factors = [r.improvement_factor for r in self.results if r.improvement_factor is not None]
        if not factors:
            return None
        return np.mean(factors)
    
    def plot_summary(
        self,
        metric: str = "error_reduction",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot summary of batch results.
        
        Args:
            metric: Metric to plot ("error_reduction", "improvement_factor", "fit_quality")
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Extract metric values
        if metric == "error_reduction":
            values = [r.error_reduction for r in self.results if r.error_reduction is not None]
            ylabel = "Error Reduction"
        elif metric == "improvement_factor":
            values = [r.improvement_factor for r in self.results if r.improvement_factor is not None]
            ylabel = "Improvement Factor"
        elif metric == "fit_quality":
            values = [r.fit_quality for r in self.results]
            ylabel = "Fit Quality (R²)"
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if not values:
            print(f"No data available for metric: {metric}")
            return fig
        
        # Plot 1: Individual results
        x_labels = range(len(values))
        if self.circuit_names:
            x_labels = self.circuit_names[:len(values)]
        
        ax1.bar(range(len(values)), values, alpha=0.7)
        ax1.set_xlabel("Circuit")
        ax1.set_ylabel(ylabel)
        ax1.set_title(f"{ylabel} by Circuit")
        
        if isinstance(x_labels[0], str):
            ax1.set_xticks(range(len(values)))
            ax1.set_xticklabels(x_labels, rotation=45, ha='right')
        
        # Plot 2: Distribution
        ax2.hist(values, bins=min(10, len(values)), alpha=0.7, edgecolor='black')
        ax2.set_xlabel(ylabel)
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"{ylabel} Distribution")
        
        # Add mean line
        mean_val = np.mean(values)
        ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def summary(self) -> str:
        """Generate summary of batch results."""
        lines = [
            "=== Batch ZNE Results Summary ===",
            f"Total circuits:        {len(self.results)}",
            f"",
        ]
        
        # Error reduction statistics
        error_reductions = [r.error_reduction for r in self.results if r.error_reduction is not None]
        if error_reductions:
            lines.extend([
                f"Error Reduction:",
                f"  Mean:                {np.mean(error_reductions):.1%}",
                f"  Std:                 {np.std(error_reductions):.1%}",
                f"  Min:                 {np.min(error_reductions):.1%}",
                f"  Max:                 {np.max(error_reductions):.1%}",
                f"",
            ])
        
        # Improvement factor statistics
        improvement_factors = [r.improvement_factor for r in self.results if r.improvement_factor is not None]
        if improvement_factors:
            lines.extend([
                f"Improvement Factor:",
                f"  Mean:                {np.mean(improvement_factors):.2f}x",
                f"  Std:                 {np.std(improvement_factors):.2f}x",
                f"  Min:                 {np.min(improvement_factors):.2f}x",
                f"  Max:                 {np.max(improvement_factors):.2f}x",
                f"",
            ])
        
        # Fit quality statistics
        fit_qualities = [r.fit_quality for r in self.results]
        lines.extend([
            f"Fit Quality (R²):",
            f"  Mean:                {np.mean(fit_qualities):.3f}",
            f"  Std:                 {np.std(fit_qualities):.3f}",
            f"  Min:                 {np.min(fit_qualities):.3f}",
            f"  Max:                 {np.max(fit_qualities):.3f}",
        ])
        
        return "\\n".join(lines)