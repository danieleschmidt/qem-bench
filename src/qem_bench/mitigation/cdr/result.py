"""Result classes for Clifford Data Regression."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class CDRResult:
    """
    Results from Clifford Data Regression.
    
    Contains both raw and mitigated expectation values, along with
    detailed information about the regression process and model performance.
    
    Attributes:
        raw_value: Expectation value from noisy execution
        mitigated_value: Corrected expectation value from CDR
        predicted_ideal_value: Regression model prediction of ideal value
        correction: Applied correction (predicted_ideal - raw_value)
        confidence_interval: Confidence interval for mitigated value
        error_reduction: Fractional error reduction (if ideal value known)
        training_data_size: Number of training circuits used
        regression_method: Regression method used
        config: Configuration used for CDR
    """
    
    raw_value: float
    mitigated_value: float
    predicted_ideal_value: float
    correction: float
    confidence_interval: Optional[Dict[str, float]] = None
    error_reduction: Optional[float] = None
    training_data_size: int = 0
    regression_method: str = "ridge"
    config: Optional[Any] = None
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result data after initialization."""
        # Basic validation
        if not isinstance(self.raw_value, (int, float)):
            raise ValueError("raw_value must be a number")
        if not isinstance(self.mitigated_value, (int, float)):
            raise ValueError("mitigated_value must be a number")
        
        # Check correction consistency
        expected_correction = self.predicted_ideal_value - self.raw_value
        if abs(self.correction - expected_correction) > 1e-10:
            self.correction = expected_correction  # Fix minor numerical errors
    
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
        
        if self.error_reduction <= 0.0:
            return 0.0  # No improvement or made worse
        
        return 1.0 / (1.0 - self.error_reduction)
    
    @property
    def absolute_correction(self) -> float:
        """Calculate absolute magnitude of correction applied."""
        return abs(self.correction)
    
    @property
    def relative_correction(self) -> float:
        """Calculate relative correction as percentage of raw value."""
        if self.raw_value == 0:
            return float('inf') if self.correction != 0 else 0.0
        return abs(self.correction / self.raw_value) * 100
    
    @property
    def model_confidence(self) -> Optional[float]:
        """Get confidence interval width as a measure of model uncertainty."""
        if self.confidence_interval is None:
            return None
        
        ci = self.confidence_interval
        if "lower" in ci and "upper" in ci:
            return ci["upper"] - ci["lower"]
        
        return None
    
    def plot_correction(
        self,
        show_confidence: bool = True,
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the correction applied by CDR.
        
        Args:
            show_confidence: Whether to show confidence intervals
            figsize: Figure size (width, height)
            save_path: Path to save the plot
            title: Custom plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Before/After comparison
        values = [self.raw_value, self.mitigated_value]
        labels = ['Raw Value', 'Mitigated Value']
        colors = ['red', 'blue']
        
        bars = ax1.bar(labels, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Expectation Value')
        ax1.set_title('CDR Correction Results')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Show confidence interval if available
        if show_confidence and self.confidence_interval:
            ci = self.confidence_interval
            if "lower" in ci and "upper" in ci:
                ax1.errorbar(
                    [1], [self.mitigated_value],
                    yerr=[[self.mitigated_value - ci["lower"]], 
                          [ci["upper"] - self.mitigated_value]],
                    fmt='none', color='black', capsize=5,
                    label='95% Confidence Interval'
                )
                ax1.legend()
        
        # Plot 2: Correction details
        correction_data = {
            'Correction': abs(self.correction),
            'Relative Correction (%)': self.relative_correction
        }
        
        if self.error_reduction is not None:
            correction_data['Error Reduction (%)'] = self.error_reduction * 100
        
        bars2 = ax2.bar(range(len(correction_data)), list(correction_data.values()), 
                       color=['green', 'orange', 'purple'][:len(correction_data)], alpha=0.7)
        ax2.set_xticks(range(len(correction_data)))
        ax2.set_xticklabels(list(correction_data.keys()), rotation=45, ha='right')
        ax2.set_ylabel('Value')
        ax2.set_title('Correction Metrics')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, correction_data.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        if title:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle(f'Clifford Data Regression Results ({self.regression_method})', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_analysis(
        self,
        training_data: Optional[List[Dict[str, Any]]] = None,
        figsize: tuple = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot analysis of training data and model performance.
        
        Args:
            training_data: Training data from CDR model
            figsize: Figure size
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure
        """
        if training_data is None:
            print("No training data provided for analysis")
            return plt.figure(figsize=figsize)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Extract training values
        noisy_values = [data['noisy_value'] for data in training_data]
        ideal_values = [data['ideal_value'] for data in training_data]
        
        # Plot 1: Noisy vs Ideal scatter
        ax1.scatter(ideal_values, noisy_values, alpha=0.7, color='blue')
        ax1.plot([min(ideal_values), max(ideal_values)], 
                [min(ideal_values), max(ideal_values)], 'r--', alpha=0.8)
        ax1.set_xlabel('Ideal Values')
        ax1.set_ylabel('Noisy Values')
        ax1.set_title('Training Data: Noisy vs Ideal')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        errors = np.array(noisy_values) - np.array(ideal_values)
        ax2.hist(errors, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Error (Noisy - Ideal)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(errors):.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training circuit length distribution
        if 'circuit' in training_data[0]:
            circuit_lengths = []
            for data in training_data:
                circuit = data['circuit']
                if hasattr(circuit, 'depth'):
                    circuit_lengths.append(circuit.depth())
                elif hasattr(circuit, 'get_depth'):
                    circuit_lengths.append(circuit.get_depth())
                else:
                    circuit_lengths.append(10)  # Default
            
            ax3.hist(circuit_lengths, bins=15, alpha=0.7, color='orange', edgecolor='black')
            ax3.set_xlabel('Circuit Depth')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Training Circuit Depth Distribution')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Circuit depth data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Model performance metrics
        metrics = {
            'Training Size': self.training_data_size,
            'Abs Correction': self.absolute_correction,
            'Rel Correction (%)': self.relative_correction
        }
        
        if self.error_reduction is not None:
            metrics['Error Reduction (%)'] = self.error_reduction * 100
        
        bars = ax4.bar(range(len(metrics)), list(metrics.values()), 
                      color=['purple', 'brown', 'pink', 'gray'][:len(metrics)], alpha=0.7)
        ax4.set_xticks(range(len(metrics)))
        ax4.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax4.set_ylabel('Value')
        ax4.set_title('Model Performance Metrics')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.suptitle(f'CDR Training Analysis ({self.regression_method})', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def summary(self) -> str:
        """Generate a summary string of the CDR results."""
        lines = [
            "=== Clifford Data Regression Results ===",
            f"Raw value:              {self.raw_value:.6f}",
            f"Mitigated value:        {self.mitigated_value:.6f}",
            f"Predicted ideal value:  {self.predicted_ideal_value:.6f}",
            f"Applied correction:     {self.correction:.6f}",
            f"",
            f"Regression method:      {self.regression_method}",
            f"Training circuits:      {self.training_data_size}",
            f"Absolute correction:    {self.absolute_correction:.6f}",
            f"Relative correction:    {self.relative_correction:.2f}%",
        ]
        
        if self.confidence_interval:
            ci = self.confidence_interval
            lines.extend([
                f"",
                f"Confidence interval:",
                f"  Lower bound:          {ci.get('lower', np.nan):.6f}",
                f"  Upper bound:          {ci.get('upper', np.nan):.6f}",
                f"  Interval width:       {self.model_confidence:.6f}"
            ])
        
        if self.error_reduction is not None:
            lines.extend([
                f"",
                f"Error reduction:        {self.error_reduction:.1%}",
                f"Improvement factor:     {self.improvement_factor:.2f}x"
            ])
        
        return "\\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "raw_value": self.raw_value,
            "mitigated_value": self.mitigated_value,
            "predicted_ideal_value": self.predicted_ideal_value,
            "correction": self.correction,
            "confidence_interval": self.confidence_interval,
            "error_reduction": self.error_reduction,
            "training_data_size": self.training_data_size,
            "regression_method": self.regression_method,
            "absolute_correction": self.absolute_correction,
            "relative_correction": self.relative_correction,
            "improvement_factor": self.improvement_factor,
            "model_confidence": self.model_confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CDRResult":
        """Create CDRResult from dictionary."""
        # Extract required fields
        required_fields = [
            "raw_value", "mitigated_value", "predicted_ideal_value", "correction"
        ]
        
        kwargs = {field: data[field] for field in required_fields}
        
        # Add optional fields
        optional_fields = [
            "confidence_interval", "error_reduction", "training_data_size",
            "regression_method", "config"
        ]
        
        for field in optional_fields:
            if field in data:
                kwargs[field] = data[field]
        
        return cls(**kwargs)
    
    def save(self, filepath: str) -> None:
        """Save result to file (JSON format)."""
        import json
        
        # Convert to serializable format
        data = self.to_dict()
        
        # Handle numpy objects
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
    def load(cls, filepath: str) -> "CDRResult":
        """Load result from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


@dataclass
class CDRBatchResult:
    """
    Results from batch Clifford Data Regression.
    
    Used when running CDR on multiple circuits or observables.
    """
    
    results: List[CDRResult]
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
    
    @property
    def mean_correction(self) -> float:
        """Calculate mean absolute correction across all results."""
        corrections = [r.absolute_correction for r in self.results]
        return np.mean(corrections)
    
    def plot_summary(
        self,
        metric: str = "error_reduction",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot summary of batch CDR results.
        
        Args:
            metric: Metric to plot ("error_reduction", "improvement_factor", "correction")
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
        elif metric == "correction":
            values = [r.absolute_correction for r in self.results]
            ylabel = "Absolute Correction"
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
        """Generate summary of batch CDR results."""
        lines = [
            "=== Batch CDR Results Summary ===",
            f"Total circuits:         {len(self.results)}",
            f"",
        ]
        
        # Error reduction statistics
        error_reductions = [r.error_reduction for r in self.results if r.error_reduction is not None]
        if error_reductions:
            lines.extend([
                f"Error Reduction:",
                f"  Mean:                 {np.mean(error_reductions):.1%}",
                f"  Std:                  {np.std(error_reductions):.1%}",
                f"  Min:                  {np.min(error_reductions):.1%}",
                f"  Max:                  {np.max(error_reductions):.1%}",
                f"",
            ])
        
        # Improvement factor statistics
        improvement_factors = [r.improvement_factor for r in self.results if r.improvement_factor is not None]
        if improvement_factors:
            lines.extend([
                f"Improvement Factor:",
                f"  Mean:                 {np.mean(improvement_factors):.2f}x",
                f"  Std:                  {np.std(improvement_factors):.2f}x",
                f"  Min:                  {np.min(improvement_factors):.2f}x",
                f"  Max:                  {np.max(improvement_factors):.2f}x",
                f"",
            ])
        
        # Correction statistics
        corrections = [r.absolute_correction for r in self.results]
        lines.extend([
            f"Absolute Correction:",
            f"  Mean:                 {np.mean(corrections):.6f}",
            f"  Std:                  {np.std(corrections):.6f}",
            f"  Min:                  {np.min(corrections):.6f}",
            f"  Max:                  {np.max(corrections):.6f}",
        ])
        
        return "\\n".join(lines)