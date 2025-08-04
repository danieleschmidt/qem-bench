"""Result classes for Probabilistic Error Cancellation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PECResult:
    """
    Results from Probabilistic Error Cancellation.
    
    Contains both raw and mitigated expectation values, along with
    detailed information about the decomposition and sampling process.
    
    Attributes:
        raw_value: Expectation value without error mitigation
        mitigated_value: PEC-mitigated expectation value
        decompositions: Quasi-probability decompositions for each noise channel
        sampling_overhead: Total sampling overhead factor
        sampling_data: Detailed information about importance sampling
        error_reduction: Fractional error reduction (if ideal value known)
        config: Configuration used for PEC
    """
    
    raw_value: float
    mitigated_value: float
    decompositions: Dict[str, Dict[str, float]]
    sampling_overhead: float
    sampling_data: Dict[str, Any]
    error_reduction: Optional[float] = None
    config: Optional[Any] = None
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result data after initialization."""
        if self.sampling_overhead <= 0:
            raise ValueError("sampling_overhead must be positive")
        
        if "num_samples" not in self.sampling_data:
            raise ValueError("sampling_data must contain 'num_samples'")
    
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
        ideal_value = self.sampling_data.get("ideal_value")
        if ideal_value is None:
            return None
        
        raw_error = abs(self.raw_value - ideal_value)
        mitigated_error = abs(self.mitigated_value - ideal_value)
        
        return raw_error - mitigated_error
    
    @property
    def num_samples(self) -> int:
        """Get number of importance sampling samples used."""
        return self.sampling_data.get("num_samples", 0)
    
    @property
    def effective_sample_size(self) -> float:
        """Get effective sample size from importance sampling."""
        return self.sampling_data.get("effective_samples", 0.0)
    
    @property
    def sampling_efficiency(self) -> float:
        """Calculate sampling efficiency as effective_samples / num_samples."""
        if self.num_samples == 0:
            return 0.0
        return self.effective_sample_size / self.num_samples
    
    @property
    def standard_error(self) -> float:
        """Get standard error of the mitigated estimate."""
        return self.sampling_data.get("std_error", 0.0)
    
    @property
    def variance(self) -> float:
        """Get variance of the importance sampling estimates."""
        return self.sampling_data.get("variance", 0.0)
    
    @property
    def total_quasi_prob_sum(self) -> float:
        """Calculate total sum of absolute quasi-probabilities."""
        total = 0.0
        for channel_decomp in self.decompositions.values():
            total += sum(abs(prob) for prob in channel_decomp.values())
        return total
    
    def plot_decompositions(
        self,
        figsize: tuple = (12, 8),
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot quasi-probability decompositions for all noise channels.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Path to save the plot
            title: Custom plot title
            
        Returns:
            matplotlib Figure object
        """
        n_channels = len(self.decompositions)
        if n_channels == 0:
            raise ValueError("No decompositions to plot")
        
        # Calculate subplot layout
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_channels == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for idx, (channel_name, decomposition) in enumerate(self.decompositions.items()):
            ax = axes_flat[idx]
            
            # Prepare data
            implementations = list(decomposition.keys())
            quasi_probs = list(decomposition.values())
            
            # Color bars based on sign
            colors = ['red' if prob < 0 else 'blue' for prob in quasi_probs]
            
            # Create bar plot
            bars = ax.bar(range(len(implementations)), quasi_probs, color=colors, alpha=0.7)
            
            # Customize subplot
            ax.set_title(f"Channel: {channel_name}")
            ax.set_xlabel("Implementation")
            ax.set_ylabel("Quasi-probability")
            ax.set_xticks(range(len(implementations)))
            ax.set_xticklabels(implementations, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add value labels on bars
            for bar, prob in zip(bars, quasi_probs):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    height + (0.01 if height >= 0 else -0.03),
                    f'{prob:.3f}',
                    ha='center', 
                    va='bottom' if height >= 0 else 'top',
                    fontsize=8
                )
        
        # Hide unused subplots
        for idx in range(n_channels, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        # Overall title
        if title is None:
            title = f'PEC Quasi-Probability Decompositions (Overhead: {self.sampling_overhead:.1f})'
        fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sampling_convergence(
        self,
        window_size: int = 100,
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot convergence of importance sampling estimates.
        
        Args:
            window_size: Size of moving average window
            figsize: Figure size
            save_path: Path to save plot
            title: Custom plot title
            
        Returns:
            matplotlib Figure
        """
        if "estimates" not in self.sampling_data:
            raise ValueError("Sampling data does not contain individual estimates")
        
        estimates = np.array(self.sampling_data["estimates"])
        
        # Calculate cumulative average
        cumulative_avg = np.cumsum(estimates) / np.arange(1, len(estimates) + 1)
        
        # Calculate moving average
        if len(estimates) >= window_size:
            moving_avg = np.convolve(estimates, np.ones(window_size)/window_size, mode='valid')
            moving_x = np.arange(window_size, len(estimates) + 1)
        else:
            moving_avg = cumulative_avg
            moving_x = np.arange(1, len(estimates) + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Individual estimates and convergence
        ax1.plot(estimates, alpha=0.3, color='gray', label='Individual estimates')
        ax1.plot(cumulative_avg, color='blue', linewidth=2, label='Cumulative average')
        if len(moving_avg) > 1:
            ax1.plot(moving_x, moving_avg, color='red', linewidth=2, 
                    label=f'Moving average (window={window_size})')
        
        # Add final mitigated value line
        ax1.axhline(y=self.mitigated_value, color='green', linestyle='--', 
                   linewidth=2, label=f'Final estimate: {self.mitigated_value:.4f}')
        
        # Add raw value for comparison
        ax1.axhline(y=self.raw_value, color='orange', linestyle='--', 
                   linewidth=2, label=f'Raw value: {self.raw_value:.4f}')
        
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Estimate Value')
        ax1.set_title('Sampling Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Running standard error
        if len(estimates) > 1:
            running_std = np.array([
                np.std(estimates[:i+1]) / np.sqrt(i+1) 
                for i in range(1, len(estimates))
            ])
            ax2.plot(range(2, len(estimates) + 1), running_std, color='purple', linewidth=2)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Standard Error')
            ax2.set_title('Running Standard Error')
            ax2.grid(True, alpha=0.3)
        
        if title is None:
            title = f'PEC Sampling Convergence ({self.num_samples} samples)'
        fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_weight_distribution(
        self,
        bins: int = 50,
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of importance sampling weights.
        
        Args:
            bins: Number of histogram bins
            figsize: Figure size
            save_path: Path to save plot
            title: Custom plot title
            
        Returns:
            matplotlib Figure
        """
        if "weights" not in self.sampling_data:
            raise ValueError("Sampling data does not contain weights")
        
        weights = np.array(self.sampling_data["weights"])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Weight histogram
        ax1.hist(weights, bins=bins, alpha=0.7, edgecolor='black', density=True)
        ax1.axvline(np.mean(weights), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(weights):.3f}')
        ax1.axvline(np.median(weights), color='green', linestyle='--', 
                   label=f'Median: {np.median(weights):.3f}')
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Weight Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log-scale weight histogram (if weights span large range)
        if np.max(np.abs(weights)) / np.min(np.abs(weights[weights != 0])) > 100:
            log_weights = np.log10(np.abs(weights[weights != 0]))
            ax2.hist(log_weights, bins=bins, alpha=0.7, edgecolor='black', density=True)
            ax2.set_xlabel('log₁₀(|Weight|)')
            ax2.set_ylabel('Density')
            ax2.set_title('Log Weight Distribution')
            ax2.grid(True, alpha=0.3)
        else:
            # Show weight vs. estimate scatter plot
            estimates = self.sampling_data.get("estimates", [])
            if len(estimates) == len(weights):
                ax2.scatter(weights, estimates, alpha=0.5)
                ax2.set_xlabel('Weight')
                ax2.set_ylabel('Estimate')
                ax2.set_title('Weight vs. Estimate')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No weight-estimate\ncorrelation data', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Weight Analysis')
        
        if title is None:
            title = f'PEC Importance Sampling Weights (Efficiency: {self.sampling_efficiency:.2%})'
        fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def summary(self) -> str:
        """Generate a summary string of the PEC results."""
        lines = [
            "=== Probabilistic Error Cancellation Results ===",
            f"Raw value:             {self.raw_value:.6f}",
            f"Mitigated value:       {self.mitigated_value:.6f}",
            f"Improvement:           {abs(self.mitigated_value - self.raw_value):.6f}",
            f"",
            f"Sampling Information:",
            f"  Total samples:       {self.num_samples}",
            f"  Effective samples:   {self.effective_sample_size:.1f}",
            f"  Sampling efficiency: {self.sampling_efficiency:.2%}",
            f"  Standard error:      {self.standard_error:.6f}",
            f"  Variance:            {self.variance:.6f}",
            f"",
            f"Decomposition Information:",
            f"  Noise channels:      {len(self.decompositions)}",
            f"  Sampling overhead:   {self.sampling_overhead:.1f}",
            f"  Total |quasi-prob|:  {self.total_quasi_prob_sum:.1f}",
        ]
        
        if self.error_reduction is not None:
            lines.extend([
                f"",
                f"Error Analysis:",
                f"  Error reduction:     {self.error_reduction:.1%}",
                f"  Improvement factor:  {self.improvement_factor:.2f}x"
            ])
        
        # Add decomposition details
        lines.extend([
            f"",
            f"Channel Decompositions:"
        ])
        
        for channel_name, decomposition in self.decompositions.items():
            channel_overhead = sum(abs(prob) for prob in decomposition.values())
            lines.append(f"  {channel_name}:")
            lines.append(f"    Overhead: {channel_overhead:.2f}")
            
            # Show top implementations by absolute quasi-probability
            sorted_impls = sorted(
                decomposition.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:3]  # Top 3
            
            for impl, prob in sorted_impls:
                lines.append(f"    {impl}: {prob:.4f}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "raw_value": self.raw_value,
            "mitigated_value": self.mitigated_value,
            "decompositions": self.decompositions,
            "sampling_overhead": self.sampling_overhead,
            "sampling_data": self.sampling_data,
            "error_reduction": self.error_reduction,
            "improvement_factor": self.improvement_factor,
            "absolute_error_reduction": self.absolute_error_reduction,
            "num_samples": self.num_samples,
            "effective_sample_size": self.effective_sample_size,
            "sampling_efficiency": self.sampling_efficiency,
            "standard_error": self.standard_error,
            "variance": self.variance,
            "total_quasi_prob_sum": self.total_quasi_prob_sum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PECResult":
        """Create PECResult from dictionary."""
        # Extract required fields
        required_fields = [
            "raw_value", "mitigated_value", "decompositions", 
            "sampling_overhead", "sampling_data"
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
        
        # Handle numpy arrays and complex numbers
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.complexfloating)):
                return float(obj.real) if obj.imag == 0 else complex(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, complex):
                return {"real": obj.real, "imag": obj.imag} if obj.imag != 0 else obj.real
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
    def load(cls, filepath: str) -> "PECResult":
        """Load result from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


@dataclass
class PECBatchResult:
    """
    Results from batch Probabilistic Error Cancellation.
    
    Used when running PEC on multiple circuits or observables.
    """
    
    results: List[PECResult]
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
    def mean_sampling_overhead(self) -> float:
        """Calculate mean sampling overhead across all results."""
        return np.mean([r.sampling_overhead for r in self.results])
    
    @property
    def mean_sampling_efficiency(self) -> float:
        """Calculate mean sampling efficiency across all results."""
        return np.mean([r.sampling_efficiency for r in self.results])
    
    def plot_summary(
        self,
        metric: str = "error_reduction",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot summary of batch results.
        
        Args:
            metric: Metric to plot ("error_reduction", "improvement_factor", 
                   "sampling_overhead", "sampling_efficiency")
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
        elif metric == "sampling_overhead":
            values = [r.sampling_overhead for r in self.results]
            ylabel = "Sampling Overhead"
        elif metric == "sampling_efficiency":
            values = [r.sampling_efficiency for r in self.results]
            ylabel = "Sampling Efficiency"
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
            "=== Batch PEC Results Summary ===",
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
        
        # Sampling statistics
        overheads = [r.sampling_overhead for r in self.results]
        efficiencies = [r.sampling_efficiency for r in self.results]
        
        lines.extend([
            f"Sampling Overhead:",
            f"  Mean:                {np.mean(overheads):.1f}",
            f"  Std:                 {np.std(overheads):.1f}",
            f"  Min:                 {np.min(overheads):.1f}",
            f"  Max:                 {np.max(overheads):.1f}",
            f"",
            f"Sampling Efficiency:",
            f"  Mean:                {np.mean(efficiencies):.1%}",
            f"  Std:                 {np.std(efficiencies):.1%}",
            f"  Min:                 {np.min(efficiencies):.1%}",
            f"  Max:                 {np.max(efficiencies):.1%}",
        ])
        
        return "\n".join(lines)