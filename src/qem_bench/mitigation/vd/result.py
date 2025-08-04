"""Result classes for Virtual Distillation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class VDResult:
    """
    Results from Virtual Distillation.
    
    Contains both raw and mitigated expectation values, along with
    detailed information about the distillation process.
    
    Attributes:
        raw_value: Expectation value from original circuit
        mitigated_value: Distilled expectation value
        num_copies: Number of copies (M) used in distillation
        verification_fidelity: Fidelity of verification circuits
        distillation_data: Detailed information about the distillation
        error_reduction: Fractional error reduction (if ideal value known)
        config: Configuration used for VD
    """
    
    raw_value: float
    mitigated_value: float
    num_copies: int
    verification_fidelity: float
    distillation_data: Dict[str, Any]
    error_reduction: Optional[float] = None
    config: Optional[Any] = None
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result data after initialization."""
        if self.num_copies < 1:
            raise ValueError("Number of copies must be at least 1")
        
        if not (0.0 <= self.verification_fidelity <= 1.0):
            raise ValueError("Verification fidelity must be between 0 and 1")
    
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
        ideal_value = self.distillation_data.get("ideal_value")
        if ideal_value is None:
            return None
        
        raw_error = abs(self.raw_value - ideal_value)
        mitigated_error = abs(self.mitigated_value - ideal_value)
        
        return raw_error - mitigated_error
    
    @property
    def distillation_method(self) -> str:
        """Get the distillation method used."""
        return self.distillation_data.get("method", "virtual_distillation")
    
    @property
    def verification_strategy(self) -> str:
        """Get the verification strategy used."""
        return self.distillation_data.get("verification_strategy", "unknown")
    
    @property
    def error_suppression_factor(self) -> float:
        """
        Calculate theoretical error suppression factor.
        
        For M-copy VD: ε_VD ≈ ε^M where ε is single-copy error rate.
        """
        return self.distillation_data.get("error_suppression_factor", 1.0)
    
    def plot_distillation(
        self,
        show_verification: bool = True,
        figsize: tuple = (12, 5),
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the distillation results and verification data.
        
        Args:
            show_verification: Whether to show verification circuit results
            figsize: Figure size (width, height)
            save_path: Path to save the plot
            title: Custom plot title
            
        Returns:
            matplotlib Figure object
        """
        if show_verification:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]/2, figsize[1]))
        
        # Plot 1: Raw vs Mitigated values
        values = [self.raw_value, self.mitigated_value]
        labels = ['Raw', 'Mitigated']
        colors = ['orange', 'blue']
        
        bars = ax1.bar(labels, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Expectation Value')
        ax1.set_title(f'VD Results (M={self.num_copies})')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom')
        
        # Add ideal value if available
        ideal_value = self.distillation_data.get("ideal_value")
        if ideal_value is not None:
            ax1.axhline(y=ideal_value, color='green', linestyle='--', 
                       label=f'Ideal: {ideal_value:.4f}')
            ax1.legend()
        
        # Plot 2: Verification data (if available and requested)
        if show_verification and "verification_data" in self.distillation_data:
            verification_data = self.distillation_data["verification_data"]
            
            if "fidelities" in verification_data:
                fidelities = verification_data["fidelities"]
                copy_indices = range(1, len(fidelities) + 1)
                
                ax2.plot(copy_indices, fidelities, 'o-', color='red', 
                        markersize=8, linewidth=2)
                ax2.set_xlabel('Copy Number')
                ax2.set_ylabel('Verification Fidelity')
                ax2.set_title('Verification Circuit Fidelities')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1)
                
                # Add mean fidelity line
                mean_fid = np.mean(fidelities)
                ax2.axhline(y=mean_fid, color='red', linestyle='--', alpha=0.7,
                           label=f'Mean: {mean_fid:.3f}')
                ax2.legend()
        
        if title is None:
            title = f'Virtual Distillation Results ({self.verification_strategy})'
        
        if show_verification:
            fig.suptitle(title)
        else:
            ax1.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def summary(self) -> str:
        """Generate a summary string of the VD results."""
        lines = [
            "=== Virtual Distillation Results ===",
            f"Raw value:             {self.raw_value:.6f}",
            f"Mitigated value:       {self.mitigated_value:.6f}",
            f"Improvement:           {abs(self.mitigated_value - self.raw_value):.6f}",
            f"",
            f"Number of copies (M):  {self.num_copies}",
            f"Verification fidelity: {self.verification_fidelity:.4f}",
            f"Verification strategy: {self.verification_strategy}",
            f"Error suppression:     {self.error_suppression_factor:.2f}x",
        ]
        
        if self.error_reduction is not None:
            lines.extend([
                f"",
                f"Error reduction:       {self.error_reduction:.1%}",
                f"Improvement factor:    {self.improvement_factor:.2f}x"
            ])
        
        # Add method-specific information
        if "execution_time" in self.distillation_data:
            lines.extend([
                f"",
                f"Execution time:        {self.distillation_data['execution_time']:.3f}s"
            ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "raw_value": self.raw_value,
            "mitigated_value": self.mitigated_value,
            "num_copies": self.num_copies,
            "verification_fidelity": self.verification_fidelity,
            "distillation_data": self.distillation_data,
            "error_reduction": self.error_reduction,
            "improvement_factor": self.improvement_factor,
            "absolute_error_reduction": self.absolute_error_reduction,
            "distillation_method": self.distillation_method,
            "verification_strategy": self.verification_strategy,
            "error_suppression_factor": self.error_suppression_factor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VDResult":
        """Create VDResult from dictionary."""
        # Extract required fields
        required_fields = [
            "raw_value", "mitigated_value", "num_copies", 
            "verification_fidelity", "distillation_data"
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
    def load(cls, filepath: str) -> "VDResult":
        """Load result from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


@dataclass
class VDBatchResult:
    """
    Results from batch Virtual Distillation.
    
    Used when running VD on multiple circuits or observables.
    """
    
    results: List[VDResult]
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
    def mean_verification_fidelity(self) -> float:
        """Calculate mean verification fidelity across all results."""
        fidelities = [r.verification_fidelity for r in self.results]
        return np.mean(fidelities)
    
    def plot_summary(
        self,
        metric: str = "error_reduction",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot summary of batch results.
        
        Args:
            metric: Metric to plot ("error_reduction", "improvement_factor", "verification_fidelity")
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
        elif metric == "verification_fidelity":
            values = [r.verification_fidelity for r in self.results]
            ylabel = "Verification Fidelity"
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
            "=== Batch VD Results Summary ===",
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
        
        # Verification fidelity statistics
        verification_fidelities = [r.verification_fidelity for r in self.results]
        lines.extend([
            f"Verification Fidelity:",
            f"  Mean:                {np.mean(verification_fidelities):.3f}",
            f"  Std:                 {np.std(verification_fidelities):.3f}",
            f"  Min:                 {np.min(verification_fidelities):.3f}",
            f"  Max:                 {np.max(verification_fidelities):.3f}",
            f"",
        ])
        
        # Copy number statistics
        copy_numbers = [r.num_copies for r in self.results]
        lines.extend([
            f"Number of Copies (M):",
            f"  Mean:                {np.mean(copy_numbers):.1f}",
            f"  Min:                 {np.min(copy_numbers)}",
            f"  Max:                 {np.max(copy_numbers)}",
        ])
        
        return "\n".join(lines)