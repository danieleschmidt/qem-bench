"""Composite noise model combining multiple noise sources."""

from typing import List, Any, Dict
from .base import NoiseModel, NoiseChannel


class CompositeNoiseModel(NoiseModel):
    """
    Composite noise model that combines multiple noise models.
    
    Applies noise from all constituent models in sequence.
    """
    
    def __init__(self, noise_models: List[NoiseModel], name: str = "composite"):
        """
        Initialize composite noise model.
        
        Args:
            noise_models: List of noise models to combine
            name: Name for the composite model
        """
        super().__init__(name)
        self.noise_models = noise_models
        
        # Combine all channels from constituent models
        for model in noise_models:
            for channel_name, channel in model.channels.items():
                # Avoid name conflicts
                composite_name = f"{model.name}_{channel_name}"
                self.channels[composite_name] = channel
    
    def get_noise_channels(self, circuit: Any) -> List[NoiseChannel]:
        """Get noise channels from all constituent models."""
        all_channels = []
        
        for model in self.noise_models:
            channels = model.get_noise_channels(circuit)
            all_channels.extend(channels)
        
        return all_channels
    
    def apply_noise(self, circuit: Any) -> Any:
        """Apply all noise models in sequence."""
        noisy_circuit = circuit
        
        for model in self.noise_models:
            noisy_circuit = model.apply_noise(noisy_circuit)
        
        return noisy_circuit
    
    def add_noise_model(self, noise_model: NoiseModel) -> None:
        """Add another noise model to the composite."""
        self.noise_models.append(noise_model)
        
        # Add channels with unique names
        for channel_name, channel in noise_model.channels.items():
            composite_name = f"{noise_model.name}_{channel_name}"
            self.channels[composite_name] = channel
    
    def remove_noise_model(self, model_name: str) -> None:
        """Remove a noise model by name."""
        self.noise_models = [m for m in self.noise_models if m.name != model_name]
        
        # Remove associated channels
        to_remove = [name for name in self.channels.keys() if name.startswith(f"{model_name}_")]
        for name in to_remove:
            del self.channels[name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict["constituent_models"] = [model.to_dict() for model in self.noise_models]
        return base_dict
    
    def __str__(self) -> str:
        """String representation."""
        lines = [f"CompositeNoiseModel: {self.name}"]
        lines.append(f"Constituent models: {len(self.noise_models)}")
        for model in self.noise_models:
            lines.append(f"  - {model.name} ({model.__class__.__name__})")
        lines.append(f"Total channels: {len(self.channels)}")
        return "\n".join(lines)