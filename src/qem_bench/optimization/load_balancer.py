"""
Load balancer implementation for quantum error mitigation.

This module provides load balancing functionality that is imported
by the auto_scaler module. It contains the core load balancing
algorithms and backend management.
"""

from .auto_scaler import LoadBalancer, LoadBalancingStrategy, BalancingStrategy

# Re-export the main classes for convenience
__all__ = ["LoadBalancer", "LoadBalancingStrategy"]

# Alias for backward compatibility
BalancingStrategy = LoadBalancingStrategy