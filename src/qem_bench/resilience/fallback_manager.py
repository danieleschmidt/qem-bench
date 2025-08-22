"""
Fallback Manager for QEM-Bench

Provides graceful degradation and fallback strategies when 
primary operations fail.
"""

from typing import Callable, Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass


class FallbackStrategy(Enum):
    """Available fallback strategies."""
    SIMPLE_FALLBACK = "simple_fallback"
    CASCADING_FALLBACK = "cascading_fallback"
    CONDITIONAL_FALLBACK = "conditional_fallback"


@dataclass
class FallbackConfig:
    """Fallback configuration."""
    strategy: FallbackStrategy = FallbackStrategy.SIMPLE_FALLBACK
    max_fallback_attempts: int = 2
    enable_caching: bool = True


class FallbackManager:
    """Manages fallback operations for resilience."""
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self._fallback_cache: Dict[str, Any] = {}
    
    def execute_with_fallback(
        self,
        primary: Callable[[], Any],
        fallback: Callable[[], Any],
        operation_name: str = "unknown"
    ) -> Any:
        """Execute operation with fallback."""
        try:
            result = primary()
            # Cache successful result
            if self.config.enable_caching:
                self._fallback_cache[operation_name] = result
            return result
        except Exception:
            # Try fallback
            return fallback()
    
    def get_cached_result(self, operation_name: str) -> Optional[Any]:
        """Get cached result for fallback."""
        return self._fallback_cache.get(operation_name)