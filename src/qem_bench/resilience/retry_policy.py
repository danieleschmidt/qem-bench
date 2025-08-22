"""
Retry Policy Implementation for QEM-Bench

Provides configurable retry strategies with exponential backoff,
jitter, and circuit breaker integration.
"""

import time
import random
from enum import Enum
from typing import Callable, Any, Optional, Type
from dataclasses import dataclass


class RetryStrategy(Enum):
    """Available retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class BackoffStrategy(Enum):
    """Backoff strategies for retry delays."""
    NO_JITTER = "no_jitter"
    FULL_JITTER = "full_jitter"
    EQUAL_JITTER = "equal_jitter"
    DECORRELATED_JITTER = "decorrelated_jitter"


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff: BackoffStrategy = BackoffStrategy.EQUAL_JITTER
    
    # Exception handling
    retry_on: tuple = (Exception,)
    stop_on: tuple = (KeyboardInterrupt,)


def calculate_delay(
    attempt: int,
    policy: RetryPolicy,
    last_delay: Optional[float] = None
) -> float:
    """Calculate retry delay based on policy."""
    if policy.strategy == RetryStrategy.FIXED_DELAY:
        base_delay = policy.base_delay
    
    elif policy.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        base_delay = policy.base_delay * (policy.multiplier ** attempt)
    
    elif policy.strategy == RetryStrategy.LINEAR_BACKOFF:
        base_delay = policy.base_delay * (1 + attempt)
    
    elif policy.strategy == RetryStrategy.FIBONACCI_BACKOFF:
        # Fibonacci sequence for delays
        if attempt <= 1:
            base_delay = policy.base_delay
        else:
            fib_a, fib_b = 1, 1
            for _ in range(attempt - 1):
                fib_a, fib_b = fib_b, fib_a + fib_b
            base_delay = policy.base_delay * fib_b
    
    else:
        base_delay = policy.base_delay
    
    # Apply jitter
    if policy.backoff == BackoffStrategy.NO_JITTER:
        delay = base_delay
    
    elif policy.backoff == BackoffStrategy.FULL_JITTER:
        delay = random.uniform(0, base_delay)
    
    elif policy.backoff == BackoffStrategy.EQUAL_JITTER:
        delay = base_delay / 2 + random.uniform(0, base_delay / 2)
    
    elif policy.backoff == BackoffStrategy.DECORRELATED_JITTER:
        if last_delay is None:
            delay = random.uniform(policy.base_delay, base_delay)
        else:
            delay = random.uniform(policy.base_delay, last_delay * 3)
    
    else:
        delay = base_delay
    
    # Clamp to max delay
    return min(delay, policy.max_delay)


class RetryExecutor:
    """Executor for retry operations."""
    
    def __init__(self, policy: RetryPolicy):
        self.policy = policy
    
    def execute(
        self,
        operation: Callable[[], Any],
        operation_name: str = "unknown"
    ) -> Any:
        """Execute operation with retry policy."""
        last_exception = None
        last_delay = None
        
        for attempt in range(self.policy.max_attempts):
            try:
                return operation()
                
            except self.policy.stop_on:
                # Don't retry on stop conditions
                raise
                
            except self.policy.retry_on as e:
                last_exception = e
                
                if attempt >= self.policy.max_attempts - 1:
                    break  # Last attempt
                
                # Calculate and apply delay
                delay = calculate_delay(attempt, self.policy, last_delay)
                last_delay = delay
                
                time.sleep(delay)
            
            except Exception as e:
                # Don't retry on unexpected exceptions
                raise
        
        # All retries exhausted
        raise last_exception