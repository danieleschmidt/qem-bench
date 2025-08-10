"""
Advanced caching system for QEM-Bench performance optimization.

Provides intelligent caching for quantum circuits, simulation results, and
mitigation data with automatic invalidation and memory management.
"""

import hashlib
import pickle
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import json
import zlib
from collections import OrderedDict
from functools import wraps
import weakref

from ..monitoring.logger import get_logger


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.timestamp
    
    def size_mb(self) -> float:
        """Get size in MB."""
        return self.size_bytes / (1024 * 1024)


class CacheEvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def should_evict(self, entry: CacheEntry, cache_size: int, max_size: int) -> bool:
        """Determine if entry should be evicted."""
        pass
    
    @abstractmethod
    def get_eviction_priority(self, entry: CacheEntry) -> float:
        """Get eviction priority (higher = more likely to evict)."""
        pass


class LRUEvictionPolicy(CacheEvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def should_evict(self, entry: CacheEntry, cache_size: int, max_size: int) -> bool:
        """Evict when cache is over capacity."""
        return cache_size > max_size
    
    def get_eviction_priority(self, entry: CacheEntry) -> float:
        """Priority based on last access time (older = higher priority)."""
        return -entry.last_accessed  # Negative for ascending sort


class LFUEvictionPolicy(CacheEvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def should_evict(self, entry: CacheEntry, cache_size: int, max_size: int) -> bool:
        """Evict when cache is over capacity."""
        return cache_size > max_size
    
    def get_eviction_priority(self, entry: CacheEntry) -> float:
        """Priority based on access count (lower count = higher priority)."""
        return entry.access_count


class TTLEvictionPolicy(CacheEvictionPolicy):
    """Time-To-Live eviction policy."""
    
    def __init__(self, default_ttl: float = 3600):  # 1 hour default
        self.default_ttl = default_ttl
    
    def should_evict(self, entry: CacheEntry, cache_size: int, max_size: int) -> bool:
        """Evict expired entries or when over capacity."""
        return entry.is_expired() or cache_size > max_size
    
    def get_eviction_priority(self, entry: CacheEntry) -> float:
        """Priority based on expiration (expired first, then oldest)."""
        if entry.is_expired():
            return float('inf')  # Highest priority for expired entries
        return -entry.timestamp


class AdaptiveEvictionPolicy(CacheEvictionPolicy):
    """Adaptive eviction policy that combines multiple strategies."""
    
    def __init__(self, lru_weight: float = 0.4, lfu_weight: float = 0.3, 
                 ttl_weight: float = 0.3):
        self.lru_weight = lru_weight
        self.lfu_weight = lfu_weight
        self.ttl_weight = ttl_weight
        
        self.lru = LRUEvictionPolicy()
        self.lfu = LFUEvictionPolicy()
        self.ttl = TTLEvictionPolicy()
    
    def should_evict(self, entry: CacheEntry, cache_size: int, max_size: int) -> bool:
        """Evict based on combined policies."""
        return (entry.is_expired() or 
                cache_size > max_size or
                entry.age_seconds() > 7200)  # 2 hours max age
    
    def get_eviction_priority(self, entry: CacheEntry) -> float:
        """Weighted combination of LRU, LFU, and TTL priorities."""
        lru_priority = self.lru.get_eviction_priority(entry)
        lfu_priority = self.lfu.get_eviction_priority(entry)
        ttl_priority = self.ttl.get_eviction_priority(entry)
        
        # Normalize priorities to [0, 1] range
        current_time = time.time()
        normalized_lru = min(1.0, max(0.0, (current_time - entry.last_accessed) / 3600))
        normalized_lfu = min(1.0, max(0.0, 1.0 / (entry.access_count + 1)))
        normalized_ttl = min(1.0, max(0.0, entry.age_seconds() / 7200))
        
        return (self.lru_weight * normalized_lru + 
                self.lfu_weight * normalized_lfu + 
                self.ttl_weight * normalized_ttl)


class MemoryAwareCache:
    """High-performance memory-aware cache with intelligent eviction."""
    
    def __init__(self, 
                 max_size_mb: float = 500,  # 500 MB default
                 max_entries: int = 10000,
                 eviction_policy: Optional[CacheEvictionPolicy] = None,
                 compression_threshold: int = 1024):  # Compress entries > 1KB
        
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.compression_threshold = compression_threshold
        
        self.eviction_policy = eviction_policy or AdaptiveEvictionPolicy()
        
        self._cache: Dict[str, CacheEntry] = {}
        self._current_size_bytes = 0
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0, 'misses': 0, 'evictions': 0, 'compressions': 0
        }
        
        self.logger = get_logger("cache")
    
    def _compute_key(self, key_data: Any) -> str:
        """Compute stable hash key from data."""
        if isinstance(key_data, str):
            return key_data
        
        # Create deterministic hash from complex objects
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            # Fallback estimation
            if hasattr(obj, '__sizeof__'):
                return obj.__sizeof__()
            return 1024  # Default estimate
    
    def _compress_value(self, value: Any) -> Tuple[Any, bool]:
        """Compress value if beneficial."""
        try:
            serialized = pickle.dumps(value)
            if len(serialized) > self.compression_threshold:
                compressed = zlib.compress(serialized)
                if len(compressed) < len(serialized) * 0.8:  # 20% compression benefit
                    self._stats['compressions'] += 1
                    return compressed, True
            return value, False
        except:
            return value, False
    
    def _decompress_value(self, value: Any, is_compressed: bool) -> Any:
        """Decompress value if needed."""
        if not is_compressed:
            return value
        
        try:
            decompressed = zlib.decompress(value)
            return pickle.loads(decompressed)
        except:
            return value
    
    def _evict_if_needed(self):
        """Evict entries based on policy."""
        if (len(self._cache) <= self.max_entries and 
            self._current_size_bytes <= self.max_size_bytes):
            return
        
        # Get eviction candidates
        candidates = []
        for entry in self._cache.values():
            if self.eviction_policy.should_evict(
                entry, self._current_size_bytes, self.max_size_bytes
            ):
                priority = self.eviction_policy.get_eviction_priority(entry)
                candidates.append((priority, entry.key))
        
        # Sort by eviction priority and evict
        candidates.sort(reverse=True)  # Highest priority first
        
        target_size = int(self.max_size_bytes * 0.8)  # Evict to 80% capacity
        target_entries = int(self.max_entries * 0.8)
        
        evicted_count = 0
        for priority, key in candidates:
            if (self._current_size_bytes <= target_size and 
                len(self._cache) <= target_entries):
                break
            
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                evicted_count += 1
                self._stats['evictions'] += 1
        
        if evicted_count > 0:
            self.logger.debug(f"Evicted {evicted_count} cache entries")
    
    def put(self, key: Any, value: Any, ttl_seconds: Optional[float] = None,
            tags: Optional[List[str]] = None) -> bool:
        """Store value in cache."""
        with self._lock:
            cache_key = self._compute_key(key)
            
            # Compress if beneficial
            stored_value, is_compressed = self._compress_value(value)
            size_bytes = self._estimate_size(stored_value)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=(stored_value, is_compressed),
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                tags=tags or []
            )
            
            # Remove existing entry if present
            if cache_key in self._cache:
                self._current_size_bytes -= self._cache[cache_key].size_bytes
            
            # Add new entry
            self._cache[cache_key] = entry
            self._current_size_bytes += size_bytes
            
            # Evict if needed
            self._evict_if_needed()
            
            return True
    
    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value from cache."""
        with self._lock:
            cache_key = self._compute_key(key)
            
            if cache_key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[cache_key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[cache_key]
                self._current_size_bytes -= entry.size_bytes
                self._stats['misses'] += 1
                return None
            
            # Update access statistics
            entry.touch()
            self._stats['hits'] += 1
            
            # Decompress value
            stored_value, is_compressed = entry.value
            return self._decompress_value(stored_value, is_compressed)
    
    def contains(self, key: Any) -> bool:
        """Check if key exists in cache."""
        return self.get(key) is not None
    
    def remove(self, key: Any) -> bool:
        """Remove entry from cache."""
        with self._lock:
            cache_key = self._compute_key(key)
            
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                del self._cache[cache_key]
                self._current_size_bytes -= entry.size_bytes
                return True
            
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            self.logger.info("Cache cleared")
    
    def clear_by_tags(self, tags: List[str]):
        """Clear cache entries by tags."""
        with self._lock:
            to_remove = []
            for key, entry in self._cache.items():
                if any(tag in entry.tags for tag in tags):
                    to_remove.append(key)
            
            removed_count = 0
            for key in to_remove:
                if key in self._cache:
                    entry = self._cache[key]
                    del self._cache[key]
                    self._current_size_bytes -= entry.size_bytes
                    removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleared {removed_count} entries by tags: {tags}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests) if total_requests > 0 else 0
            
            return {
                'entries': len(self._cache),
                'size_mb': self._current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': self._current_size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions'],
                'compressions': self._stats['compressions']
            }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        stats = self.get_stats()
        
        with self._lock:
            # Entry statistics
            if self._cache:
                sizes = [entry.size_bytes for entry in self._cache.values()]
                ages = [entry.age_seconds() for entry in self._cache.values()]
                access_counts = [entry.access_count for entry in self._cache.values()]
                
                stats.update({
                    'avg_entry_size_kb': sum(sizes) / len(sizes) / 1024,
                    'max_entry_size_kb': max(sizes) / 1024,
                    'avg_entry_age_seconds': sum(ages) / len(ages),
                    'max_entry_age_seconds': max(ages),
                    'avg_access_count': sum(access_counts) / len(access_counts),
                    'max_access_count': max(access_counts)
                })
            
            return stats


# Global cache instances for different types
_circuit_cache = MemoryAwareCache(max_size_mb=200, eviction_policy=LRUEvictionPolicy())
_simulation_cache = MemoryAwareCache(max_size_mb=300, eviction_policy=AdaptiveEvictionPolicy())
_mitigation_cache = MemoryAwareCache(max_size_mb=100, eviction_policy=TTLEvictionPolicy(1800))  # 30 min TTL


def cached(cache: Optional[MemoryAwareCache] = None, 
          ttl_seconds: Optional[float] = None,
          tags: Optional[List[str]] = None,
          key_func: Optional[Callable] = None):
    """Decorator to add caching to functions."""
    
    def decorator(func: Callable):
        # Use default cache if none specified
        target_cache = cache or _simulation_cache
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            
            # Try cache first
            cached_result = target_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            target_cache.put(cache_key, result, ttl_seconds, tags)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: target_cache.clear()
        wrapper.cache_stats = lambda: target_cache.get_stats()
        wrapper.cache_remove = lambda key: target_cache.remove(key)
        
        return wrapper
    
    return decorator


# Specialized caches for quantum computing
class QuantumCircuitCache(MemoryAwareCache):
    """Specialized cache for quantum circuits."""
    
    def __init__(self):
        super().__init__(max_size_mb=200, eviction_policy=LRUEvictionPolicy())
    
    def cache_circuit_compilation(self, circuit: Any, backend: str) -> str:
        """Cache compiled circuit representation."""
        key = f"compiled:{backend}:{self._circuit_hash(circuit)}"
        return key
    
    def cache_circuit_optimization(self, circuit: Any, optimization_level: int) -> str:
        """Cache optimized circuit."""
        key = f"optimized:{optimization_level}:{self._circuit_hash(circuit)}"
        return key
    
    def _circuit_hash(self, circuit: Any) -> str:
        """Generate hash for quantum circuit."""
        try:
            # Create deterministic representation
            circuit_data = {
                'num_qubits': getattr(circuit, 'num_qubits', 0),
                'gates': []
            }
            
            if hasattr(circuit, 'gates'):
                for gate in circuit.gates:
                    gate_data = {
                        'name': gate.get('name', ''),
                        'qubits': gate.get('qubits', []),
                        'params': gate.get('params', [])
                    }
                    circuit_data['gates'].append(gate_data)
            
            key_str = json.dumps(circuit_data, sort_keys=True)
            return hashlib.sha256(key_str.encode()).hexdigest()[:16]
        
        except Exception:
            # Fallback to object id
            return str(id(circuit))


class SimulationResultCache(MemoryAwareCache):
    """Specialized cache for simulation results."""
    
    def __init__(self):
        super().__init__(
            max_size_mb=300, 
            eviction_policy=AdaptiveEvictionPolicy(),
            compression_threshold=512  # Compress smaller results
        )
    
    def cache_statevector(self, circuit: Any, backend_config: Dict[str, Any]) -> str:
        """Cache statevector simulation result."""
        key = f"statevector:{self._simulation_key(circuit, backend_config)}"
        return key
    
    def cache_sampling(self, circuit: Any, shots: int, backend_config: Dict[str, Any]) -> str:
        """Cache sampling simulation result."""
        key = f"sampling:{shots}:{self._simulation_key(circuit, backend_config)}"
        return key
    
    def _simulation_key(self, circuit: Any, backend_config: Dict[str, Any]) -> str:
        """Generate key for simulation caching."""
        from .quantum_circuit_cache import QuantumCircuitCache
        
        circuit_hash = QuantumCircuitCache()._circuit_hash(circuit)
        config_str = json.dumps(backend_config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        
        return f"{circuit_hash}:{config_hash}"


# Global specialized caches
circuit_cache = QuantumCircuitCache()
simulation_cache = SimulationResultCache()
mitigation_cache = _mitigation_cache


# Cache management functions
def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches."""
    return {
        'circuit_cache': circuit_cache.get_stats(),
        'simulation_cache': simulation_cache.get_stats(),
        'mitigation_cache': mitigation_cache.get_stats()
    }


def clear_all_caches():
    """Clear all caches."""
    circuit_cache.clear()
    simulation_cache.clear()
    mitigation_cache.clear()


def warm_cache_with_common_circuits():
    """Pre-populate cache with commonly used circuits."""
    logger = get_logger("cache_warming")
    
    try:
        from ..benchmarks.circuits.standard import create_benchmark_circuit
        
        common_circuits = [
            ("bell_state", {"qubits": 2}),
            ("ghz_state", {"qubits": 3}),
            ("ghz_state", {"qubits": 4}),
            ("random_circuit", {"qubits": 3, "depth": 5}),
            ("quantum_volume", {"qubits": 3, "depth": 3}),
        ]
        
        cached_count = 0
        for circuit_type, params in common_circuits:
            try:
                circuit = create_benchmark_circuit(circuit_type, **params)
                key = circuit_cache._circuit_hash(circuit)
                circuit_cache.put(f"benchmark:{circuit_type}:{key}", circuit, ttl_seconds=3600)
                cached_count += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache for {circuit_type}: {e}")
        
        logger.info(f"Warmed cache with {cached_count} common circuits")
    
    except ImportError:
        logger.warning("Cannot warm cache - benchmark circuits not available")


# Cache performance monitoring
class CachePerformanceMonitor:
    """Monitor cache performance and suggest optimizations."""
    
    def __init__(self):
        self.logger = get_logger("cache_monitor")
        self._last_stats = {}
    
    def analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache performance and provide recommendations."""
        current_stats = get_cache_stats()
        recommendations = []
        
        for cache_name, stats in current_stats.items():
            cache_recommendations = []
            
            # Hit rate analysis
            if stats['hit_rate'] < 0.3:
                cache_recommendations.append(
                    f"Low hit rate ({stats['hit_rate']:.1%}) - consider increasing cache size or TTL"
                )
            elif stats['hit_rate'] > 0.9:
                cache_recommendations.append(
                    f"Excellent hit rate ({stats['hit_rate']:.1%}) - cache is well-tuned"
                )
            
            # Size utilization analysis
            if stats['utilization'] > 0.9:
                cache_recommendations.append(
                    "Cache nearly full - consider increasing max size or more aggressive eviction"
                )
            elif stats['utilization'] < 0.3:
                cache_recommendations.append(
                    "Cache underutilized - could reduce max size to save memory"
                )
            
            # Eviction analysis
            if stats['evictions'] > stats['hits'] * 0.1:
                cache_recommendations.append(
                    "High eviction rate - consider increasing cache size or adjusting eviction policy"
                )
            
            recommendations.append({
                'cache': cache_name,
                'recommendations': cache_recommendations
            })
        
        return {
            'stats': current_stats,
            'recommendations': recommendations
        }
    
    def log_performance_report(self):
        """Log cache performance report."""
        analysis = self.analyze_cache_performance()
        
        for cache_info in analysis['recommendations']:
            cache_name = cache_info['cache']
            stats = analysis['stats'][cache_name]
            
            self.logger.info(
                f"Cache {cache_name}: {stats['hit_rate']:.1%} hit rate, "
                f"{stats['utilization']:.1%} full, {stats['entries']} entries",
                "cache_performance",
                stats
            )
            
            for rec in cache_info['recommendations']:
                self.logger.info(f"Cache {cache_name} recommendation: {rec}")


# Global cache monitor
cache_monitor = CachePerformanceMonitor()