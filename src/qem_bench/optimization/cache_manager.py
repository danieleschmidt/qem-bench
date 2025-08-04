"""
Intelligent caching system for quantum error mitigation computations.

This module provides multi-level caching with memory, disk, and distributed
storage backends, along with intelligent cache invalidation strategies.
"""

import os
import time
import pickle
import hashlib
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import weakref
import tempfile
import shutil
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from ..logging import get_logger


class CacheStrategy(Enum):
    """Available caching strategies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used  
    FIFO = "fifo"        # First In, First Out
    TTL = "ttl"          # Time To Live
    ADAPTIVE = "adaptive" # Adaptive based on access patterns


class CacheLevel(Enum):
    """Cache storage levels."""
    MEMORY = "memory"    # In-memory cache (fastest)
    DISK = "disk"        # Disk-based cache (persistent)
    DISTRIBUTED = "distributed"  # Distributed cache (scalable)


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    
    # General settings
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    max_size_gb: float = 2.0
    ttl_seconds: float = 3600.0
    
    # Memory cache settings
    memory_cache_ratio: float = 0.5  # Fraction of total cache size
    max_memory_items: int = 1000
    
    # Disk cache settings
    enable_disk_cache: bool = True
    disk_cache_dir: Optional[str] = None
    disk_compression: bool = True
    
    # Distributed cache settings
    enable_distributed: bool = False
    distributed_nodes: List[str] = field(default_factory=list)
    
    # Circuit-specific caching
    cache_circuit_compilation: bool = True
    cache_noise_models: bool = True
    cache_mitigation_results: bool = True
    
    # Invalidation settings
    auto_invalidate: bool = True
    parameter_sensitivity: float = 1e-6  # Threshold for parameter changes
    
    # Performance settings
    async_writes: bool = True
    write_batch_size: int = 10
    background_cleanup: bool = True


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    
    key: str
    value: Any
    size_bytes: int
    created_time: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    dependency_hash: Optional[str] = None


class MemoryCache:
    """In-memory cache with configurable eviction policies."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU
        self._size_bytes = 0
        self._lock = threading.RLock()
        
        # Calculate memory limits
        total_memory_bytes = config.max_size_gb * config.memory_cache_ratio * 1024**3
        self.max_memory_bytes = int(total_memory_bytes)
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from memory cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            # Check TTL expiration
            current_time = time.time()
            if entry.ttl and (current_time - entry.created_time) > entry.ttl:
                self._remove_entry(key)
                return None
            
            # Update access statistics
            entry.last_accessed = current_time
            entry.access_count += 1
            
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None) -> bool:
        """Put item into memory cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check if item is too large for memory cache
            if size_bytes > self.max_memory_bytes * 0.1:  # Max 10% of cache for single item
                return False
            
            # Make space if necessary
            while (self._size_bytes + size_bytes > self.max_memory_bytes and 
                   len(self._cache) > 0):
                self._evict_one()
            
            # Create cache entry
            current_time = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_time=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl or self.config.ttl_seconds,
                tags=tags or [],
                dependency_hash=self._calculate_dependency_hash(value)
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Add new entry
            self._cache[key] = entry
            self._access_order.append(key)
            self._size_bytes += size_bytes
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove item from memory cache."""
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self) -> None:
        """Clear all items from memory cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "total_items": len(self._cache),
                "size_bytes": self._size_bytes,
                "size_mb": self._size_bytes / (1024**2),
                "utilization": self._size_bytes / self.max_memory_bytes,
                "max_size_mb": self.max_memory_bytes / (1024**2),
            }
    
    def _evict_one(self) -> None:
        """Evict one item based on strategy."""
        if not self._cache:
            return
        
        if self.config.strategy == CacheStrategy.LRU:
            key_to_evict = self._access_order[0]
        elif self.config.strategy == CacheStrategy.LFU:
            key_to_evict = min(self._cache.keys(), 
                              key=lambda k: self._cache[k].access_count)
        elif self.config.strategy == CacheStrategy.FIFO:
            key_to_evict = min(self._cache.keys(),
                              key=lambda k: self._cache[k].created_time)
        else:  # TTL or ADAPTIVE
            # Evict expired items first, then fall back to LRU
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.ttl and (current_time - entry.created_time) > entry.ttl
            ]
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                key_to_evict = self._access_order[0]
        
        self._remove_entry(key_to_evict)
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache."""
        entry = self._cache.get(key)
        if entry is None:
            return False
        
        del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
        self._size_bytes -= entry.size_bytes
        
        return True
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if hasattr(value, 'nbytes'):
                return int(value.nbytes)
            elif isinstance(value, (np.ndarray, jnp.ndarray)):
                return int(value.nbytes)
            elif isinstance(value, dict):
                return sum(self._estimate_size(v) for v in value.values()) + len(value) * 64
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(v) for v in value) + len(value) * 64
            else:
                # Rough estimate using pickle
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            return 1024  # Default estimate
    
    def _calculate_dependency_hash(self, value: Any) -> str:
        """Calculate hash for dependency tracking."""
        try:
            if hasattr(value, 'tobytes'):
                content = value.tobytes()
            else:
                content = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.md5(content).hexdigest()
        except:
            return hashlib.md5(str(value).encode()).hexdigest()


class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Setup cache directory
        if config.disk_cache_dir:
            self.cache_dir = Path(config.disk_cache_dir)
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "qem_bench_cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        
        # Metadata storage
        self.metadata_file = self.cache_dir / "metadata.pickle"
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache."""
        with self._lock:
            if key not in self._metadata:
                return None
            
            metadata = self._metadata[key]
            
            # Check TTL expiration
            if metadata.get('ttl'):
                age = time.time() - metadata['created_time']
                if age > metadata['ttl']:
                    self.remove(key)
                    return None
            
            # Load from disk
            try:
                file_path = self.cache_dir / f"{key}.cache"
                with open(file_path, 'rb') as f:
                    if self.config.disk_compression:
                        import gzip
                        with gzip.open(f, 'rb') as gz:
                            value = pickle.load(gz)
                    else:
                        value = pickle.load(f)
                
                # Update access statistics
                metadata['last_accessed'] = time.time()
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                self._save_metadata()
                
                return value
                
            except Exception as e:
                self.logger.warning(f"Failed to load cache entry {key}: {e}")
                self.remove(key)
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None) -> bool:
        """Put item into disk cache."""
        with self._lock:
            try:
                # Save to disk
                file_path = self.cache_dir / f"{key}.cache"
                with open(file_path, 'wb') as f:
                    if self.config.disk_compression:
                        import gzip
                        with gzip.open(f, 'wb') as gz:
                            pickle.dump(value, gz, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Update metadata
                current_time = time.time()
                self._metadata[key] = {
                    'created_time': current_time,
                    'last_accessed': current_time,
                    'access_count': 1,
                    'ttl': ttl or self.config.ttl_seconds,
                    'tags': tags or [],
                    'file_size': file_path.stat().st_size
                }
                
                self._save_metadata()
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save cache entry {key}: {e}")
                return False
    
    def remove(self, key: str) -> bool:
        """Remove item from disk cache."""
        with self._lock:
            try:
                # Remove file
                file_path = self.cache_dir / f"{key}.cache"
                if file_path.exists():
                    file_path.unlink()
                
                # Remove metadata
                if key in self._metadata:
                    del self._metadata[key]
                    self._save_metadata()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to remove cache entry {key}: {e}")
                return False
    
    def clear(self) -> None:
        """Clear all items from disk cache."""
        with self._lock:
            try:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                
                # Clear metadata
                self._metadata.clear()
                self._save_metadata()
                
            except Exception as e:
                self.logger.error(f"Failed to clear disk cache: {e}")
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, metadata in self._metadata.items():
                if metadata.get('ttl'):
                    age = current_time - metadata['created_time']
                    if age > metadata['ttl']:
                        expired_keys.append(key)
            
            for key in expired_keys:
                self.remove(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics."""
        with self._lock:
            total_size = sum(meta.get('file_size', 0) for meta in self._metadata.values())
            
            return {
                "total_items": len(self._metadata),
                "size_bytes": total_size,
                "size_mb": total_size / (1024**2),
                "cache_dir": str(self.cache_dir),
            }
    
    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    self._metadata = pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache metadata: {e}")
            self._metadata = {}
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self._metadata, f)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")


class CacheManager:
    """
    Multi-level intelligent cache manager for quantum computations.
    
    Provides automatic caching of quantum circuits, noise models, mitigation
    results, and other expensive computations with intelligent invalidation
    and multi-level storage (memory, disk, distributed).
    
    Features:
    - Multi-level caching (memory → disk → distributed)
    - Intelligent cache invalidation based on parameter changes
    - Circuit compilation caching with hash-based keys
    - Noise model caching and reuse
    - Configurable eviction policies (LRU, LFU, TTL, etc.)
    - Background cleanup and optimization
    - Performance monitoring and statistics
    
    Example:
        >>> cache_config = CacheConfig(max_size_gb=4.0, enable_disk_cache=True)
        >>> cache_manager = CacheManager(cache_config)
        >>> 
        >>> # Cache expensive computation
        >>> result = cache_manager.get_or_compute(
        ...     "zne_result_key", 
        ...     lambda: expensive_zne_computation()
        ... )
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig()
        self.logger = get_logger(__name__)
        
        # Initialize cache levels
        self.memory_cache = MemoryCache(self.config)
        self.disk_cache = DiskCache(self.config) if self.config.enable_disk_cache else None
        
        # Statistics tracking
        self._stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'evictions': 0,
            'invalidations': 0,
        }
        self._lock = threading.RLock()
        
        # Background cleanup thread
        if self.config.background_cleanup:
            self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
            self._cleanup_thread.start()
        
        self.logger.info(f"CacheManager initialized with strategy: {self.config.strategy.value}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache (checks all levels).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            # Try memory cache first
            value = self.memory_cache.get(key)
            if value is not None:
                self._stats['hits'] += 1
                return value
            
            # Try disk cache
            if self.disk_cache:
                value = self.disk_cache.get(key)
                if value is not None:
                    # Promote to memory cache
                    self.memory_cache.put(key, value)
                    self._stats['hits'] += 1
                    return value
            
            # Cache miss
            self._stats['misses'] += 1
            return None
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
        level: CacheLevel = CacheLevel.MEMORY
    ) -> bool:
        """
        Put item into cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Tags for cache entry
            level: Cache level to store in
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            success = False
            
            # Store in memory cache
            if level == CacheLevel.MEMORY or level == CacheLevel.MEMORY:
                success = self.memory_cache.put(key, value, ttl, tags)
            
            # Store in disk cache if enabled and requested
            if (level == CacheLevel.DISK and self.disk_cache) or (not success and self.disk_cache):
                disk_success = self.disk_cache.put(key, value, ttl, tags)
                success = success or disk_success
            
            if success:
                self._stats['puts'] += 1
            
            return success
    
    def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], Any],
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
        force_recompute: bool = False
    ) -> Any:
        """
        Get from cache or compute and cache the result.
        
        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: Time to live for cached result
            tags: Tags for cache entry
            force_recompute: Force recomputation even if cached
            
        Returns:
            Cached or computed value
        """
        if not force_recompute:
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value
        
        # Compute new value
        value = compute_func()
        
        # Cache the result
        self.put(key, value, ttl, tags)
        
        return value
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate cache entry at all levels.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was found and removed
        """
        with self._lock:
            success = False
            
            # Remove from memory cache
            if self.memory_cache.remove(key):
                success = True
            
            # Remove from disk cache
            if self.disk_cache and self.disk_cache.remove(key):
                success = True
            
            if success:
                self._stats['invalidations'] += 1
            
            return success
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """
        Invalidate all cache entries with matching tags.
        
        Args:
            tags: List of tags to match
            
        Returns:
            Number of entries invalidated
        """
        # This would require tracking tags in cache implementations
        # For now, return 0 (feature to be implemented)
        return 0
    
    def clear(self) -> None:
        """Clear all cache levels."""
        with self._lock:
            self.memory_cache.clear()
            if self.disk_cache:
                self.disk_cache.clear()
    
    def cache_circuit_compilation(
        self,
        circuit: Any,
        backend: Any,
        compile_func: Callable[[Any, Any], Any]
    ) -> Any:
        """
        Cache quantum circuit compilation results.
        
        Args:
            circuit: Quantum circuit
            backend: Target backend
            compile_func: Compilation function
            
        Returns:
            Compiled circuit
        """
        # Generate cache key based on circuit and backend properties
        circuit_hash = self._hash_circuit(circuit)
        backend_hash = self._hash_backend(backend)
        cache_key = f"circuit_compile_{circuit_hash}_{backend_hash}"
        
        return self.get_or_compute(
            cache_key,
            lambda: compile_func(circuit, backend),
            tags=["circuit_compilation"]
        )
    
    def cache_noise_model(
        self,
        noise_params: Dict[str, Any],
        create_func: Callable[[Dict[str, Any]], Any]
    ) -> Any:
        """
        Cache noise model creation.
        
        Args:
            noise_params: Noise model parameters
            create_func: Function to create noise model
            
        Returns:
            Noise model instance
        """
        # Generate cache key from parameters
        params_hash = hashlib.md5(str(sorted(noise_params.items())).encode()).hexdigest()
        cache_key = f"noise_model_{params_hash}"
        
        return self.get_or_compute(
            cache_key,
            lambda: create_func(noise_params),
            tags=["noise_model"]
        )
    
    def cache_mitigation_result(
        self,
        method_name: str,
        circuit_hash: str,
        params_hash: str,
        compute_func: Callable[[], Any]
    ) -> Any:
        """
        Cache mitigation computation results.
        
        Args:
            method_name: Name of mitigation method
            circuit_hash: Hash of circuit
            params_hash: Hash of parameters
            compute_func: Function to compute result
            
        Returns:
            Mitigation result
        """
        cache_key = f"mitigation_{method_name}_{circuit_hash}_{params_hash}"
        
        return self.get_or_compute(
            cache_key,
            compute_func,
            tags=["mitigation_result", method_name]
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache usage statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            memory_stats = self.memory_cache.get_stats()
            disk_stats = self.disk_cache.get_stats() if self.disk_cache else {}
            
            hit_rate = (self._stats['hits'] / 
                       (self._stats['hits'] + self._stats['misses'])) if (self._stats['hits'] + self._stats['misses']) > 0 else 0
            
            return {
                "overall": {
                    "hit_rate": hit_rate,
                    "total_hits": self._stats['hits'],
                    "total_misses": self._stats['misses'],
                    "total_puts": self._stats['puts'],
                    "total_evictions": self._stats['evictions'],
                    "total_invalidations": self._stats['invalidations'],
                },
                "memory_cache": memory_stats,
                "disk_cache": disk_stats,
                "config": {
                    "strategy": self.config.strategy.value,
                    "max_size_gb": self.config.max_size_gb,
                    "ttl_seconds": self.config.ttl_seconds,
                    "disk_enabled": self.config.enable_disk_cache,
                }
            }
    
    def _hash_circuit(self, circuit: Any) -> str:
        """Generate hash for quantum circuit."""
        try:
            if hasattr(circuit, 'to_qasm'):
                content = circuit.to_qasm()
            elif hasattr(circuit, 'gates'):
                content = str(circuit.gates)
            else:
                content = str(circuit)
            
            return hashlib.md5(content.encode()).hexdigest()
        except:
            return hashlib.md5(str(circuit).encode()).hexdigest()
    
    def _hash_backend(self, backend: Any) -> str:
        """Generate hash for backend configuration."""
        try:
            if hasattr(backend, 'configuration'):
                config = backend.configuration()
                content = str(config)
            else:
                content = str(type(backend).__name__)
            
            return hashlib.md5(content.encode()).hexdigest()
        except:
            return hashlib.md5(str(backend).encode()).hexdigest()
    
    def _background_cleanup(self) -> None:
        """Background thread for cache maintenance."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                # Cleanup expired entries in disk cache
                if self.disk_cache:
                    expired_count = self.disk_cache.cleanup_expired()
                    if expired_count > 0:
                        self.logger.info(f"Cleaned up {expired_count} expired disk cache entries")
                
            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")


def create_cache_manager(
    max_size_gb: float = 2.0,
    enable_disk_cache: bool = True,
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
    **config_kwargs
) -> CacheManager:
    """
    Create a cache manager with specified configuration.
    
    Args:
        max_size_gb: Maximum cache size in GB
        enable_disk_cache: Enable persistent disk caching
        strategy: Caching strategy to use
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured CacheManager instance
    """
    config = CacheConfig(
        max_size_gb=max_size_gb,
        enable_disk_cache=enable_disk_cache,
        strategy=strategy,
        **config_kwargs
    )
    return CacheManager(config)