"""
Memory management and pooling for quantum state vectors and large datasets.

This module provides intelligent memory allocation, pooling, and cleanup
strategies to optimize memory usage in quantum error mitigation computations.
"""

import gc
import mmap
import os
import tempfile
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import psutil

import numpy as np
import jax.numpy as jnp

from ..logging import get_logger


class MemoryStrategy(Enum):
    """Memory allocation strategies."""
    POOL = "pool"              # Memory pooling
    DIRECT = "direct"          # Direct allocation
    MAPPED = "mapped"          # Memory-mapped files
    STREAMING = "streaming"    # Streaming processing
    HYBRID = "hybrid"          # Hybrid approach


class CleanupStrategy(Enum):
    """Memory cleanup strategies."""
    IMMEDIATE = "immediate"    # Immediate cleanup
    DEFERRED = "deferred"      # Deferred cleanup
    THRESHOLD = "threshold"    # Threshold-based cleanup
    PERIODIC = "periodic"      # Periodic cleanup


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    
    # Pool settings
    pool_size_gb: float = 4.0
    max_block_size_mb: float = 512.0
    min_block_size_mb: float = 1.0
    pool_growth_factor: float = 1.5
    
    # Memory mapping settings
    enable_memory_mapping: bool = True
    temp_dir: Optional[str] = None
    mmap_threshold_mb: float = 100.0
    
    # Cleanup settings
    cleanup_strategy: CleanupStrategy = CleanupStrategy.THRESHOLD
    gc_threshold: float = 0.8  # Trigger cleanup at 80% memory usage
    cleanup_interval: float = 30.0  # Cleanup every 30 seconds
    
    # Monitoring settings
    enable_monitoring: bool = True
    monitor_interval: float = 5.0
    memory_warning_threshold: float = 0.9
    
    # Performance settings
    enable_preallocation: bool = True
    zero_fill_blocks: bool = False
    use_huge_pages: bool = False  # Linux huge pages support
    
    # Streaming settings
    stream_chunk_size_mb: float = 50.0
    max_streams: int = 4


@dataclass 
class MemoryBlock:
    """Represents a memory block in the pool."""
    
    id: str
    data: Union[np.ndarray, jnp.ndarray]
    size_bytes: int
    allocated_time: float
    last_accessed: float
    ref_count: int = 0
    is_locked: bool = False
    tags: List[str] = field(default_factory=list)


class MemoryPool:
    """Memory pool for efficient allocation and deallocation."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Pool storage
        self.blocks: Dict[str, MemoryBlock] = {}
        self.free_blocks: Dict[int, List[str]] = {}  # Size -> block IDs
        self.allocated_blocks: Dict[str, MemoryBlock] = {}
        
        # Pool statistics
        self.total_size_bytes = 0
        self.allocated_bytes = 0
        self.peak_usage_bytes = 0
        self.allocation_count = 0
        self.deallocation_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Calculate pool size in bytes
        self.max_pool_size = int(config.pool_size_gb * 1024**3)
        
        # Initialize pool with pre-allocated blocks
        if config.enable_preallocation:
            self._preallocate_blocks()
    
    def allocate(
        self,
        size_bytes: int,
        dtype: np.dtype = np.float32,
        fill_value: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> Tuple[str, Union[np.ndarray, jnp.ndarray]]:
        """
        Allocate memory block from pool.
        
        Args:
            size_bytes: Size of memory block in bytes
            dtype: Data type for the array
            fill_value: Optional fill value for initialization
            tags: Optional tags for the block
            
        Returns:
            Tuple of (block_id, memory_array)
        """
        with self._lock:
            # Find suitable block
            block_id = self._find_suitable_block(size_bytes)
            
            if block_id is None:
                # Create new block
                block_id = self._create_new_block(size_bytes, dtype, fill_value, tags)
            else:
                # Reuse existing block
                block = self.blocks[block_id]
                if fill_value is not None:
                    block.data.fill(fill_value)
                block.tags = tags or []
            
            # Mark as allocated
            block = self.blocks[block_id]
            block.ref_count += 1
            block.last_accessed = time.time()
            self.allocated_blocks[block_id] = block
            
            # Update statistics
            self.allocated_bytes += block.size_bytes
            self.peak_usage_bytes = max(self.peak_usage_bytes, self.allocated_bytes)
            self.allocation_count += 1
            
            return block_id, block.data
    
    def deallocate(self, block_id: str) -> bool:
        """
        Deallocate memory block back to pool.
        
        Args:
            block_id: ID of block to deallocate
            
        Returns:
            True if successfully deallocated
        """
        with self._lock:
            if block_id not in self.allocated_blocks:
                return False
            
            block = self.allocated_blocks[block_id]
            block.ref_count -= 1
            
            if block.ref_count <= 0:
                # Return to free pool
                size_class = self._get_size_class(block.size_bytes)
                if size_class not in self.free_blocks:
                    self.free_blocks[size_class] = []
                
                self.free_blocks[size_class].append(block_id)
                
                # Update statistics
                self.allocated_bytes -= block.size_bytes
                self.deallocation_count += 1
                
                # Remove from allocated blocks
                del self.allocated_blocks[block_id]
            
            return True
    
    def get_block_info(self, block_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a memory block."""
        with self._lock:
            if block_id in self.blocks:
                block = self.blocks[block_id]
                return {
                    'id': block.id,
                    'size_bytes': block.size_bytes,
                    'size_mb': block.size_bytes / (1024**2),
                    'allocated_time': block.allocated_time,
                    'last_accessed': block.last_accessed,
                    'ref_count': block.ref_count,
                    'is_locked': block.is_locked,
                    'tags': block.tags,
                    'dtype': str(block.data.dtype),
                    'shape': block.data.shape,
                }
            return None
    
    def cleanup(self, force: bool = False) -> int:
        """
        Clean up unused memory blocks.
        
        Args:
            force: Force cleanup regardless of thresholds
            
        Returns:
            Number of blocks cleaned up
        """
        with self._lock:
            cleaned_count = 0
            current_time = time.time()
            
            # Determine if cleanup is needed
            usage_ratio = self.allocated_bytes / self.max_pool_size
            
            if not force and usage_ratio < self.config.gc_threshold:
                return 0
            
            # Find blocks to clean up
            blocks_to_remove = []
            
            for size_class, block_ids in self.free_blocks.items():
                for block_id in block_ids[:]:  # Copy list to avoid modification during iteration
                    block = self.blocks[block_id]
                    
                    # Clean up old unused blocks
                    if (current_time - block.last_accessed) > 300:  # 5 minutes
                        blocks_to_remove.append(block_id)
                        block_ids.remove(block_id)
            
            # Remove blocks
            for block_id in blocks_to_remove:
                if block_id in self.blocks:
                    block = self.blocks[block_id]
                    self.total_size_bytes -= block.size_bytes
                    del self.blocks[block_id]
                    cleaned_count += 1
                    
                    # Clear the data explicitly
                    del block.data
            
            # Force garbage collection if needed
            if cleaned_count > 0:
                gc.collect()
            
            self.logger.debug(f"Cleaned up {cleaned_count} memory blocks")
            return cleaned_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            utilization = self.allocated_bytes / self.max_pool_size if self.max_pool_size > 0 else 0
            
            return {
                'total_blocks': len(self.blocks),
                'allocated_blocks': len(self.allocated_blocks),
                'free_blocks': sum(len(ids) for ids in self.free_blocks.values()),
                'total_size_mb': self.total_size_bytes / (1024**2),
                'allocated_mb': self.allocated_bytes / (1024**2),
                'peak_usage_mb': self.peak_usage_bytes / (1024**2),
                'max_pool_size_mb': self.max_pool_size / (1024**2),
                'utilization': utilization,
                'allocation_count': self.allocation_count,
                'deallocation_count': self.deallocation_count,
                'efficiency': (self.deallocation_count / self.allocation_count) if self.allocation_count > 0 else 0,
            }
    
    def _find_suitable_block(self, size_bytes: int) -> Optional[str]:
        """Find a suitable free block for the requested size."""
        size_class = self._get_size_class(size_bytes)
        
        # Look for exact size match first
        if size_class in self.free_blocks and self.free_blocks[size_class]:
            return self.free_blocks[size_class].pop()
        
        # Look for larger blocks that can be used
        for larger_size in sorted(self.free_blocks.keys()):
            if larger_size >= size_class and self.free_blocks[larger_size]:
                return self.free_blocks[larger_size].pop()
        
        return None
    
    def _create_new_block(
        self,
        size_bytes: int,
        dtype: np.dtype,
        fill_value: Optional[float],
        tags: Optional[List[str]]
    ) -> str:
        """Create a new memory block."""
        # Check if we have space in the pool
        if self.total_size_bytes + size_bytes > self.max_pool_size:
            # Try to clean up space
            self.cleanup(force=True)
            
            if self.total_size_bytes + size_bytes > self.max_pool_size:
                raise MemoryError(f"Cannot allocate {size_bytes} bytes: pool size exceeded")
        
        # Calculate array shape
        element_size = np.dtype(dtype).itemsize
        array_size = size_bytes // element_size
        
        # Create array
        if fill_value is not None:
            data = np.full(array_size, fill_value, dtype=dtype)
        elif self.config.zero_fill_blocks:
            data = np.zeros(array_size, dtype=dtype)
        else:
            data = np.empty(array_size, dtype=dtype)
        
        # Create block
        block_id = f"block_{int(time.time() * 1000000)}"  # Microsecond timestamp
        current_time = time.time()
        
        block = MemoryBlock(
            id=block_id,
            data=data,
            size_bytes=size_bytes,
            allocated_time=current_time,
            last_accessed=current_time,
            tags=tags or []
        )
        
        self.blocks[block_id] = block
        self.total_size_bytes += size_bytes
        
        return block_id
    
    def _get_size_class(self, size_bytes: int) -> int:
        """Get size class for memory block."""
        # Round up to nearest power of 2 for efficient allocation
        size_mb = size_bytes / (1024**2)
        return int(2 ** np.ceil(np.log2(max(size_mb, 1))))
    
    def _preallocate_blocks(self) -> None:
        """Pre-allocate common block sizes."""
        common_sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
        for size_mb in common_sizes_mb:
            if size_mb <= self.config.max_block_size_mb:
                try:
                    size_bytes = int(size_mb * 1024**2)
                    block_id = self._create_new_block(size_bytes, np.float32, None, ["preallocated"])
                    
                    # Add to free pool immediately
                    size_class = self._get_size_class(size_bytes)
                    if size_class not in self.free_blocks:
                        self.free_blocks[size_class] = []
                    self.free_blocks[size_class].append(block_id)
                    
                except MemoryError:
                    break  # Stop if we run out of space


class MemoryMappedArray:
    """Memory-mapped array for large data processing."""
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        temp_dir: Optional[str] = None,
        fill_value: Optional[float] = None
    ):
        self.shape = shape
        self.dtype = dtype
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(
            dir=self.temp_dir, 
            delete=False,
            prefix="qem_mmap_"
        )
        
        # Calculate size
        self.size_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        
        # Create memory-mapped array
        self.temp_file.seek(self.size_bytes - 1)
        self.temp_file.write(b'\0')
        self.temp_file.flush()
        
        self.mmap = mmap.mmap(self.temp_file.fileno(), 0)
        self.array = np.frombuffer(self.mmap, dtype=dtype).reshape(shape)
        
        # Initialize with fill value if provided
        if fill_value is not None:
            self.array.fill(fill_value)
    
    def __del__(self):
        """Clean up memory-mapped file."""
        try:
            if hasattr(self, 'mmap'):
                self.mmap.close()
            if hasattr(self, 'temp_file'):
                self.temp_file.close()
                os.unlink(self.temp_file.name)
        except:
            pass  # Ignore cleanup errors
    
    def close(self):
        """Explicitly close and clean up."""
        self.__del__()


class MemoryManager:
    """
    Comprehensive memory manager for quantum computations.
    
    This class provides intelligent memory allocation, pooling, cleanup,
    and monitoring for quantum error mitigation computations involving
    large state vectors and datasets.
    
    Features:
    - Memory pooling for efficient allocation/deallocation
    - Memory-mapped files for very large arrays
    - Automatic memory cleanup and garbage collection
    - Memory usage monitoring and alerts
    - Streaming processing for large datasets
    - Thread-safe operations
    - Performance profiling and optimization
    
    Example:
        >>> memory_manager = MemoryManager()
        >>> 
        >>> # Allocate memory for quantum state
        >>> state_id, state_array = memory_manager.allocate_state_vector(num_qubits=10)
        >>> 
        >>> # Process quantum circuit
        >>> result = process_circuit(state_array)
        >>> 
        >>> # Deallocate when done
        >>> memory_manager.deallocate(state_id)
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory manager.
        
        Args:
            config: Memory management configuration
        """
        self.config = config or MemoryConfig()
        self.logger = get_logger(__name__)
        
        # Initialize memory pool
        self.pool = MemoryPool(self.config)
        
        # Memory monitoring
        self.process = psutil.Process()
        self.memory_history: List[Dict[str, Any]] = []
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Memory-mapped arrays
        self.mapped_arrays: Dict[str, MemoryMappedArray] = {}
        
        # Cleanup management
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start background processes
        if self.config.enable_monitoring:
            self._start_monitoring()
        
        if self.config.cleanup_strategy == CleanupStrategy.PERIODIC:
            self._start_periodic_cleanup()
        
        self.logger.info("MemoryManager initialized")
    
    def allocate_state_vector(
        self,
        num_qubits: int,
        dtype: np.dtype = np.complex64,
        initial_state: Optional[str] = "zero"
    ) -> Tuple[str, Union[np.ndarray, jnp.ndarray]]:
        """
        Allocate memory for quantum state vector.
        
        Args:
            num_qubits: Number of qubits
            dtype: Data type for state vector
            initial_state: Initial state ("zero", "plus", "random", or None)
            
        Returns:
            Tuple of (allocation_id, state_vector)
        """
        # Calculate size
        hilbert_size = 2 ** num_qubits
        size_bytes = hilbert_size * np.dtype(dtype).itemsize
        
        # Determine allocation strategy
        size_mb = size_bytes / (1024**2)
        
        if size_mb > self.config.mmap_threshold_mb and self.config.enable_memory_mapping:
            # Use memory mapping for large state vectors
            return self._allocate_mapped_state(num_qubits, dtype, initial_state)
        else:
            # Use memory pool
            return self._allocate_pooled_state(num_qubits, dtype, initial_state, size_bytes)
    
    def allocate_array(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        fill_value: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> Tuple[str, Union[np.ndarray, jnp.ndarray]]:
        """
        Allocate general array with specified shape and type.
        
        Args:
            shape: Array shape
            dtype: Data type
            fill_value: Optional fill value
            tags: Optional tags for memory tracking
            
        Returns:
            Tuple of (allocation_id, array)
        """
        size_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        allocation_id, data = self.pool.allocate(size_bytes, dtype, fill_value, tags)
        
        # Reshape to requested shape
        reshaped_data = data[:np.prod(shape)].reshape(shape)
        
        return allocation_id, reshaped_data
    
    def deallocate(self, allocation_id: str) -> bool:
        """
        Deallocate memory allocation.
        
        Args:
            allocation_id: ID of allocation to deallocate
            
        Returns:
            True if successfully deallocated
        """
        with self._lock:
            # Check if it's a memory-mapped array
            if allocation_id in self.mapped_arrays:
                mapped_array = self.mapped_arrays[allocation_id]
                mapped_array.close()
                del self.mapped_arrays[allocation_id]
                return True
            
            # Otherwise deallocate from pool
            return self.pool.deallocate(allocation_id)
    
    def create_streaming_processor(
        self,
        total_size: int,
        chunk_size: Optional[int] = None,
        dtype: np.dtype = np.float32
    ) -> 'StreamingProcessor':
        """
        Create streaming processor for large datasets.
        
        Args:
            total_size: Total size of data to process
            chunk_size: Size of each chunk (auto-calculated if None)
            dtype: Data type
            
        Returns:
            StreamingProcessor instance
        """
        if chunk_size is None:
            chunk_size = int(self.config.stream_chunk_size_mb * 1024**2 / np.dtype(dtype).itemsize)
        
        return StreamingProcessor(self, total_size, chunk_size, dtype)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        process_memory = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        pool_stats = self.pool.get_stats()
        
        return {
            'process_memory': {
                'rss_mb': process_memory.rss / (1024**2),
                'vms_mb': process_memory.vms / (1024**2),
                'percent': self.process.memory_percent(),
            },
            'system_memory': {
                'total_gb': system_memory.total / (1024**3),
                'available_gb': system_memory.available / (1024**3),
                'used_gb': system_memory.used / (1024**3),
                'percent': system_memory.percent,
            },
            'pool_stats': pool_stats,
            'mapped_arrays': {
                'count': len(self.mapped_arrays),
                'total_size_mb': sum(
                    arr.size_bytes / (1024**2) 
                    for arr in self.mapped_arrays.values()
                ),
            },
            'config': {
                'pool_size_gb': self.config.pool_size_gb,
                'mmap_threshold_mb': self.config.mmap_threshold_mb,
                'gc_threshold': self.config.gc_threshold,
            }
        }
    
    def cleanup(self, force: bool = False) -> Dict[str, int]:
        """
        Perform memory cleanup.
        
        Args:
            force: Force cleanup regardless of thresholds
            
        Returns:
            Dictionary with cleanup statistics
        """
        with self._lock:
            pool_cleaned = self.pool.cleanup(force)
            
            # Clean up old mapped arrays
            mapped_cleaned = 0
            current_time = time.time()
            old_mappings = []
            
            for array_id, mapped_array in self.mapped_arrays.items():
                # This is a simple heuristic - in practice you'd track access times
                old_mappings.append(array_id)
            
            # Force garbage collection
            if pool_cleaned > 0 or mapped_cleaned > 0:
                gc.collect()
            
            return {
                'pool_blocks_cleaned': pool_cleaned,
                'mapped_arrays_cleaned': mapped_cleaned,
                'total_cleaned': pool_cleaned + mapped_cleaned,
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory performance report."""
        memory_usage = self.get_memory_usage()
        
        # Analyze memory history for trends
        history_analysis = {}
        if self.memory_history:
            recent_history = self.memory_history[-10:]  # Last 10 samples
            
            rss_values = [entry['process_memory']['rss_mb'] for entry in recent_history]
            history_analysis = {
                'memory_trend': 'increasing' if rss_values[-1] > rss_values[0] else 'stable',
                'average_rss_mb': np.mean(rss_values),
                'peak_rss_mb': np.max(rss_values),
                'memory_volatility': np.std(rss_values),
            }
        
        return {
            'current_usage': memory_usage,
            'history_analysis': history_analysis,
            'recommendations': self._generate_memory_recommendations(memory_usage),
            'monitoring_active': self._monitoring_active,
            'cleanup_stats': {
                'strategy': self.config.cleanup_strategy.value,
                'last_cleanup': getattr(self, '_last_cleanup_time', None),
            }
        }
    
    def _allocate_pooled_state(
        self,
        num_qubits: int,
        dtype: np.dtype,
        initial_state: Optional[str],
        size_bytes: int
    ) -> Tuple[str, Union[np.ndarray, jnp.ndarray]]:
        """Allocate state vector using memory pool."""
        # Determine fill value based on initial state
        fill_value = None
        if initial_state == "zero":
            fill_value = 0.0
        
        allocation_id, data = self.pool.allocate(
            size_bytes, dtype, fill_value, tags=["state_vector", f"qubits_{num_qubits}"]
        )
        
        # Initialize state
        hilbert_size = 2 ** num_qubits
        state_vector = data[:hilbert_size]
        
        if initial_state == "zero":
            state_vector[0] = 1.0  # |00...0⟩ state
        elif initial_state == "plus":
            state_vector.fill(1.0 / np.sqrt(hilbert_size))  # |++...+⟩ state
        elif initial_state == "random":
            state_vector[:] = np.random.random(hilbert_size) + 1j * np.random.random(hilbert_size)
            state_vector /= np.linalg.norm(state_vector)
        
        return allocation_id, state_vector
    
    def _allocate_mapped_state(
        self,
        num_qubits: int,
        dtype: np.dtype,
        initial_state: Optional[str]
    ) -> Tuple[str, np.ndarray]:
        """Allocate state vector using memory mapping."""
        hilbert_size = 2 ** num_qubits
        
        mapped_array = MemoryMappedArray(
            shape=(hilbert_size,),
            dtype=dtype,
            temp_dir=self.config.temp_dir
        )
        
        allocation_id = f"mapped_{int(time.time() * 1000000)}"
        self.mapped_arrays[allocation_id] = mapped_array
        
        # Initialize state
        if initial_state == "zero":
            mapped_array.array.fill(0.0)
            mapped_array.array[0] = 1.0
        elif initial_state == "plus":
            mapped_array.array.fill(1.0 / np.sqrt(hilbert_size))
        elif initial_state == "random":
            mapped_array.array[:] = np.random.random(hilbert_size) + 1j * np.random.random(hilbert_size)
            mapped_array.array /= np.linalg.norm(mapped_array.array)
        
        return allocation_id, mapped_array.array
    
    def _start_monitoring(self) -> None:
        """Start memory monitoring thread."""
        def monitor_loop():
            while self._monitoring_active:
                try:
                    usage = self.get_memory_usage()
                    self.memory_history.append({
                        'timestamp': time.time(),
                        **usage
                    })
                    
                    # Maintain history size
                    if len(self.memory_history) > 100:
                        self.memory_history.pop(0)
                    
                    # Check for memory warnings
                    if usage['system_memory']['percent'] > self.config.memory_warning_threshold * 100:
                        self.logger.warning(
                            f"High system memory usage: {usage['system_memory']['percent']:.1f}%"
                        )
                    
                    time.sleep(self.config.monitor_interval)
                    
                except Exception as e:
                    self.logger.error(f"Memory monitoring error: {e}")
                    time.sleep(self.config.monitor_interval)
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _start_periodic_cleanup(self) -> None:
        """Start periodic cleanup thread."""
        def cleanup_loop():
            while not self._stop_cleanup.wait(self.config.cleanup_interval):
                try:
                    self.cleanup()
                except Exception as e:
                    self.logger.error(f"Periodic cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _generate_memory_recommendations(self, usage: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # Check system memory usage
        if usage['system_memory']['percent'] > 90:
            recommendations.append("System memory usage is very high - consider reducing batch sizes")
        
        # Check pool efficiency
        pool_util = usage['pool_stats']['utilization']
        if pool_util > 0.9:
            recommendations.append("Memory pool utilization is high - consider increasing pool size")
        elif pool_util < 0.2:
            recommendations.append("Memory pool utilization is low - consider decreasing pool size")
        
        # Check pool efficiency
        pool_efficiency = usage['pool_stats']['efficiency']
        if pool_efficiency < 0.5:
            recommendations.append("Low memory pool efficiency - review allocation/deallocation patterns")
        
        # Check memory mapping usage
        if usage['mapped_arrays']['count'] > 0:
            recommendations.append("Using memory-mapped arrays for large data - consider SSD storage for better performance")
        
        if not recommendations:
            recommendations.append("Memory usage is optimal")
        
        return recommendations


class StreamingProcessor:
    """Streaming processor for large datasets."""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        total_size: int,
        chunk_size: int,
        dtype: np.dtype
    ):
        self.memory_manager = memory_manager
        self.total_size = total_size
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.current_position = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[str, np.ndarray]:
        if self.current_position >= self.total_size:
            raise StopIteration
        
        # Calculate actual chunk size
        remaining = self.total_size - self.current_position
        actual_chunk_size = min(self.chunk_size, remaining)
        
        # Allocate chunk
        allocation_id, chunk_data = self.memory_manager.allocate_array(
            shape=(actual_chunk_size,),
            dtype=self.dtype,
            tags=["streaming_chunk"]
        )
        
        self.current_position += actual_chunk_size
        
        return allocation_id, chunk_data


def create_memory_manager(
    pool_size_gb: float = 4.0,
    enable_memory_mapping: bool = True,
    cleanup_strategy: CleanupStrategy = CleanupStrategy.THRESHOLD,
    **config_kwargs
) -> MemoryManager:
    """
    Create a memory manager with specified configuration.
    
    Args:
        pool_size_gb: Size of memory pool in GB
        enable_memory_mapping: Enable memory-mapped files for large arrays
        cleanup_strategy: Memory cleanup strategy
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured MemoryManager instance
    """
    config = MemoryConfig(
        pool_size_gb=pool_size_gb,
        enable_memory_mapping=enable_memory_mapping,
        cleanup_strategy=cleanup_strategy,
        **config_kwargs
    )
    return MemoryManager(config)