"""Data layer for QEM-Bench results and metadata storage."""

from .models import (
    ExperimentResult,
    BenchmarkRun,
    CircuitMetadata,
    NoiseCharacterization,
    DeviceCalibration
)
# Repositories module not implemented yet
# from .repositories import (
#     ResultRepository,
#     BenchmarkRepository,
#     CircuitRepository,
#     NoiseRepository,
#     DeviceRepository
# )
from .storage import (
    StorageBackend,
    InMemoryStorageBackend,
    JSONStorageBackend,
    SQLiteStorageBackend
    # create_storage_backend - function not implemented
)
# Cache module not implemented yet
# from .cache import (
#     CacheManager,
#     InMemoryCache,
#     RedisCache,
#     create_cache_manager
# )

__all__ = [
    # Models
    "ExperimentResult",
    "BenchmarkRun", 
    "CircuitMetadata",
    "NoiseCharacterization",
    "DeviceCalibration",
    
    # Storage (implemented)
    "StorageBackend",
    "InMemoryStorageBackend",
    "JSONStorageBackend",
    "SQLiteStorageBackend",
    
    # Repositories (not yet implemented)
    # "ResultRepository",
    # "BenchmarkRepository",
    # "CircuitRepository", 
    # "NoiseRepository",
    # "DeviceRepository",
    
    # Cache (not yet implemented)
    # "CacheManager",
    # "InMemoryCache",
    # "RedisCache",
    # "create_cache_manager"
]