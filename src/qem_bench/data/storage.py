"""Storage backends for QEM-Bench data persistence."""

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type
from datetime import datetime
import threading
import uuid


class StorageBackend(ABC):
    """Abstract base class for data storage backends."""
    
    @abstractmethod
    def store(self, key: str, data: Dict[str, Any]) -> None:
        """Store data with the given key."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data by key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key. Returns True if deleted, False if not found."""
        pass
    
    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys, optionally filtered by prefix."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all data."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the storage backend."""
        pass


class JSONStorageBackend(StorageBackend):
    """
    JSON file-based storage backend.
    
    Stores each record as a separate JSON file in a directory structure.
    Good for development and small-scale deployments.
    """
    
    def __init__(self, base_path: Union[str, Path], create_dirs: bool = True):
        """
        Initialize JSON storage backend.
        
        Args:
            base_path: Base directory for storing JSON files
            create_dirs: Whether to create directories automatically
        """
        self.base_path = Path(base_path)
        self.create_dirs = create_dirs
        self._lock = threading.RLock()
        
        if create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)
    
    def store(self, key: str, data: Dict[str, Any]) -> None:
        """Store data as JSON file."""
        with self._lock:
            file_path = self._key_to_path(key)
            
            if self.create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            storage_data = {
                "stored_at": datetime.utcnow().isoformat(),
                "key": key,
                "data": data
            }
            
            with open(file_path, 'w') as f:
                json.dump(storage_data, f, indent=2, default=str)
    
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from JSON file."""
        with self._lock:
            file_path = self._key_to_path(key)
            
            if not file_path.exists():
                return None
            
            try:
                with open(file_path, 'r') as f:
                    storage_data = json.load(f)
                return storage_data.get("data")
            except (json.JSONDecodeError, IOError):
                return None
    
    def delete(self, key: str) -> bool:
        """Delete JSON file."""
        with self._lock:
            file_path = self._key_to_path(key)
            
            if file_path.exists():
                file_path.unlink()
                return True
            return False
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys by scanning directory structure."""
        with self._lock:
            keys = []
            
            if not self.base_path.exists():
                return keys
            
            for json_file in self.base_path.rglob("*.json"):
                # Convert file path back to key
                relative_path = json_file.relative_to(self.base_path)
                key = str(relative_path).replace('.json', '').replace('/', ':')
                
                if not prefix or key.startswith(prefix):
                    keys.append(key)
            
            return sorted(keys)
    
    def exists(self, key: str) -> bool:
        """Check if JSON file exists."""
        return self._key_to_path(key).exists()
    
    def clear(self) -> None:
        """Remove all JSON files."""
        with self._lock:
            for json_file in self.base_path.rglob("*.json"):
                json_file.unlink()
    
    def close(self) -> None:
        """Nothing to close for JSON backend."""
        pass
    
    def _key_to_path(self, key: str) -> Path:
        """Convert storage key to file path."""
        # Replace colons with directory separators
        path_parts = key.split(':')
        return self.base_path / Path(*path_parts).with_suffix('.json')


class SQLiteStorageBackend(StorageBackend):
    """
    SQLite database storage backend.
    
    Stores data in a SQLite database with JSON serialization.
    Good for production deployments requiring ACID properties.
    """
    
    def __init__(self, db_path: Union[str, Path], table_name: str = "qem_data"):
        """
        Initialize SQLite storage backend.
        
        Args:
            db_path: Path to SQLite database file
            table_name: Name of table to store data
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._lock = threading.RLock()
        
        # Create database and table
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        key TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index on stored_at for performance
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_stored_at 
                    ON {self.table_name}(stored_at)
                """)
                
                conn.commit()
            finally:
                conn.close()
    
    def store(self, key: str, data: Dict[str, Any]) -> None:
        """Store data in SQLite database."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                json_data = json.dumps(data, default=str)
                
                conn.execute(f"""
                    INSERT OR REPLACE INTO {self.table_name} 
                    (key, data, updated_at) 
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, json_data))
                
                conn.commit()
            finally:
                conn.close()
    
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from SQLite database."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(f"""
                    SELECT data FROM {self.table_name} WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
            except (json.JSONDecodeError, sqlite3.Error):
                return None
            finally:
                conn.close()
    
    def delete(self, key: str) -> bool:
        """Delete data from SQLite database."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(f"""
                    DELETE FROM {self.table_name} WHERE key = ?
                """, (key,))
                
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys from SQLite database."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                if prefix:
                    cursor = conn.execute(f"""
                        SELECT key FROM {self.table_name} 
                        WHERE key LIKE ? 
                        ORDER BY key
                    """, (f"{prefix}%",))
                else:
                    cursor = conn.execute(f"""
                        SELECT key FROM {self.table_name} 
                        ORDER BY key
                    """)
                
                return [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in SQLite database."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(f"""
                    SELECT 1 FROM {self.table_name} WHERE key = ? LIMIT 1
                """, (key,))
                
                return cursor.fetchone() is not None
            finally:
                conn.close()
    
    def clear(self) -> None:
        """Clear all data from SQLite database."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute(f"DELETE FROM {self.table_name}")
                conn.commit()
            finally:
                conn.close()
    
    def close(self) -> None:
        """Close SQLite database connections."""
        # SQLite connections are closed after each operation
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute(f"""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(stored_at) as earliest_record,
                        MAX(stored_at) as latest_record
                    FROM {self.table_name}
                """)
                
                row = cursor.fetchone()
                return {
                    "total_records": row[0],
                    "earliest_record": row[1],
                    "latest_record": row[2],
                    "database_size": self.db_path.stat().st_size if self.db_path.exists() else 0
                }
            finally:
                conn.close()


class InMemoryStorageBackend(StorageBackend):
    """
    In-memory storage backend.
    
    Stores data in memory using dictionaries.
    Good for testing and temporary storage.
    """
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._data: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def store(self, key: str, data: Dict[str, Any]) -> None:
        """Store data in memory."""
        with self._lock:
            self._data[key] = data.copy()
            self._metadata[key] = {
                "stored_at": datetime.utcnow().isoformat(),
                "key": key
            }
    
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from memory."""
        with self._lock:
            return self._data.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete data from memory."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                if key in self._metadata:
                    del self._metadata[key]
                return True
            return False
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys in memory."""
        with self._lock:
            keys = list(self._data.keys())
            if prefix:
                keys = [k for k in keys if k.startswith(prefix)]
            return sorted(keys)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory."""
        with self._lock:
            return key in self._data
    
    def clear(self) -> None:
        """Clear all data from memory."""
        with self._lock:
            self._data.clear()
            self._metadata.clear()
    
    def close(self) -> None:
        """Nothing to close for in-memory backend."""
        pass


def create_storage_backend(
    backend_type: str,
    **kwargs
) -> StorageBackend:
    """
    Factory function to create storage backends.
    
    Args:
        backend_type: Type of backend ("json", "sqlite", "memory")
        **kwargs: Backend-specific configuration
        
    Returns:
        Configured storage backend
    """
    if backend_type == "json":
        base_path = kwargs.get("base_path", "./data")
        create_dirs = kwargs.get("create_dirs", True)
        return JSONStorageBackend(base_path, create_dirs)
    
    elif backend_type == "sqlite":
        db_path = kwargs.get("db_path", "./qem_bench.db")
        table_name = kwargs.get("table_name", "qem_data")
        return SQLiteStorageBackend(db_path, table_name)
    
    elif backend_type == "memory":
        return InMemoryStorageBackend()
    
    else:
        raise ValueError(f"Unknown storage backend type: {backend_type}")