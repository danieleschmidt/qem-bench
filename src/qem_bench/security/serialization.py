"""
Secure Serialization for QEM-Bench

This module provides secure serialization and deserialization with:
- Schema validation to prevent malicious data
- Safe JSON parsing with depth limits
- Encrypted serialization for sensitive data
- Protection against deserialization attacks
- Type validation and sanitization
"""

import json
import pickle
import warnings
from typing import Any, Dict, List, Optional, Type, Union, Callable
from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum
import io
import sys

from ..errors import SecurityError, SerializationError, ValidationError
from .crypto_utils import CryptoUtils
from .input_sanitizer import InputSanitizer


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    ENCRYPTED_JSON = "encrypted_json"
    SAFE_PICKLE = "safe_pickle"  # Limited pickle with allowlist


@dataclass
class SerializationSchema:
    """Schema for validating serialized data."""
    name: str
    version: str = "1.0"
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, Type] = field(default_factory=dict)
    field_validators: Dict[str, Callable] = field(default_factory=dict)
    max_nesting_depth: int = 10
    max_string_length: int = 10000
    max_array_length: int = 1000
    allow_extra_fields: bool = False


class SecureSerializer:
    """
    Secure serializer for QEM-Bench data.
    
    Provides safe serialization and deserialization with schema validation,
    encryption support, and protection against malicious data.
    """
    
    def __init__(
        self,
        enable_encryption: bool = True,
        strict_validation: bool = True,
        max_data_size: int = 100 * 1024 * 1024  # 100MB
    ):
        """
        Initialize secure serializer.
        
        Args:
            enable_encryption: Whether to support encrypted serialization
            strict_validation: Whether to use strict schema validation
            max_data_size: Maximum size of data to serialize/deserialize
        """
        self.enable_encryption = enable_encryption
        self.strict_validation = strict_validation
        self.max_data_size = max_data_size
        
        # Initialize utilities
        self.crypto = CryptoUtils() if enable_encryption else None
        self.sanitizer = InputSanitizer()
        
        # Schema registry
        self.schemas: Dict[str, SerializationSchema] = {}
        
        # Safe types for pickle (if used)
        self.safe_pickle_types = {
            int, float, str, bool, bytes, list, tuple, dict, set, frozenset,
            type(None)
        }
        
        # Setup default schemas
        self._setup_default_schemas()
    
    def _setup_default_schemas(self):
        """Setup default schemas for common QEM-Bench objects."""
        # Circuit configuration schema
        circuit_schema = SerializationSchema(
            name="circuit_config",
            required_fields=["name", "num_qubits"],
            optional_fields=["depth", "backend", "shots", "parameters"],
            field_types={
                "name": str,
                "num_qubits": int,
                "depth": int,
                "backend": str,
                "shots": int,
                "parameters": dict
            },
            field_validators={
                "num_qubits": lambda x: x > 0 and x <= 100,
                "shots": lambda x: x > 0 and x <= 1000000,
                "depth": lambda x: x >= 0 and x <= 1000
            }
        )
        self.register_schema(circuit_schema)
        
        # Mitigation result schema
        result_schema = SerializationSchema(
            name="mitigation_result",
            required_fields=["method", "expectation_value", "timestamp"],
            optional_fields=["error", "metadata", "circuit_id"],
            field_types={
                "method": str,
                "expectation_value": (float, complex),
                "timestamp": str,
                "error": float,
                "metadata": dict,
                "circuit_id": str
            }
        )
        self.register_schema(result_schema)
        
        # Configuration schema
        config_schema = SerializationSchema(
            name="configuration",
            required_fields=["version"],
            optional_fields=["settings", "credentials", "backends"],
            field_types={
                "version": str,
                "settings": dict,
                "credentials": dict,
                "backends": dict
            },
            max_nesting_depth=5
        )
        self.register_schema(config_schema)
    
    def register_schema(self, schema: SerializationSchema):
        """Register a schema for validation."""
        self.schemas[schema.name] = schema
    
    def serialize(
        self,
        data: Any,
        format_type: SerializationFormat = SerializationFormat.JSON,
        schema_name: Optional[str] = None,
        compress: bool = False
    ) -> bytes:
        """
        Serialize data with optional schema validation and encryption.
        
        Args:
            data: Data to serialize
            format_type: Serialization format
            schema_name: Optional schema for validation
            compress: Whether to compress the data
            
        Returns:
            Serialized data as bytes
        """
        # Check data size
        data_str = str(data)
        if len(data_str) > self.max_data_size:
            raise SerializationError(f"Data too large: {len(data_str)} > {self.max_data_size}")
        
        # Validate against schema if provided
        if schema_name and schema_name in self.schemas:
            self._validate_data(data, self.schemas[schema_name])
        
        # Serialize based on format
        if format_type == SerializationFormat.JSON:
            return self._serialize_json(data, compress)
        elif format_type == SerializationFormat.ENCRYPTED_JSON:
            return self._serialize_encrypted_json(data, compress)
        elif format_type == SerializationFormat.SAFE_PICKLE:
            return self._serialize_safe_pickle(data, compress)
        else:
            raise SerializationError(f"Unsupported format: {format_type}")
    
    def deserialize(
        self,
        data: bytes,
        format_type: SerializationFormat = SerializationFormat.JSON,
        schema_name: Optional[str] = None,
        expected_type: Optional[Type] = None
    ) -> Any:
        """
        Deserialize data with validation.
        
        Args:
            data: Serialized data
            format_type: Serialization format used
            schema_name: Optional schema for validation
            expected_type: Expected type of deserialized data
            
        Returns:
            Deserialized data
        """
        # Check data size
        if len(data) > self.max_data_size:
            raise SerializationError(f"Data too large: {len(data)} > {self.max_data_size}")
        
        # Deserialize based on format
        if format_type == SerializationFormat.JSON:
            result = self._deserialize_json(data)
        elif format_type == SerializationFormat.ENCRYPTED_JSON:
            result = self._deserialize_encrypted_json(data)
        elif format_type == SerializationFormat.SAFE_PICKLE:
            result = self._deserialize_safe_pickle(data)
        else:
            raise SerializationError(f"Unsupported format: {format_type}")
        
        # Type validation
        if expected_type and not isinstance(result, expected_type):
            raise SerializationError(f"Type mismatch: expected {expected_type}, got {type(result)}")
        
        # Schema validation
        if schema_name and schema_name in self.schemas:
            self._validate_data(result, self.schemas[schema_name])
        
        return result
    
    def _serialize_json(self, data: Any, compress: bool = False) -> bytes:
        """Serialize data as JSON."""
        try:
            # Handle special types
            data = self._prepare_for_json(data)
            
            # Serialize to JSON
            json_str = json.dumps(data, separators=(',', ':'), ensure_ascii=True)
            json_bytes = json_str.encode('utf-8')
            
            # Compress if requested
            if compress:
                json_bytes = self._compress_data(json_bytes)
            
            return json_bytes
            
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {e}")
    
    def _deserialize_json(self, data: bytes) -> Any:
        """Deserialize JSON data."""
        try:
            # Decompress if needed
            if self._is_compressed(data):
                data = self._decompress_data(data)
            
            # Parse JSON with safety checks
            json_str = data.decode('utf-8')
            
            # Check for suspicious patterns
            if len(json_str) > self.max_data_size:
                raise SerializationError("JSON string too large")
            
            # Parse with custom decoder for safety
            result = json.loads(json_str, object_hook=self._safe_json_object_hook)
            
            # Validate structure
            self._validate_json_structure(result)
            
            return result
            
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON: {e}")
        except Exception as e:
            raise SerializationError(f"JSON deserialization failed: {e}")
    
    def _serialize_encrypted_json(self, data: Any, compress: bool = False) -> bytes:
        """Serialize and encrypt data as JSON."""
        if not self.crypto:
            raise SerializationError("Encryption not available")
        
        # First serialize as JSON
        json_data = self._serialize_json(data, compress)
        
        # Then encrypt
        encrypted_data = self.crypto.encrypt(json_data)
        
        return encrypted_data
    
    def _deserialize_encrypted_json(self, data: bytes) -> Any:
        """Decrypt and deserialize JSON data."""
        if not self.crypto:
            raise SerializationError("Decryption not available")
        
        try:
            # Decrypt first
            decrypted_data = self.crypto.decrypt(data)
            
            # Then deserialize JSON
            return self._deserialize_json(decrypted_data)
            
        except Exception as e:
            raise SerializationError(f"Encrypted JSON deserialization failed: {e}")
    
    def _serialize_safe_pickle(self, data: Any, compress: bool = False) -> bytes:
        """Serialize data using safe pickle (limited types only)."""
        warnings.warn("Pickle serialization should be avoided when possible")
        
        try:
            # Validate that data contains only safe types
            self._validate_pickle_safety(data)
            
            # Serialize
            buffer = io.BytesIO()
            pickle.dump(data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_data = buffer.getvalue()
            
            # Compress if requested
            if compress:
                pickle_data = self._compress_data(pickle_data)
            
            return pickle_data
            
        except Exception as e:
            raise SerializationError(f"Pickle serialization failed: {e}")
    
    def _deserialize_safe_pickle(self, data: bytes) -> Any:
        """Deserialize pickle data with safety restrictions."""
        warnings.warn("Pickle deserialization should be avoided when possible")
        
        try:
            # Decompress if needed
            if self._is_compressed(data):
                data = self._decompress_data(data)
            
            # Create restricted unpickler
            buffer = io.BytesIO(data)
            unpickler = pickle.Unpickler(buffer)
            
            # Override find_class to restrict types
            unpickler.find_class = self._safe_pickle_find_class
            
            result = unpickler.load()
            
            # Final safety validation
            self._validate_pickle_safety(result)
            
            return result
            
        except Exception as e:
            raise SerializationError(f"Pickle deserialization failed: {e}")
    
    def _validate_data(self, data: Any, schema: SerializationSchema):
        """Validate data against a schema."""
        if not isinstance(data, dict):
            raise ValidationError("Data must be a dictionary for schema validation")
        
        # Check required fields
        for field in schema.required_fields:
            if field not in data:
                raise ValidationError(f"Required field missing: {field}")
        
        # Check field types and values
        for field, value in data.items():
            # Check if field is allowed
            if (field not in schema.required_fields and 
                field not in schema.optional_fields and 
                not schema.allow_extra_fields):
                if self.strict_validation:
                    raise ValidationError(f"Unknown field: {field}")
                else:
                    warnings.warn(f"Unknown field in data: {field}")
                    continue
            
            # Type validation
            if field in schema.field_types:
                expected_type = schema.field_types[field]
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        raise ValidationError(f"Field {field} has wrong type: {type(value)}")
                else:
                    if not isinstance(value, expected_type):
                        raise ValidationError(f"Field {field} has wrong type: {type(value)}")
            
            # Custom validation
            if field in schema.field_validators:
                validator = schema.field_validators[field]
                try:
                    if not validator(value):
                        raise ValidationError(f"Field {field} failed validation")
                except Exception as e:
                    raise ValidationError(f"Field {field} validation error: {e}")
            
            # String length validation
            if isinstance(value, str) and len(value) > schema.max_string_length:
                raise ValidationError(f"String field {field} too long: {len(value)}")
            
            # Array length validation
            if isinstance(value, (list, tuple)) and len(value) > schema.max_array_length:
                raise ValidationError(f"Array field {field} too long: {len(value)}")
        
        # Check nesting depth
        self._check_nesting_depth(data, schema.max_nesting_depth)
    
    def _check_nesting_depth(self, obj: Any, max_depth: int, current_depth: int = 0):
        """Check nesting depth to prevent stack overflow."""
        if current_depth > max_depth:
            raise ValidationError(f"Nesting depth exceeded: {current_depth} > {max_depth}")
        
        if isinstance(obj, dict):
            for value in obj.values():
                self._check_nesting_depth(value, max_depth, current_depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._check_nesting_depth(item, max_depth, current_depth + 1)
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization by handling special types."""
        if is_dataclass(data):
            return asdict(data)
        elif isinstance(data, Enum):
            return data.value
        elif isinstance(data, (set, frozenset)):
            return list(data)
        elif isinstance(data, bytes):
            return data.hex()
        elif isinstance(data, complex):
            return {"__complex__": True, "real": data.real, "imag": data.imag}
        elif isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._prepare_for_json(item) for item in data]
        else:
            return data
    
    def _safe_json_object_hook(self, obj: Dict[str, Any]) -> Any:
        """Safe object hook for JSON deserialization."""
        # Handle complex numbers
        if obj.get("__complex__"):
            return complex(obj["real"], obj["imag"])
        
        # Validate object size
        if len(str(obj)) > self.max_data_size // 10:  # Reasonable limit per object
            raise SerializationError("JSON object too large")
        
        return obj
    
    def _validate_json_structure(self, data: Any, depth: int = 0):
        """Validate JSON structure for safety."""
        if depth > 50:  # Prevent infinite recursion
            raise SerializationError("JSON structure too deeply nested")
        
        if isinstance(data, dict):
            if len(data) > 10000:  # Reasonable limit
                raise SerializationError("JSON object has too many keys")
            for key, value in data.items():
                if not isinstance(key, str):
                    raise SerializationError("JSON object keys must be strings")
                if len(key) > 1000:
                    raise SerializationError("JSON object key too long")
                self._validate_json_structure(value, depth + 1)
        elif isinstance(data, list):
            if len(data) > 100000:  # Reasonable limit
                raise SerializationError("JSON array too long")
            for item in data:
                self._validate_json_structure(item, depth + 1)
        elif isinstance(data, str):
            if len(data) > self.max_data_size // 100:
                raise SerializationError("JSON string too long")
    
    def _validate_pickle_safety(self, obj: Any, depth: int = 0):
        """Validate that an object is safe for pickle operations."""
        if depth > 20:  # Prevent infinite recursion
            raise SerializationError("Pickle object structure too deeply nested")
        
        obj_type = type(obj)
        
        # Check if type is in safe list
        if obj_type not in self.safe_pickle_types:
            # Allow some additional safe types
            if hasattr(obj, '__dict__') and not callable(obj):
                # Simple objects with __dict__ might be safe
                for value in obj.__dict__.values():
                    self._validate_pickle_safety(value, depth + 1)
            else:
                raise SerializationError(f"Unsafe type for pickle: {obj_type}")
        
        # Recursively check containers
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._validate_pickle_safety(key, depth + 1)
                self._validate_pickle_safety(value, depth + 1)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                self._validate_pickle_safety(item, depth + 1)
    
    def _safe_pickle_find_class(self, module: str, name: str):
        """Safe find_class method for pickle unpickling."""
        # Only allow built-in types and specific safe modules
        safe_modules = {
            'builtins': {'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set', 'frozenset'},
            'collections': {'defaultdict', 'OrderedDict', 'deque'},
            'datetime': {'datetime', 'date', 'time', 'timedelta'},
            'decimal': {'Decimal'},
            'fractions': {'Fraction'}
        }
        
        if module in safe_modules and name in safe_modules[module]:
            return getattr(sys.modules[module], name)
        
        raise SerializationError(f"Unsafe class for pickle: {module}.{name}")
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        import gzip
        return gzip.compress(data)
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress gzip data."""
        import gzip
        return gzip.decompress(data)
    
    def _is_compressed(self, data: bytes) -> bool:
        """Check if data is gzip compressed."""
        return data.startswith(b'\x1f\x8b')
    
    def get_schema(self, name: str) -> Optional[SerializationSchema]:
        """Get a registered schema by name."""
        return self.schemas.get(name)
    
    def list_schemas(self) -> List[str]:
        """List all registered schema names."""
        return list(self.schemas.keys())


# Global secure serializer instance
_global_serializer: Optional[SecureSerializer] = None


def get_secure_serializer() -> SecureSerializer:
    """Get the global secure serializer instance."""
    global _global_serializer
    if _global_serializer is None:
        _global_serializer = SecureSerializer()
    return _global_serializer


def set_secure_serializer(serializer: SecureSerializer):
    """Set the global secure serializer instance."""
    global _global_serializer
    _global_serializer = serializer


# Convenience functions
def serialize_circuit_config(config: Dict[str, Any], encrypt: bool = False) -> bytes:
    """Serialize circuit configuration with validation."""
    serializer = get_secure_serializer()
    format_type = SerializationFormat.ENCRYPTED_JSON if encrypt else SerializationFormat.JSON
    return serializer.serialize(config, format_type, "circuit_config")


def deserialize_circuit_config(data: bytes, encrypted: bool = False) -> Dict[str, Any]:
    """Deserialize circuit configuration with validation."""
    serializer = get_secure_serializer()
    format_type = SerializationFormat.ENCRYPTED_JSON if encrypted else SerializationFormat.JSON
    return serializer.deserialize(data, format_type, "circuit_config")


def serialize_result(result: Dict[str, Any], encrypt: bool = False) -> bytes:
    """Serialize mitigation result with validation."""
    serializer = get_secure_serializer()
    format_type = SerializationFormat.ENCRYPTED_JSON if encrypt else SerializationFormat.JSON
    return serializer.serialize(result, format_type, "mitigation_result")


def deserialize_result(data: bytes, encrypted: bool = False) -> Dict[str, Any]:
    """Deserialize mitigation result with validation."""
    serializer = get_secure_serializer()
    format_type = SerializationFormat.ENCRYPTED_JSON if encrypted else SerializationFormat.JSON
    return serializer.deserialize(data, format_type, "mitigation_result")