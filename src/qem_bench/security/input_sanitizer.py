"""
Input Sanitization and Validation for QEM-Bench

This module provides comprehensive input validation and sanitization to prevent:
- Code injection attacks
- Path traversal attacks
- SQL injection (for future database features)
- Cross-site scripting (XSS) in web interfaces
- Buffer overflow attempts
- Malformed data attacks
- Resource exhaustion through large inputs
"""

import re
import html
import json
import math
import warnings
from typing import Any, Dict, List, Optional, Union, Callable, Pattern
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import urllib.parse

from ..errors import SecurityError, ValidationError, InputSanitizationError


class InputType(Enum):
    """Types of inputs that can be sanitized."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    PATH = "path"
    URL = "url"
    EMAIL = "email"
    JSON = "json"
    CIRCUIT_NAME = "circuit_name"
    BACKEND_NAME = "backend_name"
    PARAMETER_NAME = "parameter_name"
    FILENAME = "filename"


@dataclass
class SanitizationRule:
    """Rule for sanitizing a specific type of input."""
    input_type: InputType
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    allowed_chars: Optional[Pattern] = None
    forbidden_chars: Optional[Pattern] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    strip_whitespace: bool = True
    escape_html: bool = False
    normalize_case: Optional[str] = None  # 'lower', 'upper', or None


class InputSanitizer:
    """
    Comprehensive input sanitizer for QEM-Bench.
    
    Provides validation and sanitization for all types of user inputs
    to prevent security vulnerabilities and ensure data integrity.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize input sanitizer.
        
        Args:
            strict_mode: Whether to use strict validation rules
        """
        self.strict_mode = strict_mode
        self.rules = self._setup_default_rules()
        
        # Common regex patterns
        self.patterns = {
            'alphanumeric': re.compile(r'^[a-zA-Z0-9_-]+$'),
            'alphanumeric_spaces': re.compile(r'^[a-zA-Z0-9_\s-]+$'),
            'safe_filename': re.compile(r'^[a-zA-Z0-9._-]+$'),
            'circuit_name': re.compile(r'^[a-zA-Z][a-zA-Z0-9._-]*$'),
            'backend_name': re.compile(r'^[a-zA-Z][a-zA-Z0-9._-]*$'),
            'parameter_name': re.compile(r'^[a-zA-Z_][a-zA-Z0-9._]*$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'dangerous_chars': re.compile(r'[<>"\';()&]'),
            'control_chars': re.compile(r'[\x00-\x1f\x7f]'),
            'path_traversal': re.compile(r'\.\.[/\\]'),
            'script_tags': re.compile(r'<script.*?</script>', re.IGNORECASE | re.DOTALL),
            'sql_injection': re.compile(r'(union|select|insert|update|delete|drop|create|alter|exec)', re.IGNORECASE)
        }
    
    def _setup_default_rules(self) -> Dict[InputType, SanitizationRule]:
        """Setup default sanitization rules."""
        rules = {}
        
        # String sanitization
        rules[InputType.STRING] = SanitizationRule(
            input_type=InputType.STRING,
            max_length=10000,
            min_length=0,
            forbidden_chars=re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'),  # Control characters
            strip_whitespace=True,
            escape_html=False
        )
        
        # Integer sanitization
        rules[InputType.INTEGER] = SanitizationRule(
            input_type=InputType.INTEGER,
            custom_validator=self._validate_integer
        )
        
        # Float sanitization
        rules[InputType.FLOAT] = SanitizationRule(
            input_type=InputType.FLOAT,
            custom_validator=self._validate_float
        )
        
        # Boolean sanitization
        rules[InputType.BOOLEAN] = SanitizationRule(
            input_type=InputType.BOOLEAN,
            custom_validator=self._validate_boolean
        )
        
        # Path sanitization
        rules[InputType.PATH] = SanitizationRule(
            input_type=InputType.PATH,
            max_length=4096,
            custom_validator=self._validate_path
        )
        
        # URL sanitization
        rules[InputType.URL] = SanitizationRule(
            input_type=InputType.URL,
            max_length=2048,
            custom_validator=self._validate_url
        )
        
        # Email sanitization
        rules[InputType.EMAIL] = SanitizationRule(
            input_type=InputType.EMAIL,
            max_length=254,
            allowed_chars=re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            normalize_case='lower'
        )
        
        # JSON sanitization
        rules[InputType.JSON] = SanitizationRule(
            input_type=InputType.JSON,
            max_length=1000000,  # 1MB
            custom_validator=self._validate_json
        )
        
        # Circuit name sanitization
        rules[InputType.CIRCUIT_NAME] = SanitizationRule(
            input_type=InputType.CIRCUIT_NAME,
            max_length=100,
            min_length=1,
            allowed_chars=re.compile(r'^[a-zA-Z][a-zA-Z0-9._-]*$'),
            strip_whitespace=True
        )
        
        # Backend name sanitization
        rules[InputType.BACKEND_NAME] = SanitizationRule(
            input_type=InputType.BACKEND_NAME,
            max_length=50,
            min_length=1,
            allowed_chars=re.compile(r'^[a-zA-Z][a-zA-Z0-9._-]*$'),
            strip_whitespace=True,
            normalize_case='lower'
        )
        
        # Parameter name sanitization
        rules[InputType.PARAMETER_NAME] = SanitizationRule(
            input_type=InputType.PARAMETER_NAME,
            max_length=100,
            min_length=1,
            allowed_chars=re.compile(r'^[a-zA-Z_][a-zA-Z0-9._]*$'),
            strip_whitespace=True
        )
        
        # Filename sanitization
        rules[InputType.FILENAME] = SanitizationRule(
            input_type=InputType.FILENAME,
            max_length=255,
            min_length=1,
            allowed_chars=re.compile(r'^[a-zA-Z0-9._-]+$'),
            strip_whitespace=True
        )
        
        return rules
    
    def sanitize(self, value: Any, input_type: InputType, **kwargs) -> Any:
        """
        Sanitize an input value according to its type.
        
        Args:
            value: The value to sanitize
            input_type: Type of input being sanitized
            **kwargs: Additional sanitization parameters
            
        Returns:
            Sanitized value
            
        Raises:
            InputSanitizationError: If sanitization fails
        """
        if value is None:
            return None
        
        try:
            rule = self.rules.get(input_type)
            if not rule:
                raise InputSanitizationError(f"No sanitization rule for type: {input_type}")
            
            # Convert to string for most operations
            if input_type not in [InputType.INTEGER, InputType.FLOAT, InputType.BOOLEAN]:
                value = str(value)
            
            # Apply sanitization steps
            sanitized_value = self._apply_sanitization_rule(value, rule, **kwargs)
            
            return sanitized_value
            
        except Exception as e:
            if isinstance(e, InputSanitizationError):
                raise
            raise InputSanitizationError(f"Failed to sanitize {input_type.value}: {e}")
    
    def _apply_sanitization_rule(self, value: Any, rule: SanitizationRule, **kwargs) -> Any:
        """Apply a sanitization rule to a value."""
        # Handle non-string types first
        if rule.input_type in [InputType.INTEGER, InputType.FLOAT, InputType.BOOLEAN]:
            if rule.custom_validator:
                return rule.custom_validator(value)
            return value
        
        # String-based sanitization
        if not isinstance(value, str):
            value = str(value)
        
        # Strip whitespace
        if rule.strip_whitespace:
            value = value.strip()
        
        # Length validation
        if rule.max_length and len(value) > rule.max_length:
            if self.strict_mode:
                raise InputSanitizationError(
                    f"Input too long: {len(value)} > {rule.max_length}"
                )
            value = value[:rule.max_length]
        
        if rule.min_length and len(value) < rule.min_length:
            raise InputSanitizationError(
                f"Input too short: {len(value)} < {rule.min_length}"
            )
        
        # Check for dangerous patterns
        self._check_dangerous_patterns(value)
        
        # Character validation
        if rule.forbidden_chars and rule.forbidden_chars.search(value):
            raise InputSanitizationError("Input contains forbidden characters")
        
        if rule.allowed_chars and not rule.allowed_chars.match(value):
            raise InputSanitizationError("Input contains invalid characters")
        
        # Value validation
        if rule.allowed_values and value not in rule.allowed_values:
            raise InputSanitizationError(f"Invalid value. Allowed: {rule.allowed_values}")
        
        # HTML escaping
        if rule.escape_html:
            value = html.escape(value)
        
        # Case normalization
        if rule.normalize_case == 'lower':
            value = value.lower()
        elif rule.normalize_case == 'upper':
            value = value.upper()
        
        # Custom validation
        if rule.custom_validator:
            value = rule.custom_validator(value)
        
        return value
    
    def _check_dangerous_patterns(self, value: str):
        """Check for dangerous patterns in input."""
        # Path traversal
        if self.patterns['path_traversal'].search(value):
            raise InputSanitizationError("Path traversal attempt detected")
        
        # Script tags
        if self.patterns['script_tags'].search(value):
            raise InputSanitizationError("Script tag detected")
        
        # SQL injection patterns (basic check)
        if self.patterns['sql_injection'].search(value):
            warnings.warn("Potential SQL injection pattern detected")
        
        # Control characters
        if self.patterns['control_chars'].search(value):
            raise InputSanitizationError("Control characters not allowed")
    
    def _validate_integer(self, value: Any) -> int:
        """Validate and convert to integer."""
        try:
            if isinstance(value, str):
                value = value.strip()
            int_value = int(value)
            
            # Check for reasonable bounds
            if abs(int_value) > 10**18:  # Prevent overflow issues
                raise InputSanitizationError("Integer value too large")
            
            return int_value
        except (ValueError, TypeError) as e:
            raise InputSanitizationError(f"Invalid integer: {value}")
    
    def _validate_float(self, value: Any) -> float:
        """Validate and convert to float."""
        try:
            if isinstance(value, str):
                value = value.strip()
            float_value = float(value)
            
            # Check for special values
            if math.isnan(float_value):
                raise InputSanitizationError("NaN values not allowed")
            
            if math.isinf(float_value):
                raise InputSanitizationError("Infinite values not allowed")
            
            return float_value
        except (ValueError, TypeError) as e:
            raise InputSanitizationError(f"Invalid float: {value}")
    
    def _validate_boolean(self, value: Any) -> bool:
        """Validate and convert to boolean."""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ['true', '1', 'yes', 'on']:
                return True
            elif value in ['false', '0', 'no', 'off']:
                return False
        
        if isinstance(value, (int, float)):
            return bool(value)
        
        raise InputSanitizationError(f"Invalid boolean value: {value}")
    
    def _validate_path(self, value: str) -> str:
        """Validate file system path."""
        value = value.strip()
        
        # Check for path traversal
        if '..' in value:
            raise InputSanitizationError("Path traversal not allowed")
        
        # Check for absolute paths (optional restriction)
        if self.strict_mode and Path(value).is_absolute():
            raise InputSanitizationError("Absolute paths not allowed in strict mode")
        
        # Check for null bytes
        if '\x00' in value:
            raise InputSanitizationError("Null bytes in path not allowed")
        
        # Basic path validation
        try:
            Path(value)
        except Exception as e:
            raise InputSanitizationError(f"Invalid path: {e}")
        
        return value
    
    def _validate_url(self, value: str) -> str:
        """Validate URL."""
        value = value.strip()
        
        try:
            parsed = urllib.parse.urlparse(value)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                raise InputSanitizationError("Only HTTP/HTTPS URLs allowed")
            
            # Check for localhost/private IPs in strict mode
            if self.strict_mode:
                hostname = parsed.hostname
                if hostname and (
                    hostname in ['localhost', '127.0.0.1', '::1'] or
                    hostname.startswith('192.168.') or
                    hostname.startswith('10.') or
                    hostname.startswith('172.')
                ):
                    raise InputSanitizationError("Private network URLs not allowed")
            
            return value
        except Exception as e:
            raise InputSanitizationError(f"Invalid URL: {e}")
    
    def _validate_json(self, value: str) -> Dict[str, Any]:
        """Validate JSON string."""
        value = value.strip()
        
        try:
            data = json.loads(value)
            
            # Check nesting depth to prevent stack overflow
            if self._get_json_depth(data) > 20:
                raise InputSanitizationError("JSON nesting too deep")
            
            return data
        except json.JSONDecodeError as e:
            raise InputSanitizationError(f"Invalid JSON: {e}")
    
    def _get_json_depth(self, obj: Any, depth: int = 0) -> int:
        """Get maximum nesting depth of JSON object."""
        if depth > 50:  # Prevent infinite recursion
            return depth
        
        if isinstance(obj, dict):
            return max([self._get_json_depth(v, depth + 1) for v in obj.values()] + [depth])
        elif isinstance(obj, list):
            return max([self._get_json_depth(item, depth + 1) for item in obj] + [depth])
        else:
            return depth
    
    def sanitize_string(self, value: str, max_length: int = None, **kwargs) -> str:
        """Convenience method for string sanitization."""
        rule_kwargs = {'max_length': max_length} if max_length else {}
        rule_kwargs.update(kwargs)
        return self.sanitize(value, InputType.STRING, **rule_kwargs)
    
    def sanitize_circuit_name(self, name: str) -> str:
        """Sanitize quantum circuit name."""
        return self.sanitize(name, InputType.CIRCUIT_NAME)
    
    def sanitize_backend_name(self, name: str) -> str:
        """Sanitize quantum backend name."""
        return self.sanitize(name, InputType.BACKEND_NAME)
    
    def sanitize_parameter_name(self, name: str) -> str:
        """Sanitize parameter name."""
        return self.sanitize(name, InputType.PARAMETER_NAME)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename."""
        return self.sanitize(filename, InputType.FILENAME)
    
    def sanitize_dict(self, data: Dict[str, Any], type_map: Dict[str, InputType] = None) -> Dict[str, Any]:
        """
        Sanitize a dictionary of values.
        
        Args:
            data: Dictionary to sanitize
            type_map: Mapping of keys to input types
            
        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            raise InputSanitizationError("Input must be a dictionary")
        
        sanitized = {}
        type_map = type_map or {}
        
        for key, value in data.items():
            # Sanitize key
            sanitized_key = self.sanitize_parameter_name(key)
            
            # Sanitize value
            input_type = type_map.get(key, InputType.STRING)
            sanitized_value = self.sanitize(value, input_type)
            
            sanitized[sanitized_key] = sanitized_value
        
        return sanitized
    
    def validate_circuit_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate quantum circuit parameters.
        
        Args:
            params: Circuit parameters dictionary
            
        Returns:
            Validated parameters
        """
        type_map = {
            'num_qubits': InputType.INTEGER,
            'depth': InputType.INTEGER,
            'name': InputType.CIRCUIT_NAME,
            'backend': InputType.BACKEND_NAME,
            'shots': InputType.INTEGER,
            'theta': InputType.FLOAT,
            'phi': InputType.FLOAT,
            'lambda': InputType.FLOAT,
            'enabled': InputType.BOOLEAN
        }
        
        validated = {}
        for key, value in params.items():
            sanitized_key = self.sanitize_parameter_name(key)
            input_type = type_map.get(key, InputType.STRING)
            
            # Special validation for quantum parameters
            if key in ['num_qubits', 'depth', 'shots']:
                validated_value = self.sanitize(value, input_type)
                if validated_value < 0:
                    raise ValidationError(f"Parameter '{key}' must be non-negative")
                if key == 'num_qubits' and validated_value > 100:
                    raise ValidationError("Number of qubits cannot exceed 100")
                if key == 'shots' and validated_value > 1000000:
                    raise ValidationError("Number of shots cannot exceed 1,000,000")
            elif key in ['theta', 'phi', 'lambda']:
                validated_value = self.sanitize(value, input_type)
                # Angles should be reasonable
                if abs(validated_value) > 100 * math.pi:
                    warnings.warn(f"Large angle value for {key}: {validated_value}")
            else:
                validated_value = self.sanitize(value, input_type)
            
            validated[sanitized_key] = validated_value
        
        return validated
    
    def add_custom_rule(self, input_type: InputType, rule: SanitizationRule):
        """Add or update a sanitization rule."""
        self.rules[input_type] = rule
    
    def remove_rule(self, input_type: InputType):
        """Remove a sanitization rule."""
        if input_type in self.rules:
            del self.rules[input_type]