"""
Secure Configuration Management for QEM-Bench

This module provides secure handling of configuration data, including:
- Encryption of sensitive configuration values
- Secure storage and loading of configuration files
- Environment variable integration with validation
- Configuration validation and schema enforcement
- Secure defaults for all settings
"""

import os
import json
import warnings
from typing import Any, Dict, Optional, List, Union, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import tempfile

from ..errors import SecurityError, ConfigurationError, ValidationError
from .crypto_utils import CryptoUtils
from .input_sanitizer import InputSanitizer


class ConfigSensitivity(Enum):
    """Sensitivity levels for configuration values."""
    PUBLIC = "public"        # Can be logged and stored in plain text
    INTERNAL = "internal"    # Should not be logged but can be stored in plain text
    SENSITIVE = "sensitive"  # Should be encrypted at rest and never logged
    SECRET = "secret"        # Must be encrypted and access-controlled


@dataclass
class ConfigField:
    """Configuration field descriptor with security metadata."""
    name: str
    field_type: Type
    sensitivity: ConfigSensitivity = ConfigSensitivity.PUBLIC
    required: bool = False
    default: Any = None
    validator: Optional[callable] = None
    description: str = ""
    env_var: Optional[str] = None
    
    def validate(self, value: Any) -> Any:
        """Validate and convert the configuration value."""
        if value is None:
            if self.required:
                raise ValidationError(f"Required configuration field '{self.name}' is missing")
            return self.default
        
        # Type validation
        if not isinstance(value, self.field_type):
            try:
                value = self.field_type(value)
            except (ValueError, TypeError) as e:
                raise ValidationError(
                    f"Configuration field '{self.name}' must be of type {self.field_type.__name__}, "
                    f"got {type(value).__name__}: {value}"
                ) from e
        
        # Custom validation
        if self.validator:
            try:
                value = self.validator(value)
            except Exception as e:
                raise ValidationError(
                    f"Validation failed for configuration field '{self.name}': {e}"
                ) from e
        
        return value


class SecureConfig:
    """
    Secure configuration manager for QEM-Bench.
    
    Provides secure handling of configuration data with:
    - Automatic encryption of sensitive values
    - Environment variable integration
    - Configuration validation and schema enforcement
    - Secure defaults
    - Audit logging of configuration access
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        encryption_key: Optional[bytes] = None,
        enable_env_vars: bool = True,
        strict_mode: bool = True
    ):
        """
        Initialize secure configuration manager.
        
        Args:
            config_path: Path to configuration file
            encryption_key: Key for encrypting sensitive values (auto-generated if None)
            enable_env_vars: Whether to read from environment variables
            strict_mode: Whether to enforce strict validation
        """
        self.config_path = config_path
        self.enable_env_vars = enable_env_vars
        self.strict_mode = strict_mode
        
        # Initialize crypto utilities
        self.crypto = CryptoUtils()
        if encryption_key:
            self.crypto.set_encryption_key(encryption_key)
        else:
            self.crypto.generate_encryption_key()
        
        # Initialize sanitizer
        self.sanitizer = InputSanitizer()
        
        # Configuration schema and values
        self.schema: Dict[str, ConfigField] = {}
        self.values: Dict[str, Any] = {}
        self.encrypted_values: Dict[str, bytes] = {}
        
        # Setup default schema
        self._setup_default_schema()
        
        # Load configuration
        if config_path and config_path.exists():
            self.load_config(config_path)
    
    def _setup_default_schema(self):
        """Setup default configuration schema."""
        default_fields = [
            # Security settings
            ConfigField(
                name="security.enable_encryption",
                field_type=bool,
                default=True,
                description="Enable encryption for sensitive data"
            ),
            ConfigField(
                name="security.enable_audit_logging",
                field_type=bool,
                default=True,
                description="Enable security audit logging"
            ),
            ConfigField(
                name="security.rate_limit_enabled",
                field_type=bool,
                default=True,
                description="Enable rate limiting for API calls"
            ),
            ConfigField(
                name="security.max_request_size",
                field_type=int,
                default=10 * 1024 * 1024,  # 10MB
                validator=lambda x: x if x > 0 else 1024,
                description="Maximum request size in bytes"
            ),
            ConfigField(
                name="security.session_timeout",
                field_type=int,
                default=3600,  # 1 hour
                validator=lambda x: max(300, min(x, 86400)),  # 5 minutes to 24 hours
                description="Session timeout in seconds"
            ),
            
            # Resource limits
            ConfigField(
                name="resources.max_qubits",
                field_type=int,
                default=50,
                validator=lambda x: max(1, min(x, 1000)),
                description="Maximum number of qubits allowed"
            ),
            ConfigField(
                name="resources.max_circuits",
                field_type=int,
                default=1000,
                validator=lambda x: max(1, x),
                description="Maximum number of circuits per batch"
            ),
            ConfigField(
                name="resources.max_memory_mb",
                field_type=int,
                default=1024,
                validator=lambda x: max(100, x),
                description="Maximum memory usage in MB"
            ),
            ConfigField(
                name="resources.execution_timeout",
                field_type=int,
                default=300,  # 5 minutes
                validator=lambda x: max(10, min(x, 3600)),
                description="Circuit execution timeout in seconds"
            ),
            
            # Backend configuration
            ConfigField(
                name="backends.default_backend",
                field_type=str,
                default="jax_simulator",
                description="Default quantum backend to use"
            ),
            ConfigField(
                name="backends.enable_hardware",
                field_type=bool,
                default=False,
                description="Enable hardware backend access"
            ),
            
            # API configuration
            ConfigField(
                name="api.ibm_token",
                field_type=str,
                sensitivity=ConfigSensitivity.SECRET,
                env_var="QEM_IBM_TOKEN",
                description="IBM Quantum API token"
            ),
            ConfigField(
                name="api.aws_access_key",
                field_type=str,
                sensitivity=ConfigSensitivity.SECRET,
                env_var="QEM_AWS_ACCESS_KEY",
                description="AWS Braket access key"
            ),
            ConfigField(
                name="api.aws_secret_key",
                field_type=str,
                sensitivity=ConfigSensitivity.SECRET,
                env_var="QEM_AWS_SECRET_KEY",
                description="AWS Braket secret key"
            ),
            ConfigField(
                name="api.google_credentials_file",
                field_type=str,
                sensitivity=ConfigSensitivity.SENSITIVE,
                env_var="QEM_GOOGLE_CREDENTIALS",
                description="Path to Google Cloud credentials file"
            ),
            
            # Logging configuration
            ConfigField(
                name="logging.level",
                field_type=str,
                default="INFO",
                validator=lambda x: x.upper() if x.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] else "INFO",
                description="Logging level"
            ),
            ConfigField(
                name="logging.enable_file_logging",
                field_type=bool,
                default=True,
                description="Enable file-based logging"
            ),
            ConfigField(
                name="logging.log_file_path",
                field_type=str,
                default="qem_bench.log",
                description="Path to log file"
            ),
        ]
        
        for field in default_fields:
            self.add_field(field)
    
    def add_field(self, field: ConfigField):
        """Add a configuration field to the schema."""
        self.schema[field.name] = field
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (dot-notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Check if field is defined in schema
        if key in self.schema:
            field = self.schema[key]
            
            # Check environment variable first (if enabled)
            if self.enable_env_vars and field.env_var:
                env_value = os.getenv(field.env_var)
                if env_value is not None:
                    try:
                        return field.validate(env_value)
                    except ValidationError as e:
                        if self.strict_mode:
                            raise ConfigurationError(
                                f"Invalid environment variable {field.env_var}: {e}"
                            )
                        warnings.warn(f"Invalid environment variable {field.env_var}: {e}")
            
            # Check if value is encrypted
            if field.sensitivity in [ConfigSensitivity.SENSITIVE, ConfigSensitivity.SECRET]:
                if key in self.encrypted_values:
                    try:
                        decrypted = self.crypto.decrypt(self.encrypted_values[key])
                        return field.validate(decrypted.decode('utf-8'))
                    except Exception as e:
                        raise SecurityError(f"Failed to decrypt configuration value '{key}': {e}")
            
            # Check regular values
            if key in self.values:
                return field.validate(self.values[key])
            
            # Return default from field schema
            return field.validate(default or field.default)
        
        # Fallback for undefined keys
        if key in self.values:
            return self.values[key]
        
        if self.strict_mode:
            raise ConfigurationError(f"Unknown configuration key: {key}")
        
        return default
    
    def set(self, key: str, value: Any, encrypt: bool = None):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
            encrypt: Whether to encrypt the value (auto-detected from schema if None)
        """
        # Validate input
        value = self.sanitizer.sanitize_string(str(value)) if isinstance(value, str) else value
        
        if key in self.schema:
            field = self.schema[key]
            value = field.validate(value)
            
            # Determine if encryption is needed
            if encrypt is None:
                encrypt = field.sensitivity in [ConfigSensitivity.SENSITIVE, ConfigSensitivity.SECRET]
        
        if encrypt:
            # Encrypt and store
            encrypted_value = self.crypto.encrypt(str(value).encode('utf-8'))
            self.encrypted_values[key] = encrypted_value
            # Remove from plain text storage
            if key in self.values:
                del self.values[key]
        else:
            self.values[key] = value
            # Remove from encrypted storage
            if key in self.encrypted_values:
                del self.encrypted_values[key]
    
    def load_config(self, config_path: Path):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Validate and load configuration
            for key, value in data.items():
                if key == '_encrypted_values':
                    # Load encrypted values
                    for enc_key, enc_value in value.items():
                        self.encrypted_values[enc_key] = bytes.fromhex(enc_value)
                else:
                    self.set(key, value)
                    
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def save_config(self, config_path: Optional[Path] = None):
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration (uses default if None)
        """
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            raise ConfigurationError("No configuration path specified")
        
        # Prepare data for saving
        data = {}
        
        # Add regular values (excluding sensitive ones that should be encrypted)
        for key, value in self.values.items():
            if key in self.schema:
                field = self.schema[key]
                if field.sensitivity in [ConfigSensitivity.PUBLIC, ConfigSensitivity.INTERNAL]:
                    data[key] = value
        
        # Add encrypted values
        if self.encrypted_values:
            data['_encrypted_values'] = {
                key: value.hex() for key, value in self.encrypted_values.items()
            }
        
        try:
            # Write to temporary file first, then move (atomic operation)
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.json',
                dir=config_path.parent,
                delete=False
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_path = Path(tmp_file.name)
            
            # Set secure permissions
            os.chmod(tmp_path, 0o600)  # Read/write for owner only
            
            # Atomic move
            tmp_path.replace(config_path)
            
        except Exception as e:
            # Clean up temporary file if it exists
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def validate_all(self) -> List[str]:
        """
        Validate all configuration values.
        
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        
        for key, field in self.schema.items():
            try:
                value = self.get(key)
                field.validate(value)
            except Exception as e:
                errors.append(f"{key}: {e}")
        
        return errors
    
    def get_secure_summary(self) -> Dict[str, Any]:
        """
        Get a summary of configuration that's safe to log.
        
        Returns:
            Dictionary with non-sensitive configuration values
        """
        summary = {}
        
        for key, field in self.schema.items():
            if field.sensitivity == ConfigSensitivity.PUBLIC:
                try:
                    summary[key] = self.get(key)
                except Exception:
                    summary[key] = "<error>"
            elif field.sensitivity == ConfigSensitivity.INTERNAL:
                summary[key] = "<internal>"
            else:
                summary[key] = "<encrypted>"
        
        return summary
    
    def reset_to_defaults(self):
        """Reset all configuration values to defaults."""
        self.values.clear()
        self.encrypted_values.clear()
        
        # Set defaults from schema
        for key, field in self.schema.items():
            if field.default is not None:
                self.set(key, field.default)
    
    def create_child_config(self, prefix: str) -> "SecureConfig":
        """
        Create a child configuration with a key prefix.
        
        Args:
            prefix: Key prefix for child configuration
            
        Returns:
            New SecureConfig instance with filtered keys
        """
        child = SecureConfig(
            encryption_key=self.crypto.get_encryption_key(),
            enable_env_vars=self.enable_env_vars,
            strict_mode=self.strict_mode
        )
        
        # Copy matching schema and values
        prefix_with_dot = f"{prefix}."
        for key, field in self.schema.items():
            if key.startswith(prefix_with_dot):
                new_key = key[len(prefix_with_dot):]
                new_field = ConfigField(
                    name=new_key,
                    field_type=field.field_type,
                    sensitivity=field.sensitivity,
                    required=field.required,
                    default=field.default,
                    validator=field.validator,
                    description=field.description,
                    env_var=field.env_var
                )
                child.add_field(new_field)
                
                # Copy value if it exists
                try:
                    value = self.get(key)
                    child.set(new_key, value)
                except Exception:
                    pass
        
        return child


# Global secure configuration instance
_global_config: Optional[SecureConfig] = None


def get_secure_config() -> SecureConfig:
    """Get the global secure configuration instance."""
    global _global_config
    if _global_config is None:
        # Try to find configuration file
        config_paths = [
            Path.cwd() / "qem_bench_config.json",
            Path.home() / ".qem_bench" / "config.json",
            Path("/etc/qem_bench/config.json")
        ]
        
        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = path
                break
        
        _global_config = SecureConfig(config_path=config_path)
    
    return _global_config


def set_secure_config(config: SecureConfig):
    """Set the global secure configuration instance."""
    global _global_config
    _global_config = config