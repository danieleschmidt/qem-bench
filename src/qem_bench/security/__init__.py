"""
QEM-Bench Security Framework

This module provides comprehensive security measures for the quantum error mitigation library,
including secure configuration management, credential handling, input validation, access control,
and cryptographic operations.

The security framework is designed to:
- Protect sensitive data (API keys, certificates, credentials)
- Validate and sanitize all user inputs
- Prevent resource exhaustion attacks
- Provide secure defaults for all operations
- Maintain audit logs for security-relevant operations
- Ensure backward compatibility while adding security layers

Key Components:
- SecureConfig: Secure configuration management
- CredentialManager: API key and authentication token management
- InputSanitizer: Input validation and sanitization
- ResourceLimiter: Resource quota and rate limiting
- CryptoUtils: Cryptographic operations and utilities
- AccessControl: Permission and access management
- AuditLogger: Security event logging
"""

from .security_manager import SecurityManager, SecurityPolicy, get_security_manager, set_security_manager
from .config import SecureConfig, get_secure_config, set_secure_config
from .credentials import CredentialManager, get_credential_manager, set_credential_manager
from .input_sanitizer import InputSanitizer, InputType
from .resource_limiter import (
    ResourceLimiter, ResourceType, ResourceQuota, get_global_resource_limiter,
    set_global_resource_limiter, limit_memory, limit_qubits, limit_execution_time
)
from .crypto_utils import CryptoUtils, SecureRandom, get_crypto_utils, set_crypto_utils
from .access_control import (
    AccessControl, RateLimiter, Permission, Role, User, get_global_access_control,
    set_global_access_control
)
from .audit_logger import (
    AuditLogger, AuditEventType, AuditLevel, AuditEvent, get_audit_logger,
    set_audit_logger
)
from .serialization import (
    SecureSerializer, SerializationFormat, get_secure_serializer,
    set_secure_serializer, serialize_circuit_config, deserialize_circuit_config
)
from .file_ops import SecureFileOperations, get_secure_file_ops, set_secure_file_ops
from .decorators import (
    require_authentication,
    validate_input,
    rate_limit,
    audit_log,
    secure_operation,
    encrypt_sensitive_data,
    timeout,
    circuit_security,
    error_handler
)
from .policies import (
    SecurityPolicy, SecurityLevel, ComplianceStandard,
    get_default_policy, get_current_policy, set_current_policy,
    get_policy_for_level, create_custom_policy, apply_policy
)

__all__ = [
    # Core security components
    "SecureConfig", "get_secure_config", "set_secure_config",
    "CredentialManager", "get_credential_manager", "set_credential_manager",
    "InputSanitizer", "InputType",
    "ResourceLimiter", "ResourceType", "ResourceQuota", 
    "get_global_resource_limiter", "set_global_resource_limiter",
    "limit_memory", "limit_qubits", "limit_execution_time",
    "CryptoUtils", "SecureRandom", "get_crypto_utils", "set_crypto_utils",
    "AccessControl", "RateLimiter", "Permission", "Role", "User",
    "get_global_access_control", "set_global_access_control",
    "AuditLogger", "AuditEventType", "AuditLevel", "AuditEvent",
    "get_audit_logger", "set_audit_logger",
    
    # Serialization and file operations
    "SecureSerializer", "SerializationFormat", "get_secure_serializer",
    "set_secure_serializer", "serialize_circuit_config", "deserialize_circuit_config",
    "SecureFileOperations", "get_secure_file_ops", "set_secure_file_ops",
    
    # Security decorators
    "require_authentication", "validate_input", "rate_limit", "audit_log", 
    "secure_operation", "encrypt_sensitive_data", "timeout", "circuit_security",
    "error_handler",
    
    # Policies
    "SecurityPolicy", "SecurityLevel", "ComplianceStandard",
    "get_default_policy", "get_current_policy", "set_current_policy",
    "get_policy_for_level", "create_custom_policy", "apply_policy",
]