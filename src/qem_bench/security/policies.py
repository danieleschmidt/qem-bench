"""
Security Policies for QEM-Bench

This module defines security policies and default configurations:
- Default security settings
- Policy enforcement
- Security level configurations
- Compliance settings
- Risk management policies
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config import ConfigField, ConfigSensitivity
from .access_control import Permission, Role
from .resource_limiter import ResourceType, ResourceQuota
from .input_sanitizer import InputType


class SecurityLevel(Enum):
    """Security levels for different deployment scenarios."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    HIGH_SECURITY = "high_security"


class ComplianceStandard(Enum):
    """Compliance standards support."""
    BASIC = "basic"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    NIST = "nist"


@dataclass
class SecurityPolicy:
    """
    Comprehensive security policy definition.
    
    Defines all security settings and constraints for QEM-Bench operations.
    """
    name: str
    version: str = "1.0"
    security_level: SecurityLevel = SecurityLevel.PRODUCTION
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    
    # Authentication settings
    require_authentication: bool = True
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    password_min_length: int = 12
    require_2fa: bool = False
    
    # Authorization settings
    default_user_role: Role = Role.USER
    role_hierarchy: Dict[Role, List[Role]] = field(default_factory=dict)
    permission_inheritance: bool = True
    
    # Encryption settings
    enable_encryption: bool = True
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    key_rotation_days: int = 90
    
    # Input validation settings
    strict_input_validation: bool = True
    max_input_length: int = 10000
    allowed_file_types: List[str] = field(default_factory=list)
    sanitize_html: bool = True
    
    # Resource limits
    max_qubits_per_circuit: int = 50
    max_circuits_per_batch: int = 100
    max_execution_time_seconds: int = 300
    max_memory_mb: int = 1024
    max_file_size_mb: int = 100
    
    # Rate limiting
    api_calls_per_minute: int = 100
    circuit_executions_per_hour: int = 50
    backend_calls_per_minute: int = 20
    
    # Audit logging
    enable_audit_logging: bool = True
    log_all_operations: bool = False
    log_sensitive_operations: bool = True
    audit_log_retention_days: int = 365
    
    # Network security
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    require_https: bool = True
    
    # Data protection
    data_classification_required: bool = False
    automatic_data_encryption: bool = True
    secure_deletion: bool = True
    backup_encryption: bool = True
    
    # Monitoring and alerting
    security_monitoring: bool = True
    real_time_alerts: bool = True
    anomaly_detection: bool = False
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def get_resource_quotas(self) -> List[ResourceQuota]:
        """Get resource quotas from policy settings."""
        quotas = [
            ResourceQuota(
                resource_type=ResourceType.MEMORY,
                limit=self.max_memory_mb * 1024 * 1024,
                description="Maximum memory usage"
            ),
            ResourceQuota(
                resource_type=ResourceType.CPU_TIME,
                limit=self.max_execution_time_seconds,
                description="Maximum CPU time per operation"
            ),
            ResourceQuota(
                resource_type=ResourceType.QUBITS,
                limit=self.max_qubits_per_circuit,
                description="Maximum qubits per circuit"
            ),
            ResourceQuota(
                resource_type=ResourceType.CIRCUITS,
                limit=self.max_circuits_per_batch,
                description="Maximum circuits per batch"
            ),
            ResourceQuota(
                resource_type=ResourceType.API_CALLS,
                limit=self.api_calls_per_minute,
                window_seconds=60,
                description="API calls per minute"
            ),
        ]
        
        return quotas
    
    def get_config_fields(self) -> List[ConfigField]:
        """Get configuration fields from policy settings."""
        fields = [
            ConfigField(
                name="security.authentication_required",
                field_type=bool,
                default=self.require_authentication,
                description="Whether authentication is required"
            ),
            ConfigField(
                name="security.session_timeout",
                field_type=int,
                default=self.session_timeout_minutes * 60,
                description="Session timeout in seconds"
            ),
            ConfigField(
                name="security.encryption_enabled",
                field_type=bool,
                default=self.enable_encryption,
                description="Whether encryption is enabled"
            ),
            ConfigField(
                name="security.audit_logging",
                field_type=bool,
                default=self.enable_audit_logging,
                description="Whether audit logging is enabled"
            ),
            ConfigField(
                name="limits.max_qubits",
                field_type=int,
                default=self.max_qubits_per_circuit,
                validator=lambda x: 1 <= x <= 1000,
                description="Maximum qubits per circuit"
            ),
            ConfigField(
                name="limits.max_memory_mb",
                field_type=int,
                default=self.max_memory_mb,
                validator=lambda x: x > 0,
                description="Maximum memory in MB"
            ),
            ConfigField(
                name="limits.execution_timeout",
                field_type=int,
                default=self.max_execution_time_seconds,
                validator=lambda x: 1 <= x <= 3600,
                description="Execution timeout in seconds"
            ),
        ]
        
        return fields
    
    def validate(self) -> List[str]:
        """Validate policy settings."""
        errors = []
        
        # Basic validation
        if self.session_timeout_minutes <= 0:
            errors.append("Session timeout must be positive")
        
        if self.max_login_attempts <= 0:
            errors.append("Max login attempts must be positive")
        
        if self.password_min_length < 8:
            errors.append("Password minimum length should be at least 8")
        
        if self.max_qubits_per_circuit <= 0:
            errors.append("Max qubits must be positive")
        
        if self.max_execution_time_seconds <= 0:
            errors.append("Max execution time must be positive")
        
        if self.api_calls_per_minute <= 0:
            errors.append("API calls per minute must be positive")
        
        # Security level validation
        if self.security_level == SecurityLevel.HIGH_SECURITY:
            if not self.require_authentication:
                errors.append("Authentication required for high security level")
            
            if not self.enable_encryption:
                errors.append("Encryption required for high security level")
            
            if not self.enable_audit_logging:
                errors.append("Audit logging required for high security level")
        
        # Compliance validation
        if ComplianceStandard.GDPR in self.compliance_standards:
            if not self.enable_audit_logging:
                errors.append("GDPR compliance requires audit logging")
            
            if not self.secure_deletion:
                errors.append("GDPR compliance requires secure deletion")
        
        if ComplianceStandard.HIPAA in self.compliance_standards:
            if not self.encryption_at_rest:
                errors.append("HIPAA compliance requires encryption at rest")
            
            if not self.encryption_in_transit:
                errors.append("HIPAA compliance requires encryption in transit")
        
        return errors
    
    def apply_to_system(self):
        """Apply policy settings to the security system."""
        from .config import get_secure_config
        from .access_control import get_global_access_control
        from .resource_limiter import get_global_resource_limiter
        
        # Apply configuration settings
        config = get_secure_config()
        for field in self.get_config_fields():
            config.set(field.name, field.default)
        
        # Apply resource quotas
        resource_limiter = get_global_resource_limiter()
        for quota in self.get_resource_quotas():
            resource_limiter.add_quota(quota)
        
        # Apply access control settings
        access_control = get_global_access_control()
        
        # Set up role hierarchy if specified
        if self.role_hierarchy:
            # Implementation would depend on access control system
            pass


def get_development_policy() -> SecurityPolicy:
    """Get security policy for development environment."""
    return SecurityPolicy(
        name="development",
        security_level=SecurityLevel.DEVELOPMENT,
        require_authentication=False,
        session_timeout_minutes=480,  # 8 hours
        max_login_attempts=10,
        password_min_length=8,
        
        # Relaxed limits for development
        max_qubits_per_circuit=100,
        max_circuits_per_batch=1000,
        max_execution_time_seconds=600,  # 10 minutes
        max_memory_mb=2048,
        
        # Higher rate limits
        api_calls_per_minute=1000,
        circuit_executions_per_hour=500,
        
        # Basic logging
        enable_audit_logging=True,
        log_all_operations=False,
        audit_log_retention_days=30,
        
        # Development-friendly settings
        strict_input_validation=False,
        require_https=False,
        allowed_file_types=['.json', '.txt', '.log', '.csv', '.pkl', '.npz', '.h5', '.yaml', '.yml', '.py']
    )


def get_testing_policy() -> SecurityPolicy:
    """Get security policy for testing environment."""
    return SecurityPolicy(
        name="testing",
        security_level=SecurityLevel.TESTING,
        require_authentication=True,
        session_timeout_minutes=120,  # 2 hours
        max_login_attempts=5,
        password_min_length=10,
        
        # Moderate limits
        max_qubits_per_circuit=75,
        max_circuits_per_batch=500,
        max_execution_time_seconds=450,
        max_memory_mb=1536,
        
        # Moderate rate limits
        api_calls_per_minute=200,
        circuit_executions_per_hour=100,
        
        # Enhanced logging
        enable_audit_logging=True,
        log_sensitive_operations=True,
        audit_log_retention_days=90,
        
        # Testing settings
        strict_input_validation=True,
        require_https=True,
        allowed_file_types=['.json', '.txt', '.log', '.csv', '.pkl', '.npz', '.h5', '.yaml', '.yml']
    )


def get_production_policy() -> SecurityPolicy:
    """Get security policy for production environment."""
    return SecurityPolicy(
        name="production",
        security_level=SecurityLevel.PRODUCTION,
        require_authentication=True,
        session_timeout_minutes=60,
        max_login_attempts=3,
        password_min_length=12,
        require_2fa=False,
        
        # Production limits
        max_qubits_per_circuit=50,
        max_circuits_per_batch=100,
        max_execution_time_seconds=300,
        max_memory_mb=1024,
        
        # Standard rate limits
        api_calls_per_minute=100,
        circuit_executions_per_hour=50,
        backend_calls_per_minute=20,
        
        # Full security features
        enable_encryption=True,
        encryption_at_rest=True,
        encryption_in_transit=True,
        
        # Comprehensive logging
        enable_audit_logging=True,
        log_sensitive_operations=True,
        audit_log_retention_days=365,
        
        # Production security
        strict_input_validation=True,
        require_https=True,
        security_monitoring=True,
        real_time_alerts=True,
        
        # Data protection
        automatic_data_encryption=True,
        secure_deletion=True,
        backup_encryption=True,
        
        allowed_file_types=['.json', '.csv', '.h5', '.yaml', '.yml']
    )


def get_high_security_policy() -> SecurityPolicy:
    """Get high security policy for sensitive environments."""
    return SecurityPolicy(
        name="high_security",
        security_level=SecurityLevel.HIGH_SECURITY,
        compliance_standards=[ComplianceStandard.SOC2, ComplianceStandard.NIST],
        
        # Strict authentication
        require_authentication=True,
        session_timeout_minutes=30,
        max_login_attempts=3,
        password_min_length=16,
        require_2fa=True,
        
        # Conservative limits
        max_qubits_per_circuit=25,
        max_circuits_per_batch=50,
        max_execution_time_seconds=180,  # 3 minutes
        max_memory_mb=512,
        max_file_size_mb=50,
        
        # Strict rate limits
        api_calls_per_minute=50,
        circuit_executions_per_hour=25,
        backend_calls_per_minute=10,
        
        # Maximum security
        enable_encryption=True,
        encryption_at_rest=True,
        encryption_in_transit=True,
        key_rotation_days=30,
        
        # Comprehensive logging
        enable_audit_logging=True,
        log_all_operations=True,
        log_sensitive_operations=True,
        audit_log_retention_days=2555,  # 7 years
        
        # Maximum security settings
        strict_input_validation=True,
        max_input_length=1000,
        sanitize_html=True,
        require_https=True,
        
        # Enhanced monitoring
        security_monitoring=True,
        real_time_alerts=True,
        anomaly_detection=True,
        
        # Maximum data protection
        data_classification_required=True,
        automatic_data_encryption=True,
        secure_deletion=True,
        backup_encryption=True,
        
        allowed_file_types=['.json', '.yaml']
    )


def get_gdpr_compliant_policy() -> SecurityPolicy:
    """Get GDPR compliant security policy."""
    policy = get_production_policy()
    policy.name = "gdpr_compliant"
    policy.compliance_standards = [ComplianceStandard.GDPR]
    
    # GDPR specific requirements
    policy.enable_audit_logging = True
    policy.log_all_operations = True
    policy.audit_log_retention_days = 2555  # 7 years
    policy.secure_deletion = True
    policy.data_classification_required = True
    policy.automatic_data_encryption = True
    
    # Enhanced user rights
    policy.custom_settings.update({
        'data_portability': True,
        'right_to_be_forgotten': True,
        'consent_management': True,
        'data_minimization': True,
        'purpose_limitation': True
    })
    
    return policy


def get_hipaa_compliant_policy() -> SecurityPolicy:
    """Get HIPAA compliant security policy."""
    policy = get_high_security_policy()
    policy.name = "hipaa_compliant"
    policy.compliance_standards = [ComplianceStandard.HIPAA]
    
    # HIPAA specific requirements
    policy.encryption_at_rest = True
    policy.encryption_in_transit = True
    policy.require_2fa = True
    policy.session_timeout_minutes = 15  # Short session timeout
    
    # Enhanced access controls
    policy.custom_settings.update({
        'minimum_necessary': True,
        'access_logs_required': True,
        'workforce_training': True,
        'business_associate_agreements': True,
        'risk_assessments': True
    })
    
    return policy


# Policy registry
_POLICY_REGISTRY = {
    SecurityLevel.DEVELOPMENT: get_development_policy,
    SecurityLevel.TESTING: get_testing_policy,
    SecurityLevel.PRODUCTION: get_production_policy,
    SecurityLevel.HIGH_SECURITY: get_high_security_policy,
}

_COMPLIANCE_POLICIES = {
    ComplianceStandard.GDPR: get_gdpr_compliant_policy,
    ComplianceStandard.HIPAA: get_hipaa_compliant_policy,
}


def get_policy_for_level(level: SecurityLevel) -> SecurityPolicy:
    """Get security policy for a specific security level."""
    if level in _POLICY_REGISTRY:
        return _POLICY_REGISTRY[level]()
    else:
        return get_production_policy()  # Default to production


def get_policy_for_compliance(standard: ComplianceStandard) -> SecurityPolicy:
    """Get security policy for a specific compliance standard."""
    if standard in _COMPLIANCE_POLICIES:
        return _COMPLIANCE_POLICIES[standard]()
    else:
        return get_production_policy()  # Default to production


def get_default_policy() -> SecurityPolicy:
    """Get the default security policy."""
    return get_production_policy()


def create_custom_policy(
    base_level: SecurityLevel = SecurityLevel.PRODUCTION,
    **overrides
) -> SecurityPolicy:
    """
    Create a custom security policy based on a base level.
    
    Args:
        base_level: Base security level to start from
        **overrides: Policy settings to override
        
    Returns:
        Custom security policy
    """
    base_policy = get_policy_for_level(base_level)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(base_policy, key):
            setattr(base_policy, key, value)
        else:
            base_policy.custom_settings[key] = value
    
    base_policy.name = f"custom_{base_level.value}"
    
    return base_policy


def validate_policy(policy: SecurityPolicy) -> bool:
    """
    Validate a security policy.
    
    Args:
        policy: Security policy to validate
        
    Returns:
        True if policy is valid
    """
    errors = policy.validate()
    if errors:
        for error in errors:
            print(f"Policy validation error: {error}")
        return False
    return True


def apply_policy(policy: SecurityPolicy):
    """
    Apply a security policy to the system.
    
    Args:
        policy: Security policy to apply
    """
    if not validate_policy(policy):
        raise ValueError("Invalid security policy")
    
    policy.apply_to_system()


# Global policy instance
_current_policy: Optional[SecurityPolicy] = None


def get_current_policy() -> SecurityPolicy:
    """Get the current active security policy."""
    global _current_policy
    if _current_policy is None:
        _current_policy = get_default_policy()
    return _current_policy


def set_current_policy(policy: SecurityPolicy):
    """Set the current active security policy."""
    global _current_policy
    if validate_policy(policy):
        _current_policy = policy
        apply_policy(policy)
    else:
        raise ValueError("Invalid security policy")