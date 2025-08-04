"""
Basic tests for QEM-Bench security framework.

These tests verify that the security components are properly integrated
and can be imported without errors.
"""

import pytest
import tempfile
from pathlib import Path

# Test imports work correctly
def test_security_imports():
    """Test that all security components can be imported."""
    from qem_bench.security import (
        SecureConfig, CredentialManager, InputSanitizer, ResourceLimiter,
        AccessControl, AuditLogger, SecurityPolicy
    )
    
    # Test that classes can be instantiated
    config = SecureConfig()
    assert config is not None
    
    cred_manager = CredentialManager()
    assert cred_manager is not None
    
    sanitizer = InputSanitizer()
    assert sanitizer is not None
    
    resource_limiter = ResourceLimiter(enable_monitoring=False)
    assert resource_limiter is not None
    
    access_control = AccessControl(enable_audit_logging=False)
    assert access_control is not None
    
    audit_logger = AuditLogger(enable_console=False, enable_syslog=False)
    assert audit_logger is not None
    
    from qem_bench.security import get_default_policy
    policy = get_default_policy()
    assert policy is not None


def test_input_sanitization():
    """Test input sanitization functionality."""
    from qem_bench.security import InputSanitizer, InputType
    
    sanitizer = InputSanitizer()
    
    # Test string sanitization
    clean_string = sanitizer.sanitize("test_string", InputType.STRING)
    assert clean_string == "test_string"
    
    # Test integer sanitization
    clean_int = sanitizer.sanitize("42", InputType.INTEGER)
    assert clean_int == 42
    
    # Test float sanitization
    clean_float = sanitizer.sanitize("3.14", InputType.FLOAT)
    assert clean_float == 3.14
    
    # Test circuit name sanitization
    circuit_name = sanitizer.sanitize_circuit_name("test_circuit")
    assert circuit_name == "test_circuit"


def test_secure_config():
    """Test secure configuration management."""
    from qem_bench.security import SecureConfig
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"
        
        config = SecureConfig(config_path=config_path)
        
        # Test setting and getting values
        config.set("test_key", "test_value")
        value = config.get("test_key")
        assert value == "test_value"
        
        # Test default values
        default_value = config.get("nonexistent_key", "default")
        assert default_value == "default"


def test_resource_limiting():
    """Test resource limiting functionality."""
    from qem_bench.security import ResourceLimiter, ResourceType
    
    limiter = ResourceLimiter(enable_monitoring=False)
    
    # Test resource allocation
    success = limiter.allocate_resource(ResourceType.MEMORY, 1024)  # 1KB
    assert success is True
    
    # Test resource release
    limiter.release_resource(ResourceType.MEMORY, 1024)


def test_access_control():
    """Test access control functionality."""
    from qem_bench.security import AccessControl, Permission, Role
    
    access_control = AccessControl(enable_audit_logging=False)
    
    # Test user creation
    user = access_control.create_user("test_user", "Test User", {Role.USER})
    assert user.user_id == "test_user"
    assert user.username == "Test User"
    assert Role.USER in user.roles
    
    # Test permission checking
    has_permission = access_control.check_permission("test_user", Permission.CREATE_CIRCUIT)
    assert has_permission is True  # USER role should have CREATE_CIRCUIT permission


def test_security_decorators():
    """Test security decorators can be imported and used."""
    from qem_bench.security import validate_input, InputType
    
    @validate_input(test_param=InputType.STRING)
    def test_function(test_param: str):
        return test_param.upper()
    
    result = test_function("hello")
    assert result == "HELLO"


def test_audit_logging():
    """Test audit logging functionality."""
    from qem_bench.security import AuditLogger, AuditEventType, AuditLevel
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "audit.log"
        
        logger = AuditLogger(
            log_file=log_path,
            enable_console=False,
            enable_syslog=False
        )
        
        # Test logging an event
        logger.log_security_event(
            event_type=AuditEventType.SYSTEM_STARTUP,
            level=AuditLevel.INFO,
            details={"test": "data"}
        )
        
        # Flush to ensure event is written
        logger.flush()
        
        # Verify log file was created
        assert log_path.exists()


def test_security_policies():
    """Test security policy functionality."""
    from qem_bench.security import (
        SecurityPolicy, SecurityLevel, get_policy_for_level,
        create_custom_policy
    )
    
    # Test getting policy for security level
    prod_policy = get_policy_for_level(SecurityLevel.PRODUCTION)
    assert prod_policy.security_level == SecurityLevel.PRODUCTION
    assert prod_policy.require_authentication is True
    
    # Test creating custom policy
    custom_policy = create_custom_policy(
        SecurityLevel.DEVELOPMENT,
        max_qubits_per_circuit=100
    )
    assert custom_policy.max_qubits_per_circuit == 100


def test_credential_management():
    """Test credential management functionality."""
    from qem_bench.security import CredentialManager, CredentialType
    from datetime import datetime, timedelta
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir) / "credentials.enc"
        
        cred_manager = CredentialManager(
            storage_path=storage_path,
            auto_cleanup=False
        )
        
        # Test storing a credential
        success = cred_manager.store_credential(
            name="test_api_key",
            value="secret_key_123",
            credential_type=CredentialType.API_KEY,
            expires_at=datetime.now() + timedelta(days=30)
        )
        assert success is True
        
        # Test retrieving the credential
        retrieved_value = cred_manager.get_credential("test_api_key")
        assert retrieved_value == "secret_key_123"
        
        # Test listing credentials
        credentials = cred_manager.list_credentials()
        assert len(credentials) == 1
        assert credentials[0]['name'] == "test_api_key"


if __name__ == "__main__":
    pytest.main([__file__])