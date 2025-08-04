# QEM-Bench Security Framework

## Overview

The QEM-Bench Security Framework provides comprehensive security measures for the quantum error mitigation library, protecting sensitive data, preventing attacks, and ensuring secure operation in production environments.

## Architecture

The security framework is built around several key components:

```
qem_bench/security/
├── __init__.py              # Main security module exports
├── config.py               # Secure configuration management
├── credentials.py          # API key and credential management
├── input_sanitizer.py      # Input validation and sanitization
├── resource_limiter.py     # Resource quotas and rate limiting
├── crypto_utils.py         # Cryptographic operations
├── access_control.py       # Authentication and authorization
├── audit_logger.py         # Security event logging
├── serialization.py        # Secure data serialization
├── file_ops.py            # Secure file operations
├── decorators.py          # Security decorators
├── policies.py            # Security policies and configurations
└── secure_mitigation.py   # Example security integration
```

## Core Components

### 1. Secure Configuration (`SecureConfig`)

Manages sensitive configuration data with encryption and validation:

```python
from qem_bench.security import SecureConfig

config = SecureConfig()
config.set("api.ibm_token", "secret_token", encrypt=True)
token = config.get("api.ibm_token")  # Automatically decrypted
```

Features:
- Automatic encryption of sensitive values
- Environment variable integration
- Schema validation
- Secure file storage with atomic writes

### 2. Credential Management (`CredentialManager`)

Securely stores and manages API keys, tokens, and certificates:

```python
from qem_bench.security import CredentialManager, CredentialType

cred_manager = CredentialManager()
cred_manager.store_credential(
    name="ibm_token",
    value="your_secret_token",
    credential_type=CredentialType.API_KEY,
    expires_at=datetime.now() + timedelta(days=30)
)

token = cred_manager.get_credential("ibm_token")
```

Features:
- Encrypted storage of credentials
- Automatic expiration handling
- Credential rotation support
- Usage tracking and audit logging

### 3. Input Sanitization (`InputSanitizer`)

Validates and sanitizes all user inputs to prevent injection attacks:

```python
from qem_bench.security import InputSanitizer, InputType

sanitizer = InputSanitizer()
clean_circuit = sanitizer.sanitize_circuit_name("my_circuit")
safe_params = sanitizer.validate_circuit_parameters({
    "num_qubits": 10,
    "shots": 1000,
    "backend": "ibm_simulator"
})
```

Features:
- Type validation and conversion
- Range checking and limits
- Path traversal protection
- SQL injection prevention
- Circuit parameter validation

### 4. Resource Management (`ResourceLimiter`)

Prevents resource exhaustion attacks through quotas and monitoring:

```python
from qem_bench.security import ResourceLimiter, ResourceType

limiter = ResourceLimiter()
with limiter.allocate_resource(ResourceType.QUBITS, 50):
    # Execute quantum circuit with resource protection
    pass
```

Features:
- Memory, CPU, and qubit limits
- Per-user resource quotas
- Rate limiting for API calls
- Real-time resource monitoring
- Automatic cleanup of expired resources

### 5. Access Control (`AccessControl`)

Role-based access control with authentication and authorization:

```python
from qem_bench.security import AccessControl, Permission, Role

access_control = AccessControl()
user = access_control.create_user("researcher", "Dr. Smith", {Role.RESEARCHER})
access_control.require_permission("researcher", Permission.EXECUTE_CIRCUIT)
```

Features:
- Role-based permissions
- Rate limiting per user/operation
- Session management
- Permission inheritance
- Access logging

### 6. Audit Logging (`AuditLogger`)

Comprehensive logging of security-relevant events:

```python
from qem_bench.security import AuditLogger, AuditEventType

logger = AuditLogger()
logger.log_circuit_execution(
    circuit_name="bell_state",
    backend="ibm_quantum",
    num_qubits=2,
    shots=1000,
    user_id="researcher"
)
```

Features:
- Structured JSON logging
- Multiple output destinations
- Event filtering and rotation
- Tamper detection
- Compliance reporting

### 7. Cryptographic Operations (`CryptoUtils`)

Secure cryptographic functions for data protection:

```python
from qem_bench.security import CryptoUtils, SecureRandom

crypto = CryptoUtils()
crypto.generate_encryption_key()
encrypted_data = crypto.encrypt(b"sensitive_data")
decrypted_data = crypto.decrypt(encrypted_data)

# Secure random numbers for quantum experiments
random_gen = SecureRandom()
angles = random_gen.random_angles(3)  # Random rotation angles
```

Features:
- Symmetric and asymmetric encryption
- Secure random number generation
- Hash functions and HMAC
- Key derivation and management
- Certificate validation

### 8. Secure Serialization (`SecureSerializer`)

Safe serialization with schema validation and encryption:

```python
from qem_bench.security import SecureSerializer, SerializationFormat

serializer = SecureSerializer()
data = {"circuit": "bell_state", "shots": 1000}
encrypted_bytes = serializer.serialize(
    data, 
    SerializationFormat.ENCRYPTED_JSON,
    schema_name="circuit_config"
)
```

Features:
- Schema validation
- Encrypted serialization
- Protection against deserialization attacks
- Type safety and validation
- JSON depth limiting

### 9. Secure File Operations (`SecureFileOperations`)

Safe file handling with path validation and encryption:

```python
from qem_bench.security import SecureFileOperations

file_ops = SecureFileOperations()
content = file_ops.secure_read("data/results.json", decrypt=True)
file_ops.secure_write("data/output.json", content, encrypt=True)
```

Features:
- Path traversal protection
- Permission validation
- File integrity checking
- Secure temporary files
- Encrypted file storage

## Security Decorators

The framework provides decorators to easily add security features to functions:

### Authentication and Authorization

```python
from qem_bench.security import require_authentication, Permission

@require_authentication([Permission.EXECUTE_CIRCUIT])
def execute_quantum_circuit(circuit, user_id):
    # Function requires authentication and EXECUTE_CIRCUIT permission
    pass
```

### Input Validation

```python
from qem_bench.security import validate_input, InputType

@validate_input(
    circuit_name=InputType.CIRCUIT_NAME,
    num_qubits=InputType.INTEGER,
    shots=InputType.INTEGER
)
def create_circuit(circuit_name, num_qubits, shots):
    # All inputs are automatically validated and sanitized
    pass
```

### Rate Limiting

```python
from qem_bench.security import rate_limit

@rate_limit("circuit_execution", requests_per_minute=10)
def execute_circuit(circuit, user_id):
    # Function is rate-limited to 10 executions per minute per user
    pass
```

### Comprehensive Security

```python
from qem_bench.security import secure_operation, Permission, InputType, ResourceType

@secure_operation(
    permissions=[Permission.EXECUTE_CIRCUIT],
    validate_inputs={"shots": InputType.INTEGER},
    rate_limit_operation="zne_execution",
    resource_limits={ResourceType.QUBITS: 50}
)
def zne_mitigate(circuit, shots, user_id):
    # Comprehensive security: auth, validation, rate limiting, resource limits
    pass
```

## Security Policies

The framework includes predefined security policies for different environments:

```python
from qem_bench.security import SecurityLevel, get_policy_for_level, apply_policy

# Get production security policy
policy = get_policy_for_level(SecurityLevel.PRODUCTION)
apply_policy(policy)

# Create custom policy
from qem_bench.security import create_custom_policy
custom_policy = create_custom_policy(
    SecurityLevel.PRODUCTION,
    max_qubits_per_circuit=100,
    session_timeout_minutes=30
)
```

Available security levels:
- `DEVELOPMENT`: Relaxed security for development
- `TESTING`: Moderate security for testing environments
- `PRODUCTION`: Full security for production deployment
- `HIGH_SECURITY`: Maximum security for sensitive environments

## Integration Examples

### Securing Existing Mitigation Methods

```python
from qem_bench.security.secure_mitigation import SecureZeroNoiseExtrapolation

# Create security-enhanced ZNE
zne = SecureZeroNoiseExtrapolation(
    user_id="researcher",
    noise_factors=[1.0, 1.5, 2.0]
)

# Execute with automatic security validation
result = zne.mitigate(circuit, backend, observable, num_shots=1000)
```

### Custom Security Integration

```python
from qem_bench.security import secure_mitigation_function, Permission

@secure_mitigation_function(
    permissions=[Permission.EXECUTE_CIRCUIT],
    max_qubits=25,
    rate_limit_operation="custom_mitigation"
)
def my_mitigation_method(circuit, backend):
    # Your mitigation implementation with automatic security
    pass
```

## Configuration

### Environment Variables

The security framework supports configuration via environment variables:

```bash
export QEM_IBM_TOKEN="your_ibm_token"
export QEM_AWS_ACCESS_KEY="your_aws_key"
export QEM_SECURITY_LEVEL="production"
```

### Configuration Files

Security settings can be stored in JSON configuration files:

```json
{
  "security": {
    "authentication_required": true,
    "encryption_enabled": true,
    "audit_logging": true,
    "session_timeout": 3600
  },
  "limits": {
    "max_qubits": 50,
    "max_memory_mb": 1024,
    "api_calls_per_minute": 100
  }
}
```

## Best Practices

### 1. Always Use Authentication

```python
# Good: Require authentication for sensitive operations
@require_authentication([Permission.EXECUTE_CIRCUIT])
def execute_circuit(circuit, user_id):
    pass

# Bad: No authentication required
def execute_circuit(circuit):
    pass
```

### 2. Validate All Inputs

```python
# Good: Validate and sanitize inputs
@validate_input(num_qubits=InputType.INTEGER)
def create_circuit(num_qubits):
    if num_qubits > 100:
        raise ValueError("Too many qubits")
    
# Bad: Trust user input
def create_circuit(num_qubits):
    # No validation - vulnerable to injection
    pass
```

### 3. Use Resource Limits

```python
# Good: Apply resource limits
@secure_operation(resource_limits={ResourceType.QUBITS: 50})
def quantum_operation(circuit):
    pass

# Bad: No resource limits - vulnerable to DoS
def quantum_operation(circuit):
    # Could consume unlimited resources
    pass
```

### 4. Enable Audit Logging

```python
# Good: Log security-relevant operations
@audit_log(AuditEventType.CIRCUIT_EXECUTED)
def execute_circuit(circuit, user_id):
    pass

# Bad: No audit trail
def execute_circuit(circuit):
    # No record of execution
    pass
```

### 5. Handle Credentials Securely

```python
# Good: Use credential manager
cred_manager = get_credential_manager()
token = cred_manager.get_credential("api_token")

# Bad: Hardcoded credentials
token = "hardcoded_secret_token"  # Never do this!
```

## Compliance

The security framework supports various compliance standards:

- **GDPR**: Data protection and privacy
- **HIPAA**: Healthcare data security
- **SOC 2**: Service organization controls
- **NIST**: Cybersecurity framework

```python
from qem_bench.security import get_policy_for_compliance, ComplianceStandard

# Get GDPR-compliant policy
gdpr_policy = get_policy_for_compliance(ComplianceStandard.GDPR)
apply_policy(gdpr_policy)
```

## Testing

The security framework includes comprehensive tests to verify functionality:

```bash
# Run security tests
python -m pytest tests/security/

# Run specific security component tests
python -m pytest tests/security/test_security_basic.py
```

## Performance Considerations

The security framework is designed to have minimal performance impact:

- **Input validation**: ~1ms overhead per function call
- **Encryption**: ~5ms for typical data sizes
- **Audit logging**: Asynchronous with buffering
- **Resource monitoring**: Background thread with 1-second intervals

## Migration Guide

To add security to existing QEM-Bench code:

1. **Add authentication decorators** to sensitive functions
2. **Replace direct inputs** with validated inputs
3. **Use credential manager** instead of hardcoded secrets
4. **Add resource limits** to prevent DoS attacks
5. **Enable audit logging** for compliance

### Example Migration

Before:
```python
def execute_circuit(circuit, backend, shots=1000):
    # Execute quantum circuit
    return backend.run(circuit, shots=shots)
```

After:
```python
@secure_operation(
    permissions=[Permission.EXECUTE_CIRCUIT],
    validate_inputs={"shots": InputType.INTEGER},
    rate_limit_operation="circuit_execution",
    resource_limits={ResourceType.QUBITS: 50}
)
def execute_circuit(circuit, backend, shots=1000, user_id=None):
    # Execute quantum circuit with security
    return backend.run(circuit, shots=shots)
```

## Troubleshooting

### Common Issues

1. **ImportError**: Install required dependencies
   ```bash
   pip install cryptography psutil
   ```

2. **Permission Denied**: Check user has required permissions
   ```python
   access_control.check_permission(user_id, Permission.EXECUTE_CIRCUIT)
   ```

3. **Rate Limit Exceeded**: Reduce request frequency or increase limits
   ```python
   # Check current rate limit status
   result = access_control.check_rate_limit(user_id, "operation")
   ```

4. **Resource Limit Exceeded**: Reduce resource usage or increase limits
   ```python
   # Check current resource usage
   usage = resource_limiter.get_current_usage(ResourceType.MEMORY)
   ```

### Debug Mode

Enable debug logging to troubleshoot security issues:

```python
import logging
logging.getLogger('qem_bench_audit').setLevel(logging.DEBUG)
```

## Security Updates

The security framework is regularly updated to address new threats:

- Monitor security advisories
- Update dependencies regularly
- Review audit logs for suspicious activity
- Test security configurations periodically

## Contributing

When contributing to the security framework:

1. Follow secure coding practices
2. Add comprehensive tests
3. Update documentation
4. Consider backward compatibility
5. Review security implications

For more information, see the [Contributing Guide](CONTRIBUTING.md).