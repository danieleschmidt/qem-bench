#!/usr/bin/env python3
"""
QEM-Bench Security Framework Example

This example demonstrates how to use the comprehensive security features
in QEM-Bench for secure quantum error mitigation.
"""

import numpy as np
from datetime import datetime, timedelta

# Import QEM-Bench security components
from qem_bench.security import (
    # Core security components
    SecureConfig, CredentialManager, InputSanitizer, ResourceLimiter,
    AccessControl, AuditLogger, SecurityPolicy,
    
    # Types and enums
    CredentialType, Permission, Role, SecurityLevel, InputType,
    AuditEventType, AuditLevel,
    
    # Utility functions
    get_default_policy, apply_policy,
    
    # Security decorators
    secure_operation, require_authentication, validate_input,
    rate_limit, audit_log, circuit_security
)

# Import secure mitigation example
from qem_bench.security.secure_mitigation import SecureZeroNoiseExtrapolation


def setup_security_framework():
    """Setup the QEM-Bench security framework with example configuration."""
    
    print("Setting up QEM-Bench Security Framework...")
    
    # 1. Apply security policy
    print("\n1. Configuring security policy...")
    policy = get_default_policy()
    print(f"Applied policy: {policy.name} (Level: {policy.security_level.value})")
    apply_policy(policy)
    
    # 2. Configure secure settings
    print("\n2. Setting up secure configuration...")
    config = SecureConfig()
    
    # Set some example configuration values
    config.set("security.enable_encryption", True)
    config.set("security.enable_audit_logging", True)
    config.set("limits.max_qubits", 50)
    config.set("limits.execution_timeout", 300)
    
    print("Configuration applied successfully")
    
    # 3. Setup credential management
    print("\n3. Setting up credential management...")
    cred_manager = CredentialManager()
    
    # Store example credentials (in production, these would come from secure sources)
    cred_manager.store_credential(
        name="ibm_token",
        value="example_ibm_token_12345",
        credential_type=CredentialType.API_KEY,
        expires_at=datetime.now() + timedelta(days=30),
        metadata={"backend": "ibm", "description": "IBM Quantum API token"}
    )
    
    cred_manager.store_credential(
        name="aws_access_key",
        value="example_aws_key_67890",
        credential_type=CredentialType.ACCESS_TOKEN,
        expires_at=datetime.now() + timedelta(days=90),
        metadata={"backend": "aws_braket", "description": "AWS Braket access key"}
    )
    
    print(f"Stored {len(cred_manager.list_credentials())} credentials")
    
    # 4. Setup access control
    print("\n4. Setting up access control...")
    access_control = AccessControl()
    
    # Create example users with different roles
    researcher = access_control.create_user(
        "researcher_001", 
        "Dr. Alice Quantum", 
        {Role.RESEARCHER}
    )
    
    student = access_control.create_user(
        "student_001", 
        "Bob Graduate", 
        {Role.USER}
    )
    
    admin = access_control.create_user(
        "admin_001", 
        "Charlie Admin", 
        {Role.ADMIN}
    )
    
    print(f"Created {len([researcher, student, admin])} users")
    
    # 5. Initialize audit logging
    print("\n5. Setting up audit logging...")
    audit_logger = AuditLogger(enable_console=True, enable_syslog=False)
    
    # Log system startup
    audit_logger.log_security_event(
        event_type=AuditEventType.SYSTEM_STARTUP,
        level=AuditLevel.INFO,
        details={"component": "security_framework", "version": "1.0"}
    )
    
    print("Security framework setup complete!")
    
    return {
        'config': config,
        'cred_manager': cred_manager,
        'access_control': access_control,
        'audit_logger': audit_logger
    }


@secure_operation(
    permissions=[Permission.CREATE_CIRCUIT],
    validate_inputs={
        'circuit_name': InputType.CIRCUIT_NAME,
        'num_qubits': InputType.INTEGER,
        'depth': InputType.INTEGER
    },
    rate_limit_operation="circuit_creation",
    audit_event=AuditEventType.DATA_IMPORTED
)
def create_secure_circuit(circuit_name: str, num_qubits: int, depth: int, user_id: str):
    """Example function demonstrating comprehensive security integration."""
    
    print(f"\nCreating secure circuit '{circuit_name}' for user {user_id}")
    print(f"Parameters: {num_qubits} qubits, depth {depth}")
    
    # Simulate circuit creation
    circuit_data = {
        'name': circuit_name,
        'num_qubits': num_qubits,
        'depth': depth,
        'created_by': user_id,
        'created_at': datetime.now().isoformat()
    }
    
    return circuit_data


@circuit_security(max_qubits=25, max_shots=10000)
@validate_input(shots=InputType.INTEGER)
@rate_limit("circuit_execution", requests_per_minute=5)
@audit_log(AuditEventType.CIRCUIT_EXECUTED, include_args=True)
def execute_secure_circuit(circuit_data: dict, shots: int, user_id: str):
    """Example function showing multiple security decorators."""
    
    print(f"\nExecuting circuit '{circuit_data['name']}' with {shots} shots")
    
    # Simulate circuit execution
    np.random.seed(42)  # For reproducible results
    expectation_value = np.random.normal(0.5, 0.1)
    
    result = {
        'circuit_name': circuit_data['name'],
        'expectation_value': expectation_value,
        'shots': shots,
        'executed_by': user_id,
        'execution_time': datetime.now().isoformat()
    }
    
    return result


def demonstrate_input_sanitization():
    """Demonstrate input sanitization capabilities."""
    
    print("\n" + "="*50)
    print("DEMONSTRATING INPUT SANITIZATION")
    print("="*50)
    
    sanitizer = InputSanitizer()
    
    # Test various input types
    test_inputs = [
        ("circuit_name", "my_quantum_circuit", InputType.CIRCUIT_NAME),
        ("num_qubits", "10", InputType.INTEGER),
        ("angle", "3.14159", InputType.FLOAT),
        ("backend", "ibm_simulator", InputType.BACKEND_NAME),
        ("filename", "results.json", InputType.FILENAME)
    ]
    
    print("\nSanitizing various inputs:")
    for name, value, input_type in test_inputs:
        try:
            sanitized = sanitizer.sanitize(value, input_type)
            print(f"  {name}: '{value}' -> {sanitized} ({type(sanitized).__name__})")
        except Exception as e:
            print(f"  {name}: '{value}' -> ERROR: {e}")
    
    # Test circuit parameter validation
    print("\nValidating circuit parameters:")
    params = {
        "num_qubits": 5,
        "depth": 10,
        "shots": 1000,
        "name": "test_circuit",
        "backend": "simulator"
    }
    
    try:
        validated_params = sanitizer.validate_circuit_parameters(params)
        print("  Circuit parameters validated successfully:")
        for key, value in validated_params.items():
            print(f"    {key}: {value}")
    except Exception as e:
        print(f"  Validation error: {e}")


def demonstrate_resource_management():
    """Demonstrate resource management and limiting."""
    
    print("\n" + "="*50)
    print("DEMONSTRATING RESOURCE MANAGEMENT")
    print("="*50)
    
    from qem_bench.security import ResourceType, ResourceLimiter
    
    limiter = ResourceLimiter(enable_monitoring=False)  # Disable monitoring for demo
    
    print("\nTesting resource allocation:")
    
    # Test memory allocation
    try:
        print("  Allocating 100MB memory...")
        success = limiter.allocate_resource(ResourceType.MEMORY, 100 * 1024 * 1024)
        print(f"  Memory allocation: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            print("  Releasing memory...")
            limiter.release_resource(ResourceType.MEMORY, 100 * 1024 * 1024)
            print("  Memory released successfully")
    except Exception as e:
        print(f"  Memory allocation error: {e}")
    
    # Test qubit allocation
    try:
        print("  Allocating 25 qubits...")
        success = limiter.allocate_resource(ResourceType.QUBITS, 25)
        print(f"  Qubit allocation: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            limiter.release_resource(ResourceType.QUBITS, 25)
            print("  Qubits released successfully")
    except Exception as e:
        print(f"  Qubit allocation error: {e}")
    
    # Show resource statistics
    print("\nCurrent system resource summary:")
    summary = limiter.get_system_summary()
    print(f"  Active operations: {summary.get('active_operations', 0)}")
    print(f"  Active sessions: {summary.get('active_sessions', 0)}")


def demonstrate_secure_mitigation():
    """Demonstrate secure quantum error mitigation."""
    
    print("\n" + "="*50)
    print("DEMONSTRATING SECURE MITIGATION")
    print("="*50)
    
    # Create a secure ZNE instance
    secure_zne = SecureZeroNoiseExtrapolation(
        user_id="researcher_001",
        noise_factors=[1.0, 1.5, 2.0],
        enable_security=True
    )
    
    print(f"Created secure ZNE with noise factors: {secure_zne.noise_factors}")
    
    # Simulate a quantum circuit (mock object for demo)
    class MockCircuit:
        def __init__(self, num_qubits, depth):
            self.num_qubits = num_qubits
            self.depth = depth
    
    class MockBackend:
        def run(self, circuit, shots=1000):
            # Simulate noisy measurement
            return np.random.normal(0.5, 0.1)
    
    class MockObservable:
        def __init__(self, name):
            self.name = name
    
    # Create mock quantum objects
    circuit = MockCircuit(num_qubits=5, depth=10)
    backend = MockBackend()
    observable = MockObservable("Z")
    
    print(f"Mock circuit: {circuit.num_qubits} qubits, depth {circuit.depth}")
    
    # Get ZNE statistics (demonstrates secure method access)
    try:
        stats = secure_zne.get_statistics()
        print("ZNE configuration:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Statistics access error: {e}")


def demonstrate_credential_management(cred_manager):
    """Demonstrate credential management capabilities."""
    
    print("\n" + "="*50)
    print("DEMONSTRATING CREDENTIAL MANAGEMENT")
    print("="*50)
    
    # List all credentials
    credentials = cred_manager.list_credentials()
    print(f"Total credentials stored: {len(credentials)}")
    
    for cred in credentials:
        print(f"  {cred['name']}: {cred['type']} (expires: {cred.get('expires_at', 'never')})")
    
    # Retrieve a specific credential
    print("\nRetrieving IBM token...")
    ibm_token = cred_manager.get_credential("ibm_token")
    if ibm_token:
        print(f"  Token retrieved: {ibm_token[:10]}...")  # Only show first 10 chars
    else:
        print("  Token not found or expired")
    
    # Check expiring credentials
    expiring = cred_manager.get_expiring_credentials(days=60)
    print(f"\nCredentials expiring in 60 days: {len(expiring)}")
    for cred in expiring:
        print(f"  {cred['name']}: expires in {cred.get('days_until_expiry', '?')} days")


def main():
    """Main demonstration function."""
    
    print("QEM-Bench Security Framework Demonstration")
    print("==========================================\n")
    
    try:
        # Setup security framework
        security_components = setup_security_framework()
        
        # Demonstrate input sanitization
        demonstrate_input_sanitization()
        
        # Demonstrate resource management
        demonstrate_resource_management()
        
        # Demonstrate access control and secure operations
        print("\n" + "="*50)
        print("DEMONSTRATING SECURE OPERATIONS")
        print("="*50)
        
        # Create a circuit with security
        circuit_data = create_secure_circuit(
            circuit_name="bell_state",
            num_qubits=2,
            depth=5,
            user_id="researcher_001"
        )
        
        # Execute the circuit with security
        result = execute_secure_circuit(
            circuit_data=circuit_data,
            shots=1000,
            user_id="researcher_001"
        )
        
        print(f"Execution result: {result['expectation_value']:.4f}")
        
        # Demonstrate secure mitigation
        demonstrate_secure_mitigation()
        
        # Demonstrate credential management
        demonstrate_credential_management(security_components['cred_manager'])
        
        # Show final audit log statistics
        print("\n" + "="*50)
        print("AUDIT LOG STATISTICS")
        print("="*50)
        
        audit_stats = security_components['audit_logger'].get_statistics()
        print(f"Total events logged: {audit_stats.get('total_events', 0)}")
        print(f"Buffered events: {audit_stats.get('buffered_events', 0)}")
        print(f"Error count: {audit_stats.get('error_count', 0)}")
        
        print("\n" + "="*50)
        print("DEMONSTRATION COMPLETE")
        print("="*50)
        print("\nThe QEM-Bench Security Framework provides comprehensive")
        print("protection for quantum computing workloads while maintaining")
        print("ease of use and backward compatibility.")
        
    except Exception as e:
        print(f"\nDemonstration error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()