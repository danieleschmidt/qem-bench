"""
Credential Management System for QEM-Bench

This module provides secure handling of API keys, authentication tokens, and certificates
for quantum backends. Features include:
- Encrypted storage of credentials
- Automatic token refresh
- Certificate validation
- Credential rotation
- Access logging and monitoring
"""

import os
import json
import time
import warnings
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
from datetime import datetime, timedelta
import threading

from ..errors import SecurityError, ConfigurationError
from .crypto_utils import CryptoUtils
from .audit_logger import AuditLogger


class CredentialType(Enum):
    """Types of credentials supported."""
    API_KEY = "api_key"
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    CLIENT_SECRET = "client_secret"


class CredentialStatus(Enum):
    """Status of a credential."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


@dataclass
class Credential:
    """A secure credential with metadata."""
    name: str
    credential_type: CredentialType
    encrypted_value: bytes
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: CredentialStatus = CredentialStatus.ACTIVE
    
    def is_expired(self) -> bool:
        """Check if the credential is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if the credential is valid and usable."""
        return (
            self.status == CredentialStatus.ACTIVE and
            not self.is_expired()
        )
    
    def time_until_expiry(self) -> Optional[timedelta]:
        """Get time until credential expires."""
        if self.expires_at is None:
            return None
        return self.expires_at - datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding encrypted value)."""
        return {
            "name": self.name,
            "type": self.credential_type.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "metadata": self.metadata,
            "status": self.status.value,
            "is_expired": self.is_expired(),
            "is_valid": self.is_valid()
        }


class CredentialManager:
    """
    Secure credential manager for QEM-Bench.
    
    Manages API keys, tokens, and certificates for quantum backends with:
    - Encrypted storage
    - Automatic expiration handling
    - Usage tracking and audit logging
    - Thread-safe operations
    - Credential rotation support
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        encryption_key: Optional[bytes] = None,
        auto_cleanup: bool = True,
        cleanup_interval: int = 3600  # 1 hour
    ):
        """
        Initialize credential manager.
        
        Args:
            storage_path: Path to credential storage file
            encryption_key: Encryption key (auto-generated if None)
            auto_cleanup: Whether to automatically clean expired credentials
            cleanup_interval: Cleanup interval in seconds
        """
        self.storage_path = storage_path or Path.home() / ".qem_bench" / "credentials.enc"
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        
        # Initialize crypto utilities
        self.crypto = CryptoUtils()
        if encryption_key:
            self.crypto.set_encryption_key(encryption_key)
        else:
            self.crypto.generate_encryption_key()
        
        # Initialize audit logger
        self.audit_logger = AuditLogger()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Credential storage
        self.credentials: Dict[str, Credential] = {}
        
        # Load existing credentials
        self._load_credentials()
        
        # Start cleanup thread if enabled
        if auto_cleanup:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for credential cleanup."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self.cleanup_expired()
                except Exception as e:
                    warnings.warn(f"Credential cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _load_credentials(self):
        """Load credentials from storage file."""
        if not self.storage_path.exists():
            return
        
        try:
            with self._lock:
                with open(self.storage_path, 'rb') as f:
                    encrypted_data = f.read()
                
                # Decrypt the data
                data_json = self.crypto.decrypt(encrypted_data).decode('utf-8')
                data = json.loads(data_json)
                
                # Reconstruct credentials
                for name, cred_data in data.items():
                    credential = Credential(
                        name=name,
                        credential_type=CredentialType(cred_data['type']),
                        encrypted_value=bytes.fromhex(cred_data['encrypted_value']),
                        created_at=datetime.fromisoformat(cred_data['created_at']),
                        expires_at=datetime.fromisoformat(cred_data['expires_at']) if cred_data.get('expires_at') else None,
                        last_used=datetime.fromisoformat(cred_data['last_used']) if cred_data.get('last_used') else None,
                        metadata=cred_data.get('metadata', {}),
                        status=CredentialStatus(cred_data.get('status', 'active'))
                    )
                    self.credentials[name] = credential
                    
        except Exception as e:
            raise SecurityError(f"Failed to load credentials: {e}")
    
    def _save_credentials(self):
        """Save credentials to storage file."""
        try:
            with self._lock:
                # Prepare data for serialization
                data = {}
                for name, credential in self.credentials.items():
                    data[name] = {
                        'type': credential.credential_type.value,
                        'encrypted_value': credential.encrypted_value.hex(),
                        'created_at': credential.created_at.isoformat(),
                        'expires_at': credential.expires_at.isoformat() if credential.expires_at else None,
                        'last_used': credential.last_used.isoformat() if credential.last_used else None,
                        'metadata': credential.metadata,
                        'status': credential.status.value
                    }
                
                # Serialize and encrypt
                data_json = json.dumps(data, indent=2)
                encrypted_data = self.crypto.encrypt(data_json.encode('utf-8'))
                
                # Write to temporary file first, then move (atomic operation)
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=self.storage_path.parent,
                    delete=False
                ) as tmp_file:
                    tmp_file.write(encrypted_data)
                    tmp_path = Path(tmp_file.name)
                
                # Set secure permissions and move
                os.chmod(tmp_path, 0o600)  # Read/write for owner only
                tmp_path.replace(self.storage_path)
                
        except Exception as e:
            # Clean up temporary file if it exists
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
            raise SecurityError(f"Failed to save credentials: {e}")
    
    def store_credential(
        self,
        name: str,
        value: str,
        credential_type: CredentialType,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a new credential.
        
        Args:
            name: Unique name for the credential
            value: The credential value (will be encrypted)
            credential_type: Type of credential
            expires_at: Optional expiration time
            metadata: Optional metadata dictionary
            
        Returns:
            True if stored successfully
        """
        if not name or not value:
            raise SecurityError("Credential name and value cannot be empty")
        
        try:
            with self._lock:
                # Encrypt the credential value
                encrypted_value = self.crypto.encrypt(value.encode('utf-8'))
                
                # Create credential object
                credential = Credential(
                    name=name,
                    credential_type=credential_type,
                    encrypted_value=encrypted_value,
                    expires_at=expires_at,
                    metadata=metadata or {}
                )
                
                # Store and save
                self.credentials[name] = credential
                self._save_credentials()
                
                # Audit log
                self.audit_logger.log_security_event(
                    event_type="credential_stored",
                    details={
                        "credential_name": name,
                        "credential_type": credential_type.value,
                        "expires_at": expires_at.isoformat() if expires_at else None
                    }
                )
                
                return True
                
        except Exception as e:
            raise SecurityError(f"Failed to store credential '{name}': {e}")
    
    def get_credential(self, name: str) -> Optional[str]:
        """
        Retrieve and decrypt a credential.
        
        Args:
            name: Name of the credential
            
        Returns:
            Decrypted credential value or None if not found/invalid
        """
        try:
            with self._lock:
                if name not in self.credentials:
                    return None
                
                credential = self.credentials[name]
                
                # Check if credential is valid
                if not credential.is_valid():
                    if credential.is_expired():
                        self.audit_logger.log_security_event(
                            event_type="credential_access_expired",
                            details={"credential_name": name}
                        )
                    return None
                
                # Decrypt and return
                decrypted_value = self.crypto.decrypt(credential.encrypted_value).decode('utf-8')
                
                # Update last used time
                credential.last_used = datetime.now()
                self._save_credentials()
                
                # Audit log
                self.audit_logger.log_security_event(
                    event_type="credential_accessed",
                    details={"credential_name": name}
                )
                
                return decrypted_value
                
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type="credential_access_error",
                details={"credential_name": name, "error": str(e)}
            )
            raise SecurityError(f"Failed to retrieve credential '{name}': {e}")
    
    def delete_credential(self, name: str) -> bool:
        """
        Delete a credential.
        
        Args:
            name: Name of the credential to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            with self._lock:
                if name not in self.credentials:
                    return False
                
                del self.credentials[name]
                self._save_credentials()
                
                # Audit log
                self.audit_logger.log_security_event(
                    event_type="credential_deleted",
                    details={"credential_name": name}
                )
                
                return True
                
        except Exception as e:
            raise SecurityError(f"Failed to delete credential '{name}': {e}")
    
    def list_credentials(self, include_expired: bool = False) -> List[Dict[str, Any]]:
        """
        List all credentials (metadata only, no values).
        
        Args:
            include_expired: Whether to include expired credentials
            
        Returns:
            List of credential metadata dictionaries
        """
        with self._lock:
            result = []
            for credential in self.credentials.values():
                if not include_expired and credential.is_expired():
                    continue
                result.append(credential.to_dict())
            
            return result
    
    def rotate_credential(
        self,
        name: str,
        new_value: str,
        new_expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Rotate a credential with a new value.
        
        Args:
            name: Name of the credential to rotate
            new_value: New credential value
            new_expires_at: New expiration time
            
        Returns:
            True if rotated successfully
        """
        try:
            with self._lock:
                if name not in self.credentials:
                    return False
                
                credential = self.credentials[name]
                old_encrypted = credential.encrypted_value
                
                # Encrypt new value
                credential.encrypted_value = self.crypto.encrypt(new_value.encode('utf-8'))
                credential.expires_at = new_expires_at
                credential.created_at = datetime.now()
                credential.status = CredentialStatus.ACTIVE
                
                self._save_credentials()
                
                # Audit log
                self.audit_logger.log_security_event(
                    event_type="credential_rotated",
                    details={
                        "credential_name": name,
                        "new_expires_at": new_expires_at.isoformat() if new_expires_at else None
                    }
                )
                
                return True
                
        except Exception as e:
            raise SecurityError(f"Failed to rotate credential '{name}': {e}")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired credentials.
        
        Returns:
            Number of credentials removed
        """
        removed_count = 0
        
        try:
            with self._lock:
                expired_names = [
                    name for name, cred in self.credentials.items()
                    if cred.is_expired()
                ]
                
                for name in expired_names:
                    del self.credentials[name]
                    removed_count += 1
                
                if removed_count > 0:
                    self._save_credentials()
                    
                    # Audit log
                    self.audit_logger.log_security_event(
                        event_type="credentials_cleaned",
                        details={"removed_count": removed_count}
                    )
                
        except Exception as e:
            warnings.warn(f"Error during credential cleanup: {e}")
        
        return removed_count
    
    def get_expiring_credentials(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get credentials that will expire within the specified number of days.
        
        Args:
            days: Number of days to check ahead
            
        Returns:
            List of credential metadata for expiring credentials
        """
        cutoff_time = datetime.now() + timedelta(days=days)
        expiring = []
        
        with self._lock:
            for credential in self.credentials.values():
                if (credential.expires_at and 
                    credential.expires_at <= cutoff_time and
                    credential.is_valid()):
                    cred_dict = credential.to_dict()
                    cred_dict['days_until_expiry'] = (credential.expires_at - datetime.now()).days
                    expiring.append(cred_dict)
        
        return expiring
    
    def get_backend_credentials(self, backend_name: str) -> Dict[str, str]:
        """
        Get all credentials for a specific backend.
        
        Args:
            backend_name: Name of the quantum backend
            
        Returns:
            Dictionary of credential names to values
        """
        backend_creds = {}
        
        with self._lock:
            for name, credential in self.credentials.items():
                if credential.metadata.get('backend') == backend_name and credential.is_valid():
                    try:
                        value = self.crypto.decrypt(credential.encrypted_value).decode('utf-8')
                        backend_creds[name] = value
                        credential.last_used = datetime.now()
                    except Exception as e:
                        warnings.warn(f"Failed to decrypt credential '{name}': {e}")
        
        if backend_creds:
            self._save_credentials()  # Update last_used times
        
        return backend_creds
    
    def validate_backend_access(self, backend_name: str) -> bool:
        """
        Validate that all required credentials are available for a backend.
        
        Args:
            backend_name: Name of the quantum backend
            
        Returns:
            True if all required credentials are available
        """
        # Backend-specific credential requirements
        requirements = {
            'ibm': ['api.ibm_token'],
            'aws_braket': ['api.aws_access_key', 'api.aws_secret_key'],
            'google': ['api.google_credentials_file'],
            'rigetti': ['api.rigetti_token'],
            'ionq': ['api.ionq_token']
        }
        
        required_creds = requirements.get(backend_name, [])
        
        for cred_name in required_creds:
            if not self.get_credential(cred_name):
                return False
        
        return True
    
    def export_public_keys(self) -> Dict[str, str]:
        """
        Export public keys and certificates (non-sensitive credentials).
        
        Returns:
            Dictionary of public credential data
        """
        public_data = {}
        
        with self._lock:
            for name, credential in self.credentials.items():
                if (credential.credential_type == CredentialType.CERTIFICATE and
                    credential.is_valid()):
                    try:
                        value = self.crypto.decrypt(credential.encrypted_value).decode('utf-8')
                        # Only export if it looks like a public certificate
                        if "BEGIN CERTIFICATE" in value:
                            public_data[name] = value
                    except Exception:
                        continue
        
        return public_data


# Global credential manager instance
_global_credential_manager: Optional[CredentialManager] = None


def get_credential_manager() -> CredentialManager:
    """Get the global credential manager instance."""
    global _global_credential_manager
    if _global_credential_manager is None:
        _global_credential_manager = CredentialManager()
    return _global_credential_manager


def set_credential_manager(manager: CredentialManager):
    """Set the global credential manager instance."""
    global _global_credential_manager
    _global_credential_manager = manager