"""
Cryptographic Utilities for QEM-Bench

This module provides cryptographic operations including:
- Symmetric encryption/decryption for sensitive data
- Secure random number generation for quantum experiments
- Hash functions for data integrity
- Key derivation and management
- Certificate validation
- Digital signatures
"""

import os
import hashlib
import hmac
import secrets
from typing import Optional, Any, Dict, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography import x509
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    warnings.warn("cryptography library not available. Some security features disabled.")

from ..errors import SecurityError


class HashAlgorithm(Enum):
    """Supported hash algorithms."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"


class EncryptionMode(Enum):
    """Supported encryption modes."""
    FERNET = "fernet"  # Symmetric encryption with authentication
    AES_GCM = "aes_gcm"  # AES with Galois/Counter Mode
    RSA_OAEP = "rsa_oaep"  # RSA with OAEP padding


@dataclass
class EncryptedData:
    """Container for encrypted data with metadata."""
    ciphertext: bytes
    mode: EncryptionMode
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    salt: Optional[bytes] = None
    key_id: Optional[str] = None


class SecureRandom:
    """
    Cryptographically secure random number generator.
    
    Provides high-quality randomness for quantum experiments and security operations.
    """
    
    def __init__(self, seed: Optional[bytes] = None):
        """
        Initialize secure random generator.
        
        Args:
            seed: Optional seed for deterministic randomness (for testing)
        """
        self.seed = seed
        if seed:
            warnings.warn("Using deterministic seed - not suitable for production")
    
    def random_bytes(self, n: int) -> bytes:
        """Generate n random bytes."""
        if self.seed:
            # For testing - not cryptographically secure
            import random
            random.seed(self.seed)
            return bytes(random.getrandbits(8) for _ in range(n))
        return secrets.token_bytes(n)
    
    def random_int(self, min_val: int = 0, max_val: int = None) -> int:
        """Generate a random integer in range [min_val, max_val)."""
        if max_val is None:
            max_val = 2**32
        
        if self.seed:
            import random
            random.seed(self.seed)
            return random.randint(min_val, max_val - 1)
        
        return secrets.randbelow(max_val - min_val) + min_val
    
    def random_float(self) -> float:
        """Generate a random float in range [0.0, 1.0)."""
        random_bytes = self.random_bytes(8)
        return int.from_bytes(random_bytes, 'big') / (2**64)
    
    def random_angles(self, count: int) -> List[float]:
        """Generate random angles for quantum gates."""
        import math
        return [self.random_float() * 2 * math.pi for _ in range(count)]
    
    def random_unitary_params(self, num_qubits: int) -> Dict[str, float]:
        """Generate random parameters for unitary gates."""
        import math
        num_params = 4**num_qubits - 1  # Parameters for SU(2^n)
        return {
            f'param_{i}': self.random_float() * 2 * math.pi
            for i in range(num_params)
        }


class CryptoUtils:
    """
    Cryptographic utilities for QEM-Bench security operations.
    
    Provides encryption, hashing, key management, and other cryptographic functions.
    """
    
    def __init__(self):
        """Initialize crypto utilities."""
        self.encryption_key: Optional[bytes] = None
        self.fernet_cipher: Optional[Any] = None
        self.key_cache: Dict[str, bytes] = {}
        
        if not CRYPTOGRAPHY_AVAILABLE:
            warnings.warn("Advanced cryptographic features not available")
    
    def generate_encryption_key(self) -> bytes:
        """Generate a new encryption key."""
        if CRYPTOGRAPHY_AVAILABLE:
            key = Fernet.generate_key()
            self.set_encryption_key(key)
            return key
        else:
            # Fallback to basic key generation
            key = secrets.token_bytes(32)
            self.encryption_key = key
            return key
    
    def set_encryption_key(self, key: bytes):
        """Set the encryption key."""
        self.encryption_key = key
        if CRYPTOGRAPHY_AVAILABLE:
            self.fernet_cipher = Fernet(key)
    
    def get_encryption_key(self) -> Optional[bytes]:
        """Get the current encryption key."""
        return self.encryption_key
    
    def encrypt(self, data: bytes, mode: EncryptionMode = EncryptionMode.FERNET) -> bytes:
        """
        Encrypt data using the specified mode.
        
        Args:
            data: Data to encrypt
            mode: Encryption mode to use
            
        Returns:
            Encrypted data
        """
        if not self.encryption_key:
            raise SecurityError("No encryption key set")
        
        if mode == EncryptionMode.FERNET:
            return self._encrypt_fernet(data)
        elif mode == EncryptionMode.AES_GCM:
            return self._encrypt_aes_gcm(data)
        else:
            raise SecurityError(f"Unsupported encryption mode: {mode}")
    
    def decrypt(self, encrypted_data: bytes, mode: EncryptionMode = EncryptionMode.FERNET) -> bytes:
        """
        Decrypt data using the specified mode.
        
        Args:
            encrypted_data: Encrypted data
            mode: Encryption mode used
            
        Returns:
            Decrypted data
        """
        if not self.encryption_key:
            raise SecurityError("No encryption key set")
        
        if mode == EncryptionMode.FERNET:
            return self._decrypt_fernet(encrypted_data)
        elif mode == EncryptionMode.AES_GCM:
            return self._decrypt_aes_gcm(encrypted_data)
        else:
            raise SecurityError(f"Unsupported encryption mode: {mode}")
    
    def _encrypt_fernet(self, data: bytes) -> bytes:
        """Encrypt using Fernet (symmetric encryption with authentication)."""
        if not CRYPTOGRAPHY_AVAILABLE or not self.fernet_cipher:
            raise SecurityError("Fernet encryption not available")
        
        return self.fernet_cipher.encrypt(data)
    
    def _decrypt_fernet(self, encrypted_data: bytes) -> bytes:
        """Decrypt using Fernet."""
        if not CRYPTOGRAPHY_AVAILABLE or not self.fernet_cipher:
            raise SecurityError("Fernet decryption not available")
        
        return self.fernet_cipher.decrypt(encrypted_data)
    
    def _encrypt_aes_gcm(self, data: bytes) -> bytes:
        """Encrypt using AES-GCM."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise SecurityError("AES-GCM encryption not available")
        
        # Generate random IV
        iv = os.urandom(12)  # 96-bit IV for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.encryption_key[:32]),  # Use first 32 bytes for AES-256
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Return IV + tag + ciphertext
        return iv + encryptor.tag + ciphertext
    
    def _decrypt_aes_gcm(self, encrypted_data: bytes) -> bytes:
        """Decrypt using AES-GCM."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise SecurityError("AES-GCM decryption not available")
        
        if len(encrypted_data) < 28:  # IV (12) + tag (16)
            raise SecurityError("Invalid encrypted data length")
        
        # Extract components
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.encryption_key[:32]),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt data
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def hash_data(
        self,
        data: bytes,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        salt: Optional[bytes] = None
    ) -> bytes:
        """
        Hash data using the specified algorithm.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm to use
            salt: Optional salt for the hash
            
        Returns:
            Hash digest
        """
        if salt:
            data = salt + data
        
        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).digest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).digest()
        elif algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256(data).digest()
        elif algorithm == HashAlgorithm.SHA3_512:
            return hashlib.sha3_512(data).digest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).digest()
        else:
            raise SecurityError(f"Unsupported hash algorithm: {algorithm}")
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Hash a password with salt using PBKDF2.
        
        Args:
            password: Password to hash
            salt: Optional salt (generated if None)
            
        Returns:
            Tuple of (hash, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        if CRYPTOGRAPHY_AVAILABLE:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            password_hash = kdf.derive(password.encode('utf-8'))
        else:
            # Fallback implementation
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000
            )
        
        return password_hash, salt
    
    def verify_password(self, password: str, password_hash: bytes, salt: bytes) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Password to verify
            password_hash: Stored password hash
            salt: Salt used for hashing
            
        Returns:
            True if password is correct
        """
        computed_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, password_hash)
    
    def generate_hmac(
        self,
        data: bytes,
        key: Optional[bytes] = None,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> bytes:
        """
        Generate HMAC for data integrity.
        
        Args:
            data: Data to authenticate
            key: HMAC key (uses encryption key if None)
            algorithm: Hash algorithm for HMAC
            
        Returns:
            HMAC digest
        """
        if key is None:
            key = self.encryption_key
        
        if not key:
            raise SecurityError("No key available for HMAC")
        
        if algorithm == HashAlgorithm.SHA256:
            return hmac.new(key, data, hashlib.sha256).digest()
        elif algorithm == HashAlgorithm.SHA512:
            return hmac.new(key, data, hashlib.sha512).digest()
        else:
            raise SecurityError(f"Unsupported HMAC algorithm: {algorithm}")
    
    def verify_hmac(
        self,
        data: bytes,
        signature: bytes,
        key: Optional[bytes] = None,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> bool:
        """
        Verify HMAC signature.
        
        Args:
            data: Original data
            signature: HMAC signature to verify
            key: HMAC key (uses encryption key if None)
            algorithm: Hash algorithm used
            
        Returns:
            True if signature is valid
        """
        expected_signature = self.generate_hmac(data, key, algorithm)
        return hmac.compare_digest(signature, expected_signature)
    
    def derive_key(
        self,
        master_key: bytes,
        salt: bytes,
        info: bytes = b"",
        length: int = 32
    ) -> bytes:
        """
        Derive a key using HKDF (HMAC-based Key Derivation Function).
        
        Args:
            master_key: Master key material
            salt: Salt value
            info: Optional context information
            length: Desired key length
            
        Returns:
            Derived key
        """
        if CRYPTOGRAPHY_AVAILABLE:
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=length,
                salt=salt,
                info=info,
                backend=default_backend()
            )
            return hkdf.derive(master_key)
        else:
            # Simple fallback using HMAC
            return hmac.new(master_key, salt + info, hashlib.sha256).digest()[:length]
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """
        Generate RSA key pair.
        
        Args:
            key_size: RSA key size in bits
            
        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise SecurityError("RSA key generation not available")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def validate_certificate(self, cert_pem: bytes, ca_cert_pem: Optional[bytes] = None) -> bool:
        """
        Validate an X.509 certificate.
        
        Args:
            cert_pem: Certificate in PEM format
            ca_cert_pem: Optional CA certificate for validation
            
        Returns:
            True if certificate is valid
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            warnings.warn("Certificate validation not available")
            return True
        
        try:
            cert = x509.load_pem_x509_certificate(cert_pem, default_backend())
            
            # Check expiration
            from datetime import datetime
            now = datetime.utcnow()
            if now < cert.not_valid_before or now > cert.not_valid_after:
                return False
            
            # If CA certificate provided, verify signature
            if ca_cert_pem:
                ca_cert = x509.load_pem_x509_certificate(ca_cert_pem, default_backend())
                ca_public_key = ca_cert.public_key()
                
                try:
                    ca_public_key.verify(
                        cert.signature,
                        cert.tbs_certificate_bytes,
                        padding.PKCS1v15(),
                        cert.signature_hash_algorithm
                    )
                except Exception:
                    return False
            
            return True
            
        except Exception as e:
            warnings.warn(f"Certificate validation error: {e}")
            return False
    
    def constant_time_compare(self, a: bytes, b: bytes) -> bool:
        """
        Constant-time comparison of byte strings.
        
        Args:
            a: First byte string
            b: Second byte string
            
        Returns:
            True if strings are equal
        """
        return hmac.compare_digest(a, b)
    
    def secure_delete(self, data: bytearray):
        """
        Securely delete sensitive data from memory.
        
        Args:
            data: Data to securely delete
        """
        if isinstance(data, bytearray):
            # Overwrite with random data
            for i in range(len(data)):
                data[i] = secrets.randbits(8)
            # Clear the array
            data.clear()
        else:
            warnings.warn("Secure delete only works with bytearray objects")
    
    def generate_session_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure session token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Hex-encoded token
        """
        return secrets.token_hex(length)
    
    def generate_api_key(self, prefix: str = "qem", length: int = 24) -> str:
        """
        Generate an API key with prefix.
        
        Args:
            prefix: Key prefix
            length: Random part length in bytes
            
        Returns:
            API key string
        """
        random_part = secrets.token_urlsafe(length)
        return f"{prefix}_{random_part}"


# Global crypto utils instance
_global_crypto_utils: Optional[CryptoUtils] = None


def get_crypto_utils() -> CryptoUtils:
    """Get the global crypto utils instance."""
    global _global_crypto_utils
    if _global_crypto_utils is None:
        _global_crypto_utils = CryptoUtils()
    return _global_crypto_utils


def set_crypto_utils(crypto_utils: CryptoUtils):
    """Set the global crypto utils instance."""
    global _global_crypto_utils
    _global_crypto_utils = crypto_utils