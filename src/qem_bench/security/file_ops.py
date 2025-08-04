"""
Secure File Operations for QEM-Bench

This module provides secure file operations including:
- Safe file path validation and sanitization
- Secure temporary file handling
- Permission checks and access control
- File integrity verification
- Secure deletion of sensitive files
- Directory traversal protection
"""

import os
import tempfile
import shutil
import hashlib
import stat
import warnings
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading
import time

from ..errors import SecurityError, ValidationError
from .crypto_utils import CryptoUtils
from .input_sanitizer import InputSanitizer
from .audit_logger import AuditLogger, AuditEventType, AuditLevel


class FilePermission(Enum):
    """File permission levels."""
    OWNER_READ = "owner_read"
    OWNER_WRITE = "owner_write"
    OWNER_EXECUTE = "owner_execute"
    GROUP_READ = "group_read"
    GROUP_WRITE = "group_write"
    GROUP_EXECUTE = "group_execute"
    OTHER_READ = "other_read"
    OTHER_WRITE = "other_write"
    OTHER_EXECUTE = "other_execute"


@dataclass
class FileMetadata:
    """File metadata for security tracking."""
    path: Path
    size: int
    created_at: float
    modified_at: float
    permissions: int
    checksum: Optional[str] = None
    encrypted: bool = False
    temporary: bool = False


class SecureFileOperations:
    """
    Secure file operations manager for QEM-Bench.
    
    Provides safe file operations with path validation, permission checks,
    and security monitoring.
    """
    
    def __init__(
        self,
        base_directory: Optional[Path] = None,
        enable_encryption: bool = True,
        enable_checksums: bool = True,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        allowed_extensions: Optional[List[str]] = None
    ):
        """
        Initialize secure file operations.
        
        Args:
            base_directory: Base directory for file operations
            enable_encryption: Whether to support file encryption
            enable_checksums: Whether to calculate file checksums
            max_file_size: Maximum allowed file size
            allowed_extensions: List of allowed file extensions
        """
        self.base_directory = base_directory or Path.cwd()
        self.enable_encryption = enable_encryption
        self.enable_checksums = enable_checksums
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or [
            '.json', '.txt', '.log', '.csv', '.pkl', '.npz', '.h5', '.yaml', '.yml'
        ]
        
        # Initialize utilities
        self.crypto = CryptoUtils() if enable_encryption else None
        self.sanitizer = InputSanitizer()
        self.audit_logger = AuditLogger()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Temporary file tracking
        self.temp_files: Dict[str, FileMetadata] = {}
        
        # File metadata cache
        self.file_cache: Dict[str, FileMetadata] = {}
        
        # Ensure base directory exists and is secure
        self._setup_base_directory()
    
    def _setup_base_directory(self):
        """Setup and secure the base directory."""
        self.base_directory.mkdir(parents=True, exist_ok=True)
        
        # Set secure permissions (owner read/write/execute only)
        os.chmod(self.base_directory, 0o700)
    
    def validate_path(self, path: Union[str, Path], allow_absolute: bool = False) -> Path:
        """
        Validate and sanitize a file path.
        
        Args:
            path: File path to validate
            allow_absolute: Whether to allow absolute paths
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path is invalid or unsafe
        """
        if isinstance(path, str):
            # Sanitize string path
            path = self.sanitizer.sanitize(path, self.sanitizer.InputType.PATH)
            path = Path(path)
        
        # Check for path traversal
        path_str = str(path)
        if '..' in path_str or path_str.startswith('/'):
            if not allow_absolute:
                raise SecurityError(f"Path traversal or absolute path not allowed: {path}")
        
        # Check for null bytes
        if '\x00' in path_str:
            raise SecurityError("Null bytes in path not allowed")
        
        # Resolve path relative to base directory if not absolute
        if not path.is_absolute():
            path = self.base_directory / path
        
        # Final security check - ensure path is within base directory
        if not allow_absolute:
            try:
                path.resolve().relative_to(self.base_directory.resolve())
            except ValueError:
                raise SecurityError(f"Path outside base directory: {path}")
        
        # Check file extension
        if path.suffix and path.suffix.lower() not in self.allowed_extensions:
            warnings.warn(f"File extension not in allowed list: {path.suffix}")
        
        return path
    
    def check_permissions(self, path: Path, required_permissions: List[FilePermission]) -> bool:
        """
        Check if current user has required permissions on a file.
        
        Args:
            path: File path to check
            required_permissions: List of required permissions
            
        Returns:
            True if all permissions are available
        """
        if not path.exists():
            return False
        
        try:
            file_stat = path.stat()
            file_mode = file_stat.st_mode
            
            for permission in required_permissions:
                if permission == FilePermission.OWNER_READ and not (file_mode & stat.S_IRUSR):
                    return False
                elif permission == FilePermission.OWNER_WRITE and not (file_mode & stat.S_IWUSR):
                    return False
                elif permission == FilePermission.OWNER_EXECUTE and not (file_mode & stat.S_IXUSR):
                    return False
                elif permission == FilePermission.GROUP_READ and not (file_mode & stat.S_IRGRP):
                    return False
                elif permission == FilePermission.GROUP_WRITE and not (file_mode & stat.S_IWGRP):
                    return False
                elif permission == FilePermission.GROUP_EXECUTE and not (file_mode & stat.S_IXGRP):
                    return False
                elif permission == FilePermission.OTHER_READ and not (file_mode & stat.S_IROTH):
                    return False
                elif permission == FilePermission.OTHER_WRITE and not (file_mode & stat.S_IWOTH):
                    return False
                elif permission == FilePermission.OTHER_EXECUTE and not (file_mode & stat.S_IXOTH):
                    return False
            
            return True
            
        except Exception as e:
            warnings.warn(f"Permission check failed: {e}")
            return False
    
    def calculate_checksum(self, path: Path, algorithm: str = "sha256") -> str:
        """
        Calculate file checksum.
        
        Args:
            path: File path
            algorithm: Hash algorithm to use
            
        Returns:
            Hex-encoded checksum
        """
        if not path.exists():
            raise SecurityError(f"File not found: {path}")
        
        hash_func = hashlib.new(algorithm)
        
        try:
            with open(path, 'rb') as f:
                while chunk := f.read(8192):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            raise SecurityError(f"Checksum calculation failed: {e}")
    
    def verify_checksum(self, path: Path, expected_checksum: str, algorithm: str = "sha256") -> bool:
        """
        Verify file checksum.
        
        Args:
            path: File path
            expected_checksum: Expected checksum
            algorithm: Hash algorithm used
            
        Returns:
            True if checksum matches
        """
        try:
            actual_checksum = self.calculate_checksum(path, algorithm)
            return actual_checksum.lower() == expected_checksum.lower()
        except Exception:
            return False
    
    def secure_read(
        self,
        path: Union[str, Path],
        mode: str = 'r',
        encoding: str = 'utf-8',
        verify_checksum: Optional[str] = None,
        decrypt: bool = False
    ) -> Union[str, bytes]:
        """
        Securely read a file with validation.
        
        Args:
            path: File path
            mode: File open mode
            encoding: Text encoding (for text mode)
            verify_checksum: Optional checksum to verify
            decrypt: Whether to decrypt the file
            
        Returns:
            File contents
        """
        path = self.validate_path(path)
        
        # Check permissions
        if not self.check_permissions(path, [FilePermission.OWNER_READ]):
            raise SecurityError(f"Insufficient permissions to read: {path}")
        
        # Check file size
        if path.stat().st_size > self.max_file_size:
            raise SecurityError(f"File too large: {path.stat().st_size} > {self.max_file_size}")
        
        # Verify checksum if provided
        if verify_checksum and not self.verify_checksum(path, verify_checksum):
            raise SecurityError(f"Checksum verification failed: {path}")
        
        try:
            # Read file
            with open(path, mode, encoding=encoding if 'b' not in mode else None) as f:
                content = f.read()
            
            # Decrypt if requested
            if decrypt and self.crypto:
                if isinstance(content, str):
                    content = content.encode('utf-8')
                content = self.crypto.decrypt(content)
                if 'b' not in mode:
                    content = content.decode('utf-8')
            
            # Audit log
            self.audit_logger.log_security_event(
                event_type=AuditEventType.DATA_IMPORTED,
                details={'file_path': str(path), 'file_size': path.stat().st_size}
            )
            
            return content
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                level=AuditLevel.ERROR,
                details={'operation': 'file_read', 'path': str(path), 'error': str(e)}
            )
            raise SecurityError(f"File read failed: {e}")
    
    def secure_write(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        mode: str = 'w',
        encoding: str = 'utf-8',
        encrypt: bool = False,
        calculate_checksum: bool = None,
        atomic: bool = True
    ) -> Optional[str]:
        """
        Securely write content to a file.
        
        Args:
            path: File path
            content: Content to write
            mode: File open mode
            encoding: Text encoding (for text mode)
            encrypt: Whether to encrypt the content
            calculate_checksum: Whether to calculate checksum
            atomic: Whether to use atomic write (write to temp file first)
            
        Returns:
            File checksum if calculated
        """
        path = self.validate_path(path)
        
        if calculate_checksum is None:
            calculate_checksum = self.enable_checksums
        
        # Check content size
        content_size = len(content) if isinstance(content, (str, bytes)) else 0
        if content_size > self.max_file_size:
            raise SecurityError(f"Content too large: {content_size} > {self.max_file_size}")
        
        # Encrypt if requested
        if encrypt and self.crypto:
            if isinstance(content, str):
                content = content.encode('utf-8')
            content = self.crypto.encrypt(content)
            if 'b' not in mode:
                content = content.decode('utf-8')
        
        try:
            if atomic:
                # Atomic write using temporary file
                temp_path = path.with_suffix(path.suffix + '.tmp')
                
                with open(temp_path, mode, encoding=encoding if 'b' not in mode else None) as f:
                    f.write(content)
                
                # Set secure permissions
                os.chmod(temp_path, 0o600)
                
                # Atomic move
                temp_path.replace(path)
            else:
                # Direct write
                with open(path, mode, encoding=encoding if 'b' not in mode else None) as f:
                    f.write(content)
                
                # Set secure permissions
                os.chmod(path, 0o600)
            
            # Calculate checksum if requested
            checksum = None
            if calculate_checksum:
                checksum = self.calculate_checksum(path)
            
            # Update file cache
            with self._lock:
                self.file_cache[str(path)] = FileMetadata(
                    path=path,
                    size=path.stat().st_size,
                    created_at=time.time(),
                    modified_at=time.time(),
                    permissions=path.stat().st_mode,
                    checksum=checksum,
                    encrypted=encrypt
                )
            
            # Audit log
            self.audit_logger.log_security_event(
                event_type=AuditEventType.DATA_EXPORTED,
                details={
                    'file_path': str(path),
                    'file_size': content_size,
                    'encrypted': encrypt
                }
            )
            
            return checksum
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                level=AuditLevel.ERROR,
                details={'operation': 'file_write', 'path': str(path), 'error': str(e)}
            )
            raise SecurityError(f"File write failed: {e}")
    
    def create_temp_file(
        self,
        suffix: str = '',
        prefix: str = 'qem_',
        content: Optional[Union[str, bytes]] = None,
        encrypt: bool = False
    ) -> Path:
        """
        Create a secure temporary file.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            content: Optional initial content
            encrypt: Whether to encrypt the content
            
        Returns:
            Path to temporary file
        """
        # Validate suffix
        if suffix and not suffix.startswith('.'):
            suffix = '.' + suffix
        if suffix and suffix not in self.allowed_extensions:
            warnings.warn(f"Temporary file suffix not in allowed list: {suffix}")
        
        try:
            # Create temporary file
            fd, temp_path = tempfile.mkstemp(
                suffix=suffix,
                prefix=prefix,
                dir=self.base_directory
            )
            os.close(fd)  # Close file descriptor
            
            temp_path = Path(temp_path)
            
            # Set secure permissions
            os.chmod(temp_path, 0o600)
            
            # Write initial content if provided
            if content is not None:
                self.secure_write(temp_path, content, encrypt=encrypt, atomic=False)
            
            # Track temporary file
            with self._lock:
                file_id = str(temp_path)
                self.temp_files[file_id] = FileMetadata(
                    path=temp_path,
                    size=temp_path.stat().st_size if temp_path.exists() else 0,
                    created_at=time.time(),
                    modified_at=time.time(),
                    permissions=temp_path.stat().st_mode if temp_path.exists() else 0o600,
                    temporary=True,
                    encrypted=encrypt
                )
            
            return temp_path
            
        except Exception as e:
            raise SecurityError(f"Temporary file creation failed: {e}")
    
    def secure_delete(self, path: Union[str, Path], secure_wipe: bool = True) -> bool:
        """
        Securely delete a file.
        
        Args:
            path: File path to delete
            secure_wipe: Whether to securely wipe file contents
            
        Returns:
            True if deletion was successful
        """
        path = self.validate_path(path, allow_absolute=True)
        
        if not path.exists():
            return True
        
        try:
            # Check permissions
            if not self.check_permissions(path, [FilePermission.OWNER_WRITE]):
                raise SecurityError(f"Insufficient permissions to delete: {path}")
            
            file_size = path.stat().st_size
            
            # Secure wipe if requested
            if secure_wipe and file_size > 0:
                self._secure_wipe_file(path)
            
            # Delete file
            path.unlink()
            
            # Remove from caches
            with self._lock:
                file_id = str(path)
                if file_id in self.file_cache:
                    del self.file_cache[file_id]
                if file_id in self.temp_files:
                    del self.temp_files[file_id]
            
            # Audit log
            self.audit_logger.log_security_event(
                event_type=AuditEventType.DATA_EXPORTED,
                details={
                    'operation': 'file_delete',
                    'file_path': str(path),
                    'file_size': file_size,
                    'secure_wipe': secure_wipe
                }
            )
            
            return True
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                level=AuditLevel.ERROR,
                details={'operation': 'file_delete', 'path': str(path), 'error': str(e)}
            )
            warnings.warn(f"File deletion failed: {e}")
            return False
    
    def _secure_wipe_file(self, path: Path):
        """Securely wipe file contents before deletion."""
        try:
            file_size = path.stat().st_size
            
            # Overwrite with random data multiple times
            with open(path, 'r+b') as f:
                for _ in range(3):  # 3 passes
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
                
                # Final pass with zeros
                f.seek(0)
                f.write(b'\x00' * file_size)
                f.flush()
                os.fsync(f.fileno())
                
        except Exception as e:
            warnings.warn(f"Secure wipe failed: {e}")
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files.
        
        Args:
            max_age_hours: Maximum age of temporary files in hours
            
        Returns:
            Number of files cleaned up
        """
        cleanup_count = 0
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self._lock:
            temp_files_to_remove = []
            
            for file_id, metadata in self.temp_files.items():
                if metadata.created_at < cutoff_time:
                    try:
                        if metadata.path.exists():
                            self.secure_delete(metadata.path)
                        temp_files_to_remove.append(file_id)
                        cleanup_count += 1
                    except Exception as e:
                        warnings.warn(f"Failed to cleanup temp file {metadata.path}: {e}")
            
            # Remove from tracking
            for file_id in temp_files_to_remove:
                del self.temp_files[file_id]
        
        if cleanup_count > 0:
            self.audit_logger.log_security_event(
                event_type=AuditEventType.SYSTEM_STARTUP,
                details={'operation': 'temp_file_cleanup', 'cleaned_files': cleanup_count}
            )
        
        return cleanup_count
    
    def list_files(
        self,
        directory: Optional[Union[str, Path]] = None,
        pattern: str = "*",
        include_hidden: bool = False
    ) -> List[FileMetadata]:
        """
        List files in a directory with metadata.
        
        Args:
            directory: Directory to list (uses base directory if None)
            pattern: File pattern to match
            include_hidden: Whether to include hidden files
            
        Returns:
            List of file metadata
        """
        if directory is None:
            directory = self.base_directory
        else:
            directory = self.validate_path(directory)
        
        if not directory.is_dir():
            raise SecurityError(f"Not a directory: {directory}")
        
        files = []
        
        try:
            for path in directory.glob(pattern):
                if path.is_file():
                    # Skip hidden files unless requested
                    if not include_hidden and path.name.startswith('.'):
                        continue
                    
                    # Get or create metadata
                    file_id = str(path)
                    if file_id in self.file_cache:
                        metadata = self.file_cache[file_id]
                    else:
                        stat_info = path.stat()
                        metadata = FileMetadata(
                            path=path,
                            size=stat_info.st_size,
                            created_at=stat_info.st_ctime,
                            modified_at=stat_info.st_mtime,
                            permissions=stat_info.st_mode
                        )
                        
                        # Calculate checksum if enabled
                        if self.enable_checksums:
                            try:
                                metadata.checksum = self.calculate_checksum(path)
                            except Exception:
                                pass
                        
                        # Cache metadata
                        with self._lock:
                            self.file_cache[file_id] = metadata
                    
                    files.append(metadata)
        
        except Exception as e:
            raise SecurityError(f"Directory listing failed: {e}")
        
        return files
    
    def get_file_info(self, path: Union[str, Path]) -> Optional[FileMetadata]:
        """Get file metadata."""
        path = self.validate_path(path)
        
        if not path.exists():
            return None
        
        file_id = str(path)
        
        # Check cache first
        if file_id in self.file_cache:
            return self.file_cache[file_id]
        
        # Create new metadata
        try:
            stat_info = path.stat()
            metadata = FileMetadata(
                path=path,
                size=stat_info.st_size,
                created_at=stat_info.st_ctime,
                modified_at=stat_info.st_mtime,
                permissions=stat_info.st_mode
            )
            
            # Calculate checksum if enabled
            if self.enable_checksums:
                try:
                    metadata.checksum = self.calculate_checksum(path)
                except Exception:
                    pass
            
            # Cache metadata
            with self._lock:
                self.file_cache[file_id] = metadata
            
            return metadata
            
        except Exception as e:
            warnings.warn(f"Failed to get file info for {path}: {e}")
            return None


# Global secure file operations instance
_global_file_ops: Optional[SecureFileOperations] = None


def get_secure_file_ops() -> SecureFileOperations:
    """Get the global secure file operations instance."""
    global _global_file_ops
    if _global_file_ops is None:
        _global_file_ops = SecureFileOperations()
    return _global_file_ops


def set_secure_file_ops(file_ops: SecureFileOperations):
    """Set the global secure file operations instance."""
    global _global_file_ops
    _global_file_ops = file_ops