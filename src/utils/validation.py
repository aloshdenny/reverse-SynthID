"""Validation and security utilities."""

import os
import pickle
import hashlib
import numpy as np
import cv2
from typing import Any, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger('reverse_synthid.validation')


class SecureUnpickler(pickle.Unpickler):
    """Secure unpickler that restricts allowed types."""
    
    ALLOWED_MODULES = {
        'numpy',
        'numpy.core',
        'numpy.core.multiarray',
        'numpy.core.numeric',
        'numpy.ma.core',
        'builtins',
        '__builtin__',
    }
    
    def find_class(self, module, name):
        """Only allow safe classes to be unpickled."""
        if module not in self.ALLOWED_MODULES:
            raise pickle.UnpicklingError(
                f"Attempted to load unsafe module: {module}.{name}"
            )
        return super().find_class(module, name)


def secure_pickle_load(filepath: str, verify_hash: Optional[str] = None) -> Any:
    """
    Securely load a pickle file with validation.
    
    Args:
        filepath: Path to pickle file
        verify_hash: Optional SHA256 hash to verify file integrity
    
    Returns:
        Unpickled object
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If hash verification fails
        pickle.UnpicklingError: If file contains unsafe data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    # Verify hash if provided
    if verify_hash:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        if file_hash != verify_hash:
            raise ValueError(
                f"Hash verification failed for {filepath}. "
                f"Expected: {verify_hash}, Got: {file_hash}"
            )
    
    # Load with secure unpickler
    try:
        with open(filepath, 'rb') as f:
            return SecureUnpickler(f).load()
    except Exception as e:
        logger.error(f"Failed to load pickle file {filepath}: {e}")
        raise


def validate_image(
    image_path: str,
    max_size: int = 50_000_000,  # 50MB
    allowed_formats: Optional[set] = None
) -> bool:
    """
    Validate an image file before processing.
    
    Args:
        image_path: Path to image file
        max_size: Maximum file size in bytes
        allowed_formats: Set of allowed file extensions
    
    Returns:
        True if valid, False otherwise
    
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image is invalid
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Check file size
    file_size = os.path.getsize(image_path)
    if file_size > max_size:
        raise ValueError(
            f"Image too large: {file_size / 1e6:.1f}MB (max: {max_size / 1e6:.1f}MB)"
        )
    
    if file_size == 0:
        raise ValueError(f"Image file is empty: {image_path}")
    
    # Check format
    if allowed_formats is None:
        allowed_formats = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in allowed_formats:
        raise ValueError(
            f"Unsupported format: {ext}. Allowed: {allowed_formats}"
        )
    
    # Try to load image
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Check dimensions
        if img.shape[0] < 8 or img.shape[1] < 8:
            raise ValueError(
                f"Image too small: {img.shape}. Minimum: 8x8 pixels"
            )
        
        return True
    
    except Exception as e:
        logger.error(f"Image validation failed for {image_path}: {e}")
        raise


def validate_codebook(codebook: Dict[str, Any]) -> bool:
    """
    Validate codebook structure and contents.
    
    Args:
        codebook: Codebook dictionary
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If codebook is invalid
    """
    required_keys = [
        'version',
        'source',
        'n_images_analyzed',
        'image_size',
        'reference_noise',
        'carriers',
        'detection_threshold'
    ]
    
    # Check required keys
    for key in required_keys:
        if key not in codebook:
            raise ValueError(f"Missing required key in codebook: {key}")
    
    # Validate types
    if not isinstance(codebook['n_images_analyzed'], (int, np.integer)):
        raise ValueError("n_images_analyzed must be an integer")
    
    if codebook['n_images_analyzed'] < 1:
        raise ValueError("n_images_analyzed must be positive")
    
    if not isinstance(codebook['image_size'], (int, np.integer)):
        raise ValueError("image_size must be an integer")
    
    # Validate reference noise
    ref_noise = codebook['reference_noise']
    if not isinstance(ref_noise, np.ndarray):
        raise ValueError("reference_noise must be a numpy array")
    
    expected_shape = (codebook['image_size'], codebook['image_size'], 3)
    if ref_noise.shape != expected_shape:
        raise ValueError(
            f"reference_noise has wrong shape. Expected: {expected_shape}, "
            f"Got: {ref_noise.shape}"
        )
    
    # Validate carriers
    if not isinstance(codebook['carriers'], list):
        raise ValueError("carriers must be a list")
    
    if len(codebook['carriers']) == 0:
        raise ValueError("carriers list is empty")
    
    # Validate detection threshold
    threshold = codebook['detection_threshold']
    if not isinstance(threshold, (int, float, np.number)):
        raise ValueError("detection_threshold must be numeric")
    
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(
            f"detection_threshold out of range: {threshold}. Expected: [-1, 1]"
        )
    
    logger.info(
        f"Codebook validation passed: {codebook['n_images_analyzed']} images, "
        f"{len(codebook['carriers'])} carriers"
    )
    
    return True


def validate_directory(
    directory: str,
    create_if_missing: bool = False
) -> bool:
    """
    Validate a directory exists and is accessible.
    
    Args:
        directory: Directory path
        create_if_missing: Create directory if it doesn't exist
    
    Returns:
        True if valid
    
    Raises:
        NotADirectoryError: If path exists but is not a directory
        PermissionError: If directory is not accessible
    """
    if not os.path.exists(directory):
        if create_if_missing:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            return True
        else:
            raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    if not os.access(directory, os.R_OK):
        raise PermissionError(f"Directory not readable: {directory}")
    
    return True
