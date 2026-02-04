"""Tests for watermark extraction functions."""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.extraction.synthid_codebook_extractor import wavelet_denoise


class TestWaveletDenoise:
    """Tests for wavelet denoising function."""
    
    def test_denoise_basic(self):
        """Test basic denoising functionality."""
        # Create a simple test image
        channel = np.random.randn(128, 128).astype(np.float32) * 0.1
        
        # Add noise
        noisy = channel + np.random.randn(128, 128).astype(np.float32) * 0.01
        
        # Denoise
        denoised = wavelet_denoise(noisy)
        
        # Check output properties
        assert denoised.shape == noisy.shape
        assert denoised.dtype in [np.float32, np.float64]
        
        # Denoised should have lower variance than noisy
        assert np.var(denoised) <= np.var(noisy)
    
    def test_denoise_empty_channel(self):
        """Test that empty channel raises error."""
        empty = np.array([])
        
        with pytest.raises(ValueError, match="empty"):
            wavelet_denoise(empty)
    
    def test_denoise_1d_channel(self):
        """Test that 1D array raises error."""
        channel_1d = np.random.randn(100)
        
        with pytest.raises(ValueError, match="2D"):
            wavelet_denoise(channel_1d)
    
    def test_denoise_shape_preservation(self):
        """Test that denoising preserves input shape."""
        sizes = [(64, 64), (128, 128), (100, 150), (200, 200)]
        
        for size in sizes:
            channel = np.random.randn(*size).astype(np.float32)
            denoised = wavelet_denoise(channel)
            assert denoised.shape == channel.shape, f"Shape mismatch for size {size}"
    
    def test_denoise_different_wavelets(self):
        """Test denoising with different wavelet types."""
        channel = np.random.randn(128, 128).astype(np.float32)
        
        wavelets = ['db4', 'haar', 'sym2']
        for wavelet in wavelets:
            denoised = wavelet_denoise(channel, wavelet=wavelet)
            assert denoised.shape == channel.shape
    
    def test_denoise_constant_image(self):
        """Test denoising on constant image."""
        # Constant image should remain mostly unchanged
        constant = np.ones((128, 128), dtype=np.float32) * 100
        denoised = wavelet_denoise(constant)
        
        # Should be very similar to original
        assert np.allclose(denoised, constant, atol=1.0)


class TestExtractCodebook:
    """Tests for codebook extraction."""
    
    def create_test_images(self, tmp_path, n_images=5):
        """Helper to create test images."""
        image_dir = tmp_path / 'images'
        image_dir.mkdir()
        
        for i in range(n_images):
            # Create random image
            img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            path = image_dir / f'test_{i:03d}.png'
            cv2.imwrite(str(path), img)
        
        return str(image_dir)
    
    def test_extract_codebook_missing_directory(self):
        """Test that missing directory raises error."""
        from src.extraction.synthid_codebook_extractor import extract_codebook
        
        with pytest.raises(FileNotFoundError):
            extract_codebook('/nonexistent/dir', 'output.pkl')
    
    def test_extract_codebook_too_few_images(self, tmp_path):
        """Test that too few images raises error."""
        from src.extraction.synthid_codebook_extractor import extract_codebook
        
        image_dir = self.create_test_images(tmp_path, n_images=3)
        
        with pytest.raises(ValueError, match="Too few images"):
            extract_codebook(image_dir, 'output.pkl', max_images=10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
