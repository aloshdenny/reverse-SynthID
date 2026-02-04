"""Tests for watermark analysis functions."""

import pytest
import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestFrequencyAnalysis:
    """Tests for frequency domain analysis."""
    
    def test_frequency_analysis_basic(self):
        """Test basic frequency analysis."""
        from watermark_investigation.watermark_full_123k_analysis import analyze_frequency
        
        # Create two similar images
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()
        
        result = analyze_frequency(img1, img2)
        
        assert result is not None
        assert 'freq_diff_mean' in result
        assert 'freq_diff_max' in result
        
        # Difference should be small for identical images
        assert result['freq_diff_mean'] < 0.1
    
    def test_frequency_analysis_different_sizes(self):
        """Test frequency analysis with different sized images."""
        from watermark_investigation.watermark_full_123k_analysis import analyze_frequency
        
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        
        result = analyze_frequency(img1, img2)
        
        # Should handle size mismatch
        assert result is not None
    
    def test_frequency_analysis_none_inputs(self):
        """Test frequency analysis with None inputs."""
        from watermark_investigation.watermark_full_123k_analysis import analyze_frequency
        
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        assert analyze_frequency(None, img) is None
        assert analyze_frequency(img, None) is None
        assert analyze_frequency(None, None) is None


class TestLSBAnalysis:
    """Tests for LSB analysis."""
    
    def test_lsb_analysis_basic(self):
        """Test basic LSB analysis."""
        from watermark_investigation.watermark_full_123k_analysis import analyze_lsb
        
        # Create image with known LSB pattern
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 0b11111110  # LSB = 0
        img[:, :, 1] = 0b11111111  # LSB = 1
        img[:, :, 2] = 0b10101010  # LSB = 0
        
        result = analyze_lsb(img)
        
        assert result is not None
        assert 'B_lsb' in result
        assert 'G_lsb' in result
        assert 'R_lsb' in result
        
        # Check expected values
        assert result['B_lsb'] == 0.0
        assert result['G_lsb'] == 1.0
        assert result['R_lsb'] == 0.0
    
    def test_lsb_analysis_random(self):
        """Test LSB analysis on random image."""
        from watermark_investigation.watermark_full_123k_analysis import analyze_lsb
        
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = analyze_lsb(img)
        
        assert result is not None
        
        # Random LSB should be close to 0.5
        for ch in ['B_lsb', 'G_lsb', 'R_lsb']:
            assert 0.3 < result[ch] < 0.7
    
    def test_lsb_analysis_none_input(self):
        """Test LSB analysis with None input."""
        from watermark_investigation.watermark_full_123k_analysis import analyze_lsb
        
        assert analyze_lsb(None) is None


class TestColorShiftAnalysis:
    """Tests for color shift analysis."""
    
    def test_color_shift_identical(self):
        """Test color shift on identical images."""
        from watermark_investigation.watermark_full_123k_analysis import analyze_color_shift
        
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = analyze_color_shift(img, img)
        
        assert result is not None
        
        # Identical images should have zero shift
        assert abs(result['R_shift']) < 0.01
        assert abs(result['G_shift']) < 0.01
        assert abs(result['B_shift']) < 0.01
    
    def test_color_shift_known_difference(self):
        """Test color shift with known difference."""
        from watermark_investigation.watermark_full_123k_analysis import analyze_color_shift
        
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 110
        
        result = analyze_color_shift(img1, img2)
        
        assert result is not None
        
        # Should detect +10 shift in all channels
        assert abs(result['R_shift'] - 10.0) < 0.1
        assert abs(result['G_shift'] - 10.0) < 0.1
        assert abs(result['B_shift'] - 10.0) < 0.1


class TestPerceptualHash:
    """Tests for perceptual hash distance."""
    
    def test_phash_identical(self):
        """Test phash distance on identical images."""
        from watermark_investigation.watermark_full_123k_analysis import compute_phash_distance
        
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        distance = compute_phash_distance(img, img)
        
        # Identical images should have distance 0
        assert distance == 0
    
    def test_phash_similar(self):
        """Test phash distance on similar images."""
        from watermark_investigation.watermark_full_123k_analysis import compute_phash_distance
        
        img1 = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()
        img2[:10, :10] = 200  # Small modification
        
        distance = compute_phash_distance(img1, img2)
        
        # Similar images should have small distance
        assert 0 <= distance <= 20
    
    def test_phash_different(self):
        """Test phash distance on different images."""
        from watermark_investigation.watermark_full_123k_analysis import compute_phash_distance
        
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        distance = compute_phash_distance(img1, img2)
        
        # Very different images should have large distance
        assert distance > 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
