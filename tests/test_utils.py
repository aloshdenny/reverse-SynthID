"""Tests for utility functions."""

import pytest
import numpy as np
import tempfile
import os
import pickle
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config, get_nested
from src.utils.validation import (
    validate_image,
    validate_codebook,
    secure_pickle_load,
    validate_directory
)


class TestConfig:
    """Tests for configuration management."""
    
    def test_load_config_missing_file(self):
        """Test that loading missing config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/config.yaml')
    
    def test_get_nested(self):
        """Test nested config value retrieval."""
        config = {
            'detection': {
                'threshold': 0.5,
                'nested': {
                    'value': 42
                }
            }
        }
        
        assert get_nested(config, 'detection.threshold') == 0.5
        assert get_nested(config, 'detection.nested.value') == 42
        assert get_nested(config, 'nonexistent', default='default') == 'default'
        assert get_nested(config, 'detection.nonexistent.deep', default=0) == 0


class TestValidation:
    """Tests for validation functions."""
    
    def test_validate_directory_exists(self, tmp_path):
        """Test directory validation with existing directory."""
        assert validate_directory(str(tmp_path))
    
    def test_validate_directory_create(self, tmp_path):
        """Test directory creation when missing."""
        new_dir = tmp_path / 'newdir'
        assert not new_dir.exists()
        
        validate_directory(str(new_dir), create_if_missing=True)
        assert new_dir.exists()
    
    def test_validate_directory_missing(self, tmp_path):
        """Test that missing directory raises error."""
        with pytest.raises(FileNotFoundError):
            validate_directory(str(tmp_path / 'nonexistent'))
    
    def test_validate_directory_not_a_dir(self, tmp_path):
        """Test that file path raises error."""
        file_path = tmp_path / 'file.txt'
        file_path.write_text('test')
        
        with pytest.raises(NotADirectoryError):
            validate_directory(str(file_path))
    
    def test_validate_image_missing(self):
        """Test that missing image raises error."""
        with pytest.raises(FileNotFoundError):
            validate_image('/nonexistent/image.png')
    
    def test_validate_image_invalid_format(self, tmp_path):
        """Test that invalid format raises error."""
        file_path = tmp_path / 'test.txt'
        file_path.write_text('not an image')
        
        with pytest.raises(ValueError, match="Unsupported format"):
            validate_image(str(file_path))
    
    def test_validate_image_empty(self, tmp_path):
        """Test that empty file raises error."""
        file_path = tmp_path / 'empty.png'
        file_path.write_bytes(b'')
        
        with pytest.raises(ValueError, match="empty"):
            validate_image(str(file_path))
    
    def test_secure_pickle_load_missing(self):
        """Test that missing pickle raises error."""
        with pytest.raises(FileNotFoundError):
            secure_pickle_load('/nonexistent/file.pkl')
    
    def test_secure_pickle_load_valid(self, tmp_path):
        """Test loading valid pickle file."""
        data = {'test': [1, 2, 3], 'array': np.array([1, 2, 3])}
        file_path = tmp_path / 'test.pkl'
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        loaded = secure_pickle_load(str(file_path))
        assert loaded['test'] == [1, 2, 3]
        assert np.array_equal(loaded['array'], np.array([1, 2, 3]))
    
    def test_validate_codebook_missing_keys(self):
        """Test that codebook with missing keys raises error."""
        invalid_codebook = {'version': '1.0'}
        
        with pytest.raises(ValueError, match="Missing required key"):
            validate_codebook(invalid_codebook)
    
    def test_validate_codebook_valid(self):
        """Test validation of valid codebook."""
        codebook = {
            'version': '1.0',
            'source': 'test',
            'n_images_analyzed': 10,
            'image_size': 512,
            'reference_noise': np.zeros((512, 512, 3)),
            'carriers': [{'freq': (14, 14), 'coherence': 0.99}],
            'detection_threshold': 0.5
        }
        
        assert validate_codebook(codebook)
    
    def test_validate_codebook_invalid_shape(self):
        """Test that codebook with wrong shape raises error."""
        codebook = {
            'version': '1.0',
            'source': 'test',
            'n_images_analyzed': 10,
            'image_size': 512,
            'reference_noise': np.zeros((256, 256, 3)),  # Wrong size
            'carriers': [{'freq': (14, 14)}],
            'detection_threshold': 0.5
        }
        
        with pytest.raises(ValueError, match="wrong shape"):
            validate_codebook(codebook)
    
    def test_validate_codebook_invalid_threshold(self):
        """Test that codebook with invalid threshold raises error."""
        codebook = {
            'version': '1.0',
            'source': 'test',
            'n_images_analyzed': 10,
            'image_size': 512,
            'reference_noise': np.zeros((512, 512, 3)),
            'carriers': [{'freq': (14, 14)}],
            'detection_threshold': 2.0  # Out of range
        }
        
        with pytest.raises(ValueError, match="out of range"):
            validate_codebook(codebook)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
