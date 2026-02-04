"""Configuration management for reverse SynthID."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Global configuration cache
_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, searches for config.yaml
                    in current directory and project root.
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file is invalid
    """
    global _config
    
    if _config is not None and config_path is None:
        return _config
    
    # Search for config file
    if config_path is None:
        search_paths = [
            'config.yaml',
            '../config.yaml',
            '../../config.yaml',
            os.path.join(os.path.dirname(__file__), '../../config.yaml')
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(
                "config.yaml not found. Please create one or specify path."
            )
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if set
    config = _apply_env_overrides(config)
    
    _config = config
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config."""
    
    # Override paths from environment
    if 'SYNTHID_DATA_DIR' in os.environ:
        config['paths']['data_dir'] = os.environ['SYNTHID_DATA_DIR']
    
    if 'SYNTHID_OUTPUT_DIR' in os.environ:
        config['paths']['output_dir'] = os.environ['SYNTHID_OUTPUT_DIR']
    
    if 'SYNTHID_CODEBOOK' in os.environ:
        config['paths']['codebook_file'] = os.environ['SYNTHID_CODEBOOK']
    
    # Override processing settings
    if 'SYNTHID_MAX_WORKERS' in os.environ:
        config['processing']['max_workers'] = int(os.environ['SYNTHID_MAX_WORKERS'])
    
    return config


def get_config() -> Dict[str, Any]:
    """
    Get current configuration (loads if not already loaded).
    
    Returns:
        Configuration dictionary
    """
    if _config is None:
        return load_config()
    return _config


def get_nested(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'detection.correlation_threshold')
        default: Default value if path not found
    
    Returns:
        Configuration value or default
    
    Example:
        >>> config = {'detection': {'threshold': 0.5}}
        >>> get_nested(config, 'detection.threshold')
        0.5
    """
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value
