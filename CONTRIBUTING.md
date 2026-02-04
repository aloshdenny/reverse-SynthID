# Contributing to Reverse SynthID

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Remember this is research/educational software

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/reverse-SynthID.git
cd reverse-SynthID
git checkout -b feature/your-feature-name
```

### 2. Set Up Development Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov mypy black flake8
```

### 3. Make Your Changes

Follow these guidelines:

- **Code Style**: Use PEP 8 (run `black` for formatting)
- **Type Hints**: Add type annotations to function signatures
- **Docstrings**: Use Google-style docstrings
- **Tests**: Write tests for new functionality
- **Logging**: Use the logging module, not print statements

### 4. Test Your Changes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Type checking
mypy src/

# Linting
flake8 src/
```

### 5. Submit Pull Request

- Write a clear description of changes
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed

## Development Guidelines

### Code Style

```python
# Good: Type hints and docstring
def detect_watermark(image: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Detect watermark in image.
    
    Args:
        image: Input image as numpy array
        threshold: Detection threshold (0-1)
    
    Returns:
        True if watermark detected, False otherwise
    """
    pass

# Bad: No types or documentation
def detect_watermark(image, threshold=0.5):
    pass
```

### Error Handling

```python
# Good: Specific exceptions with context
def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    try:
        return cv2.imread(path)
    except Exception as e:
        raise ValueError(f"Failed to load {path}: {e}")

# Bad: Silent failures
def load_image(path):
    try:
        return cv2.imread(path)
    except:
        return None
```

### Testing

Write tests for:
- New functions
- Bug fixes
- Edge cases
- Error conditions

```python
def test_detect_watermark_missing_image():
    """Test that missing image raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        detect_watermark('/nonexistent/image.png')
```

### Logging

```python
# Good: Use logger
import logging
logger = logging.getLogger(__name__)

def process_image(path):
    logger.info(f"Processing {path}")
    try:
        img = load_image(path)
        logger.debug(f"Image shape: {img.shape}")
    except Exception as e:
        logger.error(f"Failed to process {path}: {e}")
        raise

# Bad: Use print
def process_image(path):
    print(f"Processing {path}")  # Don't do this
```

## Areas for Contribution

### High Priority

- [ ] GPU acceleration for wavelet transforms
- [ ] Support for additional watermark types
- [ ] Performance optimizations
- [ ] Additional test coverage
- [ ] Documentation improvements

### Feature Requests

- [ ] Real-time video watermark detection
- [ ] Web UI for detection
- [ ] Docker containerization
- [ ] Multi-scale analysis
- [ ] Robustness testing suite

### Bug Reports

When reporting bugs, include:
- Python version
- Operating system
- Full error traceback
- Minimal reproduction code
- Expected vs actual behavior

## Project Structure

```
reverse-SynthID/
├── src/
│   ├── analysis/        # Analysis algorithms
│   ├── extraction/      # Codebook extraction
│   └── utils/           # Utilities (config, validation)
├── tests/               # Unit tests
├── watermark_investigation/  # Nano-150k analysis
├── artifacts/           # Generated artifacts
├── config.yaml          # Configuration
└── docs/               # Documentation
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """
    Brief description of what the function does.
    
    Longer description if needed. Can span multiple lines
    and include examples.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: "default")
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string
    
    Example:
        >>> result = example_function(42, "test")
        >>> print(result)
        True
    """
    pass
```

### README Updates

When adding features, update:
- README.md (usage examples)
- CHANGELOG.md (version history)
- config.yaml (new options)

## Git Workflow

### Commit Messages

Follow conventional commits:

```
feat: Add batch detection mode
fix: Handle empty images in wavelet denoising
docs: Update installation guide
test: Add tests for config loading
refactor: Extract validation to separate module
perf: Optimize frequency analysis with caching
```

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `test/description` - Tests
- `refactor/description` - Refactoring

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex sections
- [ ] Updated documentation
- [ ] No new warnings
```

## Questions?

- Open an issue for discussions
- Check existing issues before creating new ones
- Be patient - this is maintained by volunteers

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## Thank You!

Every contribution, no matter how small, is valuable. Thank you for helping improve this project!
