# Installation Guide

## Requirements

- Python 3.10 or higher
- macOS, Linux, or Windows
- At least 4GB RAM (8GB+ recommended for large datasets)
- Optional: CUDA for GPU acceleration (not yet implemented)

## Quick Install

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/reverse-SynthID.git
cd reverse-SynthID
```

### 2. Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Run tests
pytest

# Check CLI tools
python src/extraction/synthid_codebook_extractor.py --help
```

## Platform-Specific Notes

### macOS

If you encounter OpenCV issues:

```bash
# Install via Homebrew first
brew install opencv

# Then install Python package
pip install opencv-python==4.7.0.72
```

### Linux

Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-opencv libopencv-dev

# Fedora/RHEL
sudo dnf install python3-opencv opencv-devel
```

### Windows

Ensure Visual C++ Redistributables are installed:
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

## Configuration

### 1. Copy Config Template

```bash
# Config already exists, but you can customize it
cp config.yaml config.local.yaml
# Edit config.local.yaml with your settings
```

### 2. Set Environment Variables (Optional)

**macOS/Linux:**
```bash
export SYNTHID_DATA_DIR=/path/to/your/data
export SYNTHID_OUTPUT_DIR=/path/to/output
export SYNTHID_MAX_WORKERS=8
```

**Windows:**
```cmd
set SYNTHID_DATA_DIR=C:\path\to\your\data
set SYNTHID_OUTPUT_DIR=C:\path\to\output
set SYNTHID_MAX_WORKERS=8
```

## Development Setup

For development with additional tools:

```bash
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov mypy black flake8

# Run type checking
mypy src/

# Run linter
flake8 src/

# Run with coverage
pytest --cov=src tests/
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Memory Issues

For large datasets, increase available memory:

```yaml
# In config.yaml
processing:
  max_workers: 2  # Reduce parallelism
  batch_size: 50  # Reduce batch size
```

### OpenCV Issues on macOS

```bash
# Use headless version
pip uninstall opencv-python
pip install opencv-python-headless==4.7.0.72
```

### Windows Long Path Issues

Enable long paths in Windows:

```
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## Updating

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstall

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Remove cache files
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## Docker (Alternative)

Coming soon: Docker image for easier deployment.

```bash
# Will be available in future release
docker pull yourusername/reverse-synthid:latest
docker run -it -v $(pwd):/workspace reverse-synthid
```

## Next Steps

After installation, see:
- [README.md](README.md) for usage examples
- [config.yaml](config.yaml) for configuration options
- [tests/](tests/) for example code
