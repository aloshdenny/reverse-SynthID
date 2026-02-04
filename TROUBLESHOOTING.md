# Troubleshooting Guide

Common issues and solutions for Reverse SynthID.

## Installation Issues

### "ModuleNotFoundError: No module named 'cv2'"

**Problem**: OpenCV not installed correctly

**Solution**:
```bash
pip uninstall opencv-python
pip install opencv-python==4.7.0.72

# If still fails on macOS:
brew install opencv
pip install opencv-python==4.7.0.72

# If fails on headless server:
pip install opencv-python-headless==4.7.0.72
```

### "ImportError: No module named 'src'"

**Problem**: Python can't find the source modules

**Solution**:
```bash
# Run from project root
cd /path/to/reverse-SynthID

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### "yaml.scanner.ScannerError"

**Problem**: Invalid YAML syntax in config file

**Solution**:
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check for tabs (use spaces instead)
# Check for proper indentation
```

## Runtime Issues

### "MemoryError" or System Runs Out of RAM

**Problem**: Processing too many images in parallel

**Solution**:
```yaml
# Edit config.yaml
processing:
  max_workers: 2  # Reduce from default 8
  batch_size: 50  # Reduce from 100

# Or use command line
python script.py --workers 2 --max-images 100
```

### "pickle.UnpicklingError: Attempted to load unsafe module"

**Problem**: Trying to load codebook created with old version

**Solution**:
```bash
# Re-extract codebook with new version
python src/extraction/synthid_codebook_extractor.py extract \
    data/pure_white/ --output new_codebook.pkl

# Or disable security (not recommended)
# Edit src/utils/validation.py and remove restrictions
```

### "FileNotFoundError: config.yaml not found"

**Problem**: Script can't find configuration file

**Solution**:
```bash
# Copy from repository
cp config.yaml /path/to/your/location

# Or specify path explicitly
python script.py --config /path/to/config.yaml

# Or use environment variable
export SYNTHID_CONFIG=/path/to/config.yaml
```

### Slow Performance

**Problem**: Processing is too slow

**Solutions**:

1. **Enable parallel processing**:
```bash
python src/analysis/synthid_codebook_finder.py \
    data/images/ --workers 8
```

2. **Reduce image size**:
```yaml
# config.yaml
image:
  target_size: 256  # Instead of 512
```

3. **Use fewer images**:
```bash
python script.py --max-images 100
```

4. **Check CPU usage**:
```bash
# Should see high CPU usage with parallel processing
top -o cpu
```

## Detection Issues

### "All Images Detected as Non-Watermarked"

**Problem**: Codebook doesn't match images

**Solutions**:

1. **Verify codebook source**: Codebook must be from same watermarking system
2. **Check threshold**: May need adjustment
```python
# In detect_synthid function
threshold = codebook['detection_threshold'] * 0.8  # More lenient
```

3. **Inspect intermediate results**:
```bash
python src/extraction/synthid_codebook_extractor.py detect \
    image.png --codebook codebook.pkl --log-level DEBUG
```

### "High False Positive Rate"

**Problem**: Detecting watermarks in clean images

**Solution**:
```python
# Increase threshold
# In config.yaml:
detection:
  correlation_threshold: 0.25  # Instead of 0.179
```

## Test Failures

### Tests Fail with "Image not found"

**Problem**: Test expects specific image files

**Solution**:
```bash
# Tests create temporary images, but may fail if permissions wrong
chmod 755 tests/

# Run with verbose output
pytest -v tests/test_extraction.py
```

### Tests Fail on Windows

**Problem**: Path separator differences

**Solution**:
```python
# Use pathlib instead of os.path
from pathlib import Path
path = Path('dir') / 'file.txt'
```

## Platform-Specific Issues

### macOS: "zsh: command not found: python"

**Solution**:
```bash
# Use python3 instead
python3 script.py

# Or create alias
echo "alias python=python3" >> ~/.zshrc
source ~/.zshrc
```

### Windows: "Access Denied" Errors

**Solution**:
```cmd
# Run as Administrator
# Or check antivirus isn't blocking

# Enable long paths
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Linux: OpenCV Can't Find Libraries

**Solution**:
```bash
# Install system dependencies
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# For older systems
sudo apt-get install libsm6 libxext6 libxrender-dev
```

## Data Issues

### "Failed to load image" for Valid Images

**Problem**: Corrupted files or unsupported format

**Solutions**:

1. **Verify image**:
```bash
file image.png  # Should show "PNG image data"
identify image.png  # If ImageMagick installed
```

2. **Convert format**:
```bash
convert image.webp image.png  # Using ImageMagick
```

3. **Check file size**:
```bash
# Empty or truncated files
ls -lh image.png

# Validate with Python
python -c "import cv2; img=cv2.imread('image.png'); print(img.shape if img is not None else 'FAILED')"
```

## Performance Profiling

### Profile Slow Code

```bash
# Profile script
python -m cProfile -o profile.stats script.py

# View results
python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumtime'); p.print_stats(20)"

# Or use snakeviz for visualization
pip install snakeviz
snakeviz profile.stats
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile function
python -m memory_profiler script.py
```

## Getting Help

If issue persists:

1. **Check logs**:
```bash
# Enable debug logging
python script.py --log-level DEBUG --log-file debug.log
cat debug.log
```

2. **Create minimal reproduction**:
```python
# Simplest code that shows the problem
import cv2
import numpy as np
from src.extraction.synthid_codebook_extractor import detect_synthid

result = detect_synthid('test.png', 'codebook.pkl')
print(result)
```

3. **Open GitHub issue** with:
   - Python version: `python --version`
   - OS: `uname -a` (Linux/Mac) or `systeminfo` (Windows)
   - Error traceback
   - Minimal reproduction code

4. **Check existing issues**: https://github.com/yourusername/reverse-SynthID/issues

## Common Error Messages

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `FileNotFoundError: config.yaml` | Wrong directory | Run from project root or use `--config` |
| `MemoryError` | Too much data | Reduce `max_workers` or `batch_size` |
| `ValueError: Image too small` | Invalid image | Check image is at least 8x8 pixels |
| `UnpicklingError` | Corrupted codebook | Re-extract codebook |
| `ImportError: No module` | Missing dependency | `pip install -r requirements.txt` |
| `PermissionError` | No write access | Check file/directory permissions |

## Debug Mode

Enable full debugging:

```bash
# Maximum verbosity
export PYTHONPATH="$(pwd)"
python -u script.py \
    --log-level DEBUG \
    --log-file debug.log \
    2>&1 | tee console.log

# Then inspect logs
less debug.log
```

## Still Stuck?

1. Read the [INSTALLATION.md](INSTALLATION.md)
2. Check [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
3. Review test files in `tests/` for usage examples
4. Open an issue with full details
