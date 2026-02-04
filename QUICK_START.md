# Quick Start Guide

Get up and running with Reverse SynthID in 5 minutes.

## Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/reverse-SynthID.git
cd reverse-SynthID

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
pytest
```

## Basic Usage

### 1. Detect Watermark in Single Image

```bash
python src/extraction/synthid_codebook_extractor.py detect \
    assets/synthid-watermark.jpeg \
    --codebook artifacts/codebook/synthid_codebook.pkl
```

**Output:**
```
============================================================
SynthID Watermark Detection Results
============================================================
Image:            assets/synthid-watermark.jpeg
Watermarked:      YES
Confidence:       0.9850
Correlation:      0.5355 (threshold: 0.1790)
Phase Match:      0.9571
Structure Ratio:  1.2753 (expected: ~1.32)
============================================================
```

### 2. Batch Detection on Directory

```bash
python src/extraction/synthid_codebook_extractor.py detect \
    /path/to/images/ \
    --batch \
    --codebook artifacts/codebook/synthid_codebook.pkl \
    --output results.json
```

**Output:**
```
✓ image1.png: WATERMARKED (confidence: 0.985)
✓ image2.png: CLEAN (confidence: 0.123)
✓ image3.png: WATERMARKED (confidence: 0.876)

Summary: 2/3 images watermarked
Results saved to results.json
```

### 3. Extract Custom Codebook

```bash
# From your own watermarked images
python src/extraction/synthid_codebook_extractor.py extract \
    /path/to/watermarked/images/ \
    --output my_codebook.pkl \
    --max-images 250
```

### 4. Analyze Image Pairs for Watermarks

```bash
# Analyze editing watermarks
python watermark_investigation/watermark_full_123k_analysis.py \
    pairs.jsonl \
    --base-path /path/to/images \
    --output analysis_results.json \
    --max-pairs 1000
```

## Configuration

### Method 1: Edit Config File

```yaml
# config.yaml
detection:
  correlation_threshold: 0.179
  phase_match_threshold: 0.5

processing:
  max_workers: 8
  batch_size: 100
```

### Method 2: Environment Variables

```bash
export SYNTHID_DATA_DIR=/path/to/data
export SYNTHID_OUTPUT_DIR=/path/to/output
export SYNTHID_MAX_WORKERS=8
```

### Method 3: Command Line

```bash
python script.py \
    --config custom_config.yaml \
    --log-level DEBUG \
    --log-file debug.log
```

## Common Commands

### Using Makefile (Recommended)

```bash
# Show all commands
make help

# Run tests
make test

# Run tests with coverage
make coverage

# Format code
make format

# Lint code
make lint

# Clean cache files
make clean
```

### Manual Commands

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Format code
black src/ tests/

# Lint code
flake8 src/

# Type check
mypy src/
```

## Examples

### Example 1: Quick Test

```bash
# Test with a single image
python src/extraction/synthid_codebook_extractor.py detect \
    assets/synthid-watermark.jpeg \
    --codebook artifacts/codebook/synthid_codebook.pkl \
    --log-level INFO
```

### Example 2: Production Batch Processing

```bash
# Process large directory with logging
python src/extraction/synthid_codebook_extractor.py detect \
    /production/images/ \
    --batch \
    --codebook production_codebook.pkl \
    --output batch_results.json \
    --log-file batch.log \
    --log-level INFO
```

### Example 3: Custom Analysis

```bash
# Extract patterns with parallel processing
python src/analysis/synthid_codebook_finder.py \
    /path/to/images/ \
    --output ./analysis_results \
    --max-images 500 \
    --workers 8 \
    --size 512
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

```bash
# Solution: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "MemoryError"

```bash
# Solution: Reduce workers in config.yaml
processing:
  max_workers: 2  # Reduce from 8
```

### Issue: "Image validation failed"

```bash
# Solution: Check image format and size
file image.png
identify image.png  # If ImageMagick installed
```

## Next Steps

1. **Read Full Documentation**
   - [README.md](README.md) - Overview
   - [INSTALLATION.md](INSTALLATION.md) - Detailed installation
   - [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

2. **Explore Examples**
   - Check `tests/` for code examples
   - Review `config.yaml` for options
   - Read `CONTRIBUTING.md` for development

3. **Get Help**
   - Open GitHub issue
   - Check existing issues
   - Read troubleshooting guide

## Cheat Sheet

```bash
# Install
pip install -r requirements.txt

# Test
pytest

# Detect single
python src/extraction/synthid_codebook_extractor.py detect image.png --codebook codebook.pkl

# Detect batch
python src/extraction/synthid_codebook_extractor.py detect images/ --batch --codebook codebook.pkl

# Extract codebook
python src/extraction/synthid_codebook_extractor.py extract images/ --output codebook.pkl

# Run tests
make test

# Coverage
make coverage

# Clean
make clean
```

## Resources

- [GitHub Repository](https://github.com/yourusername/reverse-SynthID)
- [SynthID Paper](https://arxiv.org/abs/2510.09263)
- [Google DeepMind SynthID](https://deepmind.google/technologies/synthid/)

## Support

- 📖 Documentation: See `*.md` files
- 🐛 Bug Reports: GitHub Issues
- 💬 Questions: GitHub Discussions
- 🤝 Contributing: See CONTRIBUTING.md

---

**Happy watermark hunting!** 🔍
