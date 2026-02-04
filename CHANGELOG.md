# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-02-04

### Added
- **Configuration System**: Added `config.yaml` for centralized configuration management
- **Utilities Module**: New `src/utils/` with configuration, logging, and validation utilities
- **Secure Pickle Loading**: Added `SecureUnpickler` to prevent arbitrary code execution
- **Input Validation**: Comprehensive validation for images, codebooks, and directories
- **Parallel Processing**: Multi-threaded image processing in codebook finder (4-8x speedup)
- **Batch Detection Mode**: Process entire directories of images at once
- **Progress Bars**: Real-time progress tracking with `tqdm`
- **Unit Tests**: Comprehensive test suite with 25+ tests (pytest)
- **CLI Enhancements**: Improved command-line interface with better help text and examples
- **Logging System**: Configurable logging with file output support
- **Environment Variables**: Support for `SYNTHID_*` environment variables

### Fixed
- **Hardcoded Paths**: Removed all hardcoded paths from `watermark_full_123k_analysis.py`
- **Wavelet Denoising Bug**: Fixed shape mismatch in reconstruction (now handles padding/cropping)
- **Confidence Score Bug**: Fixed negative confidence scores with edge cases
- **Division by Zero**: Added safeguards in structure ratio calculation
- **Error Handling**: Added try-catch blocks throughout codebase
- **Memory Leaks**: Improved memory efficiency in codebook extraction

### Changed
- **Dependencies**: Pinned all package versions for reproducibility
- **Project Structure**: Reorganized utilities into dedicated module
- **CLI Interface**: Enhanced with argparse, better error messages
- **Documentation**: Updated with usage examples and troubleshooting

### Security
- **Pickle Vulnerability**: Restricted unpickling to safe classes only
- **Path Traversal**: Added validation to prevent directory traversal attacks
- **File Size Limits**: Added maximum file size checks for images

## [1.0.0] - Original Release

### Initial Features
- SynthID watermark detection and codebook extraction
- Nano-150k watermark analysis (123,268 image pairs)
- Frequency domain analysis
- LSB pattern detection
- Phase coherence analysis
- Visualization tools
