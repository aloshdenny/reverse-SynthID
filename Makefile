# Makefile for Reverse SynthID

.PHONY: help install test lint format clean run-tests coverage docs

help:
	@echo "Reverse SynthID - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies"
	@echo "  make install-dev   Install dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-fast     Run fast tests only"
	@echo "  make coverage      Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linter (flake8)"
	@echo "  make format        Format code (black)"
	@echo "  make typecheck     Run type checker (mypy)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove cache and temporary files"
	@echo "  make clean-all     Remove all generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make extract       Extract codebook from sample data"
	@echo "  make detect        Run detection on sample image"

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest pytest-cov mypy black flake8

# Testing
test:
	pytest -v

test-fast:
	pytest -v -m "not slow"

coverage:
	pytest --cov=src --cov-report=html --cov-report=term tests/
	@echo "Coverage report generated in htmlcov/index.html"

# Code Quality
lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503

format:
	black src/ tests/ --line-length=100

typecheck:
	mypy src/ --ignore-missing-imports

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/
	@echo "Cleaned cache files"

clean-all: clean
	rm -rf venv/ env/ build/ dist/
	rm -f *.log *.pkl
	@echo "Cleaned all generated files"

# Examples
extract:
	@echo "Example: Extract codebook (requires images in data/pure_white/)"
	python src/extraction/synthid_codebook_extractor.py extract data/pure_white/ \
		--output example_codebook.pkl \
		--max-images 10

detect:
	@echo "Example: Detect watermark (requires codebook.pkl and image.png)"
	python src/extraction/synthid_codebook_extractor.py detect assets/synthid-watermark.jpeg \
		--codebook artifacts/codebook/synthid_codebook.pkl

# Development
dev-setup: install-dev
	@echo "Setting up development environment..."
	pre-commit install || echo "pre-commit not installed, skipping"
	@echo "Development environment ready!"

# Documentation
docs:
	@echo "Opening documentation..."
	@open README.md || xdg-open README.md || start README.md

# Quick checks before commit
check: format lint typecheck test
	@echo "All checks passed!"
