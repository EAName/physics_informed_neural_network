.PHONY: install test lint format clean build run-train run-predict docker-build docker-run docker-test generate-data

# Python environment
install:
	pip install -r requirements.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v --cov=src

# Linting and formatting
lint:
	flake8 src tests
	mypy src tests

format:
	black src tests
	isort src tests

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python setup.py sdist bdist_wheel

# Run training
run-train:
	python -m src.cli.train --config config/power_grid_config.yaml --model-type power_grid

# Run prediction
run-predict:
	python -m src.cli.predict --model-path models/power_grid_model.h5 --model-type power_grid --input-data data/input.json --output-file data/predictions.json

# Generate synthetic data
generate-data-power-grid:
	python -m src.cli.generate_data --data-type power_grid --n-samples 1000 --bus-count 14 --line-count 20

generate-data-solar:
	python -m src.cli.generate_data --data-type renewable --system-type solar --n-samples 1000

generate-data-wind:
	python -m src.cli.generate_data --data-type renewable --system-type wind --n-samples 1000

# Docker commands
docker-build:
	docker-compose build

docker-run:
	docker-compose up pinn

docker-test:
	docker-compose run tests

# Development environment
dev:
	docker-compose up jupyter

# Help
help:
	@echo "Available commands:"
	@echo "  install              - Install dependencies"
	@echo "  test                 - Run tests"
	@echo "  lint                 - Run linters"
	@echo "  format               - Format code"
	@echo "  clean                - Clean build artifacts"
	@echo "  build                - Build package"
	@echo "  run-train            - Run training"
	@echo "  run-predict          - Run prediction"
	@echo "  generate-data-power-grid - Generate power grid data"
	@echo "  generate-data-solar  - Generate solar data"
	@echo "  generate-data-wind   - Generate wind data"
	@echo "  docker-build         - Build Docker images"
	@echo "  docker-run           - Run Docker container"
	@echo "  docker-test          - Run tests in Docker"
	@echo "  dev                  - Start development environment" 