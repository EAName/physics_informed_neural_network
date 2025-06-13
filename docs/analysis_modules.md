# Analysis Modules Documentation

This document provides comprehensive documentation for the analysis modules in the Physics-Informed Neural Network (PINN) project.

## Table of Contents
1. [Power Grid Analysis](#power-grid-analysis)
2. [Renewable Energy Analysis](#renewable-energy-analysis)
3. [Common Utilities](#common-utilities)
4. [Testing](#testing)

## Power Grid Analysis

The power grid analysis module (`src/analysis/power_grid_analysis.py`) provides functionality for analyzing power grid systems using Physics-Informed Neural Networks.

### Key Features
- Data generation for power grid systems
- Model initialization and training
- Power flow analysis
- Voltage stability assessment
- Results visualization and saving

### Usage Example
```python
from src.analysis.power_grid_analysis import main

# Run complete analysis
main()

# Or use individual components
from src.analysis.power_grid_analysis import (
    generate_data,
    initialize_model,
    train_model,
    evaluate_model
)

# Generate data
train_data, test_data = generate_data()

# Initialize and train model
model, config = initialize_model()
history = train_model(model, train_data, test_data, config)

# Evaluate model
predictions, metrics = evaluate_model(model, test_data)
```

### Configuration
The module uses configuration from `config/power_grid_config.yaml`:
```yaml
model:
  layers: [64, 128, 64]
  activation: 'relu'
  dropout_rate: 0.2

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

## Renewable Energy Analysis

The renewable energy analysis module (`src/analysis/renewable_analysis.py`) provides functionality for analyzing both solar and wind power systems using Physics-Informed Neural Networks.

### Key Features
- Data generation for solar and wind systems
- Separate model initialization for each system type
- Training and evaluation
- Power output forecasting
- Results visualization and saving

### Usage Example
```python
from src.analysis.renewable_analysis import main

# Run complete analysis
main()

# Or use individual components
from src.analysis.renewable_analysis import (
    generate_data,
    initialize_models,
    train_models,
    evaluate_models
)

# Generate data
solar_data, wind_data = generate_data()

# Initialize and train models
solar_model, wind_model, config = initialize_models()
solar_history, wind_history = train_models(
    solar_model, wind_model, solar_data, wind_data, config
)

# Evaluate models
solar_results, wind_results = evaluate_models(
    solar_model, wind_model, solar_data[1], wind_data[1]
)
```

### Configuration
The module uses configuration from `config/renewable_config.yaml`:
```yaml
model:
  layers: [64, 128, 64]
  activation: 'relu'
  dropout_rate: 0.2

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

## Common Utilities

Both analysis modules share common utilities for:
- Directory setup
- Data generation
- Model training
- Results visualization
- Metrics calculation

### Directory Structure
```
project/
├── src/
│   └── analysis/
│       ├── power_grid_analysis.py
│       └── renewable_analysis.py
├── tests/
│   ├── test_power_grid_analysis.py
│   └── test_renewable_analysis.py
├── config/
│   ├── power_grid_config.yaml
│   └── renewable_config.yaml
├── results/
├── logs/
└── models/
```

## Testing

The analysis modules include comprehensive test suites to ensure functionality and reliability.

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_power_grid_analysis.py
python -m pytest tests/test_renewable_analysis.py

# Run with coverage
python -m pytest --cov=src.analysis tests/
```

### Test Coverage
The test suites cover:
- Data generation
- Model initialization
- Training process
- Evaluation metrics
- Forecast generation
- Directory management
- Configuration handling

### Continuous Integration
Tests are automatically run in the CI/CD pipeline for:
- Code changes
- Pull requests
- Merges to main branch

## Best Practices

1. **Data Generation**
   - Use appropriate random seeds for reproducibility
   - Validate data ranges and distributions
   - Include data preprocessing steps

2. **Model Training**
   - Monitor training progress with TensorBoard
   - Save model checkpoints
   - Implement early stopping

3. **Evaluation**
   - Use multiple metrics for comprehensive assessment
   - Generate visualizations for results
   - Save all results for future reference

4. **Code Organization**
   - Follow PEP 8 style guide
   - Include comprehensive docstrings
   - Use type hints for better code clarity

## Contributing

When contributing to the analysis modules:
1. Add tests for new functionality
2. Update documentation
3. Follow the existing code style
4. Include example usage
5. Update configuration templates if needed 