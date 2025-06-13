# Analysis Modules

This directory contains analysis modules for Physics-Informed Neural Networks (PINNs) applied to power systems.

## Modules

### Power Grid Analysis
- `power_grid_analysis.py`: Analysis of power grid systems using PINNs
- Features:
  - Voltage stability prediction
  - Power flow analysis
  - Grid state estimation
  - Load forecasting

### Renewable Energy Analysis
- `renewable_analysis.py`: Analysis of renewable energy systems using PINNs
- Features:
  - Solar power prediction
  - Wind power prediction
  - Weather impact analysis
  - Power output forecasting

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run analysis:
```bash
# Power grid analysis
python -m src.analysis.power_grid_analysis

# Renewable energy analysis
python -m src.analysis.renewable_analysis
```

3. Run tests:
```bash
python -m pytest tests/test_power_grid_analysis.py
python -m pytest tests/test_renewable_analysis.py
```

## Configuration

Configuration files are located in the `config/` directory:
- `power_grid_config.yaml`: Power grid analysis settings
- `renewable_config.yaml`: Renewable energy analysis settings

## Results

Results are saved in the following directories:
- `results/`: Analysis results and visualizations
- `logs/`: Training logs and TensorBoard files
- `models/`: Saved model checkpoints

## Documentation

For detailed documentation, see:
- [Analysis Modules Documentation](../docs/analysis_modules.md)
- [API Reference](../docs/api_reference.md)
- [Examples](../docs/examples.md) 