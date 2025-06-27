# Physics-Informed Neural Network (PINN) for Power Grid and Renewable Energy

This repository implements Physics-Informed Neural Networks (PINNs) for power grid analysis and renewable energy applications. The project combines deep learning with physical constraints to solve complex engineering problems.

## Features

- **Power Grid Analysis**
  - Voltage stability prediction
  - Power flow equations
  - Grid state estimation
  - Physics-constrained learning

- **Renewable Energy Applications**
  - Solar power generation prediction
  - Wind power forecasting
  - System efficiency optimization
  - Environmental constraints integration

## Installation

### Using pip
```bash
pip install -r requirements.txt
pip install -e .
```

### Using Docker
```bash
docker build -t pinn .
docker run -it pinn
```

## Usage

### Training Models
```bash
# Train power grid model
python -m src.cli.train --config config/power_grid_config.yaml

# Train renewable energy model
python -m src.cli.train --config config/renewable_config.yaml
```

### Making Predictions
```bash
# Predict power grid state
python -m src.cli.predict --model models/power_grid_model.h5 --input data/input.json --output predictions.json

# Predict renewable energy output
python -m src.cli.predict --model models/renewable_model.h5 --input data/input.json --output predictions.json
```

### Development
```bash
# Install development dependencies
make install

# Run tests
make test

# Format code
make format

# Run linting
make lint
```

## Project Structure

```
physics_informed_neural_network/
├── config/                 # Configuration files
├── data/                   # Data storage
├── models/                 # Saved models
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── cli/              # Command-line interfaces
│   ├── config/           # Configuration management
│   ├── models/           # Model implementations
│   ├── schemas/          # Data schemas
│   └── utils/            # Utility functions
├── tests/                # Test files
├── Dockerfile            # Docker configuration
├── Makefile             # Development commands
└── requirements.txt      # Dependencies
```

## Development

### Setting up the Development Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
```

### Running Tests
```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_power_grid.py
```

### Code Quality
```bash
# Format code
make format

# Run linting
make lint
```

## Docker Support

### Building the Image
```bash
docker build -t pinn .
```

### Running with Docker Compose
```bash
# Start development environment
docker-compose up

# Run tests
docker-compose run tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is dual-licensed.

• NON-COMMERCIAL USE → PolyForm Noncommercial License 1.0.0  
  (see NONCOMMERCIAL_LICENSE)

• COMMERCIAL USE → Parallel LLC Commercial License v1.0  
  (see COMMERCIAL_LICENSE).  
  To obtain a paid commercial license, e-mail <edwinsalguero@parallelLLC.com>.

© 2025 Edwin Salguero

## Citation

If you use this code in your research, please cite:

```bibtex
@software{physics_informed_neural_network,
  author = {Your Name},
  title = {Physics-Informed Neural Network for Power Grid and Renewable Energy},
  year = {2024},
  url = {https://github.com/EAName/physics_informed_neural_network}
}
```
