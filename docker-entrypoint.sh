#!/bin/bash
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python; then
    echo "Error: Python is not installed"
    exit 1
fi

# Check if required directories exist
for dir in "$CONFIG_DIR" "$MODEL_DIR" "$LOG_DIR" "$DATA_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# Check if config files exist
if [ ! -f "$CONFIG_DIR/power_grid_config.yaml" ]; then
    echo "Warning: power_grid_config.yaml not found in $CONFIG_DIR"
fi

if [ ! -f "$CONFIG_DIR/renewable_config.yaml" ]; then
    echo "Warning: renewable_config.yaml not found in $CONFIG_DIR"
fi

# Set Python path
export PYTHONPATH=/app:$PYTHONPATH

# Execute the command
exec "$@" 