#!/bin/bash

# This script is used to run the pipeline.py Python script.

echo "Starting the Python pipeline script..."

VENV_DIR=".lvenv"

# Create a virtual environment using uv venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with uv venv..."
    uv venv "$VENV_DIR"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR"/bin/activate
echo "Virtual environment activated."

# Install necessary libraries using uv pip
echo "Installing necessary libraries..."
uv sync --active
echo "Libraries installed."

source .lvenv/bin/activate

# Main pipeline is run for indexing and evaluation
echo "Running main pipeline"
python3 main.py

# Running Fast API in Background
echo "Running FastAPI app..."
uvicorn src.app_backend:app --host 0.0.0.0 --port 8000 &

# Running Streamlit in Background
echo "Running Streamlit app..."
streamlit run src/app_frontend.py &

wait

deactivate

echo "Python pipeline script finished."