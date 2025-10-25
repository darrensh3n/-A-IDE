#!/bin/bash

# Backend Setup Script
# This script sets up the Python virtual environment for the backend

echo "=========================================="
echo "Backend Setup Script"
echo "=========================================="

# Check if Python 3.13 is available
if ! command -v python3.13 &> /dev/null; then
    echo "Error: Python 3.13 is not installed or not in PATH"
    echo "Please install Python 3.13 to use this backend"
    exit 1
fi

echo "Python version: $(python3.13 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment with Python 3.13..."
    python3.13 -m venv .venv
    echo "✓ Virtual environment created at .venv/"
else
    echo ""
    echo "Virtual environment already exists at .venv/"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --quiet --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies from requirements.txt..."
pip install --quiet -r requirements.txt

echo ""
echo "Installing dependencies from scripts/requirements.txt..."
pip install --quiet -r scripts/requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment manually, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the backend server, run:"
echo "  python api_server.py"
echo ""
echo "Or use the main startup script from the root directory:"
echo "  ../start.sh"
echo ""
