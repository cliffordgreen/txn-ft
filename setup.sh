#!/bin/bash
# Setup script for ft-txn project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Setup complete. Activate the environment with: source venv/bin/activate"