#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the classifiers
echo "Running MNIST classifier..."
python mnist_classifier.py

echo ""
echo "Running Iris classifier..."
python iris_classifier.py

# Deactivate virtual environment
deactivate
