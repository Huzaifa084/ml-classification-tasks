#!/bin/bash
# This script provides a guided walkthrough of the MNIST and Iris classification project

clear
echo "====================================================================="
echo "          MNIST and Iris Classification Project Walkthrough           "
echo "====================================================================="
echo ""
sleep 1

echo "1. Project Overview"
echo "-------------------"
echo "This project demonstrates machine learning classification using two datasets:"
echo "  - MNIST: Handwritten digit recognition"
echo "  - Iris: Multi-class flower classification"
echo ""
sleep 2

echo "2. Setup Environment"
echo "-------------------"
echo "First, let's ensure our environment is properly set up:"
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "Installing required packages..."
pip install -r requirements.txt
echo ""
sleep 2

echo "3. Exploring the Datasets"
echo "------------------------"
echo "MNIST Dataset: 70,000 handwritten digit images (28x28 pixels)"
echo "Iris Dataset: 150 samples of iris flowers with 4 features each"
echo ""
sleep 2

echo "4. Classification Models"
echo "-----------------------"
echo "For MNIST: LinearRegressionClassifier with optimized hyperparameters"
echo "For Iris: Comparing LinearRegression, Ridge, and Lasso models"
echo ""
sleep 2

echo "5. Results Preview"
echo "-----------------"
echo "Let's look at the visualizations we've generated:"
echo ""

if [ -f "mnist_predictions.png" ]; then
    echo "MNIST Predictions Visualization (mnist_predictions.png)"
    echo "This shows sample predictions compared to actual digits"
    echo ""
fi

if [ -f "iris_pairplot.png" ]; then
    echo "Iris Dataset Visualization (iris_pairplot.png)"
    echo "This shows the relationships between different features"
    echo ""
fi

if [ -f "iris_model_comparison.png" ]; then
    echo "Iris Model Comparison (iris_model_comparison.png)"
    echo "This compares the performance of different regression models"
    echo ""
fi

if [ -f "iris_predictions.png" ]; then
    echo "Iris Predictions Visualization (iris_predictions.png)"
    echo "This shows how well our best model predicted iris classes"
    echo ""
fi
sleep 2

echo "6. Live Demonstration"
echo "--------------------"
echo "Would you like to run the classifiers now? (y/n)"
read -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running MNIST classifier (this may take a few minutes)..."
    python mnist_classifier.py
    
    echo ""
    echo "Running Iris classifier..."
    python iris_classifier.py
    
    echo ""
    echo "Classifiers completed! Check the output visualizations."
else
    echo "Skipping live demonstration."
fi
echo ""
sleep 1

echo "7. Conclusion"
echo "------------"
echo "Key takeaways:"
echo "  - MNIST: Achieved high accuracy using hyperparameter optimization"
echo "  - Iris: Successfully compared multiple regression models"
echo "  - Generated informative visualizations for both tasks"
echo ""
echo "Thank you for watching this demonstration!"
echo "====================================================================="
