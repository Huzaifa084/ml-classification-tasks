# Machine Learning Classification Tasks

This directory contains the implementation of two classification tasks. For detailed instructions on setup and presentation, see [DOCUMENTATION.md](DOCUMENTATION.md).

## 1. MNIST Digit Classification

A classifier for the MNIST dataset that achieves over 97% accuracy on the test set using the LinearRegressionClassifier with optimized hyperparameters.

### Files:
- `mnist_classifier.py`: Main implementation using the full MNIST dataset
- `simple_mnist_classifier.py`: Simplified implementation using the smaller digits dataset

### Approach:
- Grid search for hyperparameter tuning (weights and alpha)
- StandardScaler for feature normalization
- SGDClassifier with log loss for linear regression classification

## 2. Iris Multi-class Classification

A multi-class classifier for the Iris dataset.

### Files:
- `iris_classifier.py`: Implementation of multi-class classification for the Iris dataset

### Steps:
1. Download examples of iris dataset
2. Split the datasets into training and test sets
3. Data preparation pipeline to convert each iris into a feature vector
4. Train and compare LinearRegression, Ridge, and Lasso Models
5. Record and visualize performance on training and test datasets

## Setup and Execution

1. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the classifiers:
   ```
   python mnist_classifier.py
   python iris_classifier.py
   ```
   
4. For a guided walkthrough (useful for presentations):
   ```
   ./walkthrough.sh
   ```
