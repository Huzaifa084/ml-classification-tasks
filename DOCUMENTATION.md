# MNIST and Iris Classification Project

This documentation provides detailed instructions on how to set up and run the MNIST and Iris classification tasks.

## Project Overview

This project implements two machine learning classification tasks:

1. **MNIST Digit Classification**: Achieving over 97% accuracy on the test set using a LinearRegressionClassifier with optimized hyperparameters.

2. **Iris Multi-class Classification**: Training and comparing LinearRegression, Ridge, and Lasso models on the Iris dataset.

## Project Structure

```
ML_Classification_Task/
├── iris_classifier.py         # Iris dataset classifier implementation
├── iris_model_comparison.png  # Performance comparison of models on Iris dataset
├── iris_pairplot.png          # Visualization of Iris dataset features
├── iris_predictions.png       # Visualization of predictions vs. actual values
├── mnist_classifier.py        # MNIST dataset classifier implementation
├── mnist_predictions.png      # Visualization of MNIST predictions
├── README.md                  # Brief project overview
├── requirements.txt           # Project dependencies
├── run_classifiers.sh         # Shell script to run both classifiers
└── simple_mnist_classifier.py # Simplified MNIST classifier using digits dataset
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for version control)

## Setup Instructions

### 1. Clone or Download the Repository

If using Git:
```bash
git clone https://github.com/Huzaifa084/ml-classification-tasks
cd ML_Classification_Task
```

Or simply download and extract the project files to your preferred location.

### 2. Create a Virtual Environment

This isolates the project dependencies from your system Python:

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

On Linux/macOS:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Classifiers

### Option 1: Using the Shell Script (Linux/macOS)

For convenience, you can run both classifiers using the provided shell script:

```bash
./run_classifiers.sh
```

If you encounter permission issues, make the script executable:
```bash
chmod +x run_classifiers.sh
```

### Option 2: Running Individual Classifiers

To run each classifier separately:

```bash
# For MNIST classification
python mnist_classifier.py

# For Iris classification
python iris_classifier.py

# For simplified MNIST classification (faster execution)
python simple_mnist_classifier.py
```

## Understanding the Output

### MNIST Classification

The MNIST classifier:
1. Loads the MNIST dataset
2. Splits it into training, validation, and test sets
3. Preprocesses the data using StandardScaler
4. Performs grid search to find optimal hyperparameters
5. Trains a model using the best parameters
6. Evaluates performance on the test set
7. Generates visualizations of predictions

Output: `mnist_predictions.png` - Visualization of predicted vs. actual digits

### Iris Classification

The Iris classifier:
1. Loads the Iris dataset
2. Visualizes the dataset features
3. Splits data into training and test sets
4. Creates a data preparation pipeline
5. Trains LinearRegression, Ridge, and Lasso models
6. Compares model performance
7. Generates visualizations

Outputs:
- `iris_pairplot.png` - Visualization of the dataset features
- `iris_model_comparison.png` - Comparison of model performances
- `iris_predictions.png` - Visualization of predictions vs. actual values

## Presentation Tips

When presenting this project:

1. **Start with the Problem Statement**:
   - Explain the classification challenges for both MNIST and Iris datasets
   - Mention the goal of achieving >97% accuracy for MNIST

2. **Explain the Approach**:
   - Discuss hyperparameter tuning using grid search
   - Explain the preprocessing steps
   - Highlight the model selection process

3. **Show the Results**:
   - Display the generated visualizations
   - Discuss the performance metrics
   - Compare different models (especially for Iris classification)

4. **Demonstrate Live Execution** (if time permits):
   - Use the shell script to run both classifiers
   - Show the generated output files

5. **Discuss Potential Improvements**:
   - Mention alternative models that could be tried
   - Suggest additional preprocessing techniques
   - Discuss ways to improve accuracy further

## Troubleshooting

### Common Issues and Solutions

1. **ModuleNotFoundError**:
   - Ensure you've activated the virtual environment
   - Verify all dependencies are installed: `pip install -r requirements.txt`

2. **Dataset Download Issues**:
   - Check your internet connection
   - For MNIST, the dataset might be temporarily unavailable from OpenML
   - Use `simple_mnist_classifier.py` as a fallback

3. **Memory Errors**:
   - The MNIST dataset is large and might cause memory issues
   - Reduce the dataset size by modifying the code to use a smaller subset
   - Use `simple_mnist_classifier.py` which uses a smaller dataset

4. **Slow Execution**:
   - Grid search is computationally intensive
   - Reduce the hyperparameter search space
   - Use a smaller subset of the dataset for faster execution

## Contact

For any questions or issues regarding this project, please contact:
[Your Huzaifa Naseer/huzaifanaseer084@gmail.com]
