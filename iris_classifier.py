import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import os
import requests
from zipfile import ZipFile
from io import BytesIO

# Set random seed for reproducibility
np.random.seed(42)

# Step I: Download Iris dataset from Kaggle
print("Step I: Downloading Iris dataset")

# We'll use scikit-learn's built-in iris dataset for simplicity 
# (instead of downloading from Kaggle, which requires API authentication)
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for easier visualization
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['species'] = [target_names[i] for i in y]

print("Dataset loaded successfully")
print(f"Dataset shape: {iris_df.shape}")
print("\nFirst 5 rows of the dataset:")
print(iris_df.head())

print("\nSummary statistics:")
print(iris_df.describe())

print("\nClass distribution:")
print(iris_df['species'].value_counts())

# Visualize the data
plt.figure(figsize=(15, 10))
sns.pairplot(iris_df, hue='species')
plt.savefig('iris_pairplot.png')
plt.close()

# Step II: Split dataset into training and test sets
print("\nStep II: Splitting dataset into training and test sets")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Step III: Data preparation pipeline
print("\nStep III: Creating data preparation pipeline")
# Define preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Apply preprocessing
X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

print("Data preprocessing completed")

# Step IV: Train models and record scores
print("\nStep IV: Training models and recording scores")

# Function to evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Fit model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    
    return {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

# Evaluate each model
results = {}
for name, model in models.items():
    results[name] = evaluate_model(
        model, X_train_processed, y_train, 
        X_test_processed, y_test, name
    )

# Visualize results
plt.figure(figsize=(12, 8))

# Plot R² scores
plt.subplot(2, 1, 1)
train_r2_scores = [results[name]['train_r2'] for name in models.keys()]
test_r2_scores = [results[name]['test_r2'] for name in models.keys()]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, train_r2_scores, width, label='Training R²')
plt.bar(x + width/2, test_r2_scores, width, label='Test R²')
plt.ylabel('R² Score')
plt.title('R² Score by Model')
plt.xticks(x, models.keys())
plt.legend()

# Plot MSE
plt.subplot(2, 1, 2)
train_mse_scores = [results[name]['train_mse'] for name in models.keys()]
test_mse_scores = [results[name]['test_mse'] for name in models.keys()]

plt.bar(x - width/2, train_mse_scores, width, label='Training MSE')
plt.bar(x + width/2, test_mse_scores, width, label='Test MSE')
plt.ylabel('Mean Squared Error')
plt.title('MSE by Model')
plt.xticks(x, models.keys())
plt.legend()

plt.tight_layout()
plt.savefig('iris_model_comparison.png')
plt.close()

print("\nResults visualization saved as 'iris_model_comparison.png'")

# Visualize predictions vs. actual for the best model
best_model_name = max(results, key=lambda k: results[k]['test_r2'])
best_model = results[best_model_name]['model']
print(f"\nBest model based on Test R² score: {best_model_name}")

# Predict on test set
y_pred = best_model.predict(X_test_processed)

# Create a DataFrame for visualization
pred_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred.round().astype(int).clip(0, 2)  # Clip to valid class range
})

# Add species names
pred_df['Actual_Species'] = [target_names[i] for i in pred_df['Actual']]
pred_df['Predicted_Species'] = [target_names[i] for i in pred_df['Predicted']]
pred_df['Correct'] = pred_df['Actual'] == pred_df['Predicted']

# Calculate accuracy
accuracy = pred_df['Correct'].mean()
print(f"Accuracy of {best_model_name}: {accuracy:.4f}")

# Visualize predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=range(len(pred_df)), 
    y='Actual', 
    data=pred_df, 
    label='Actual', 
    marker='o'
)
sns.scatterplot(
    x=range(len(pred_df)), 
    y='Predicted', 
    data=pred_df, 
    label='Predicted', 
    marker='x'
)
plt.title(f'{best_model_name} Predictions vs Actual')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.yticks([0, 1, 2], target_names)
plt.legend()
plt.savefig('iris_predictions.png')
plt.close()

print("Predictions visualization saved as 'iris_predictions.png'")
