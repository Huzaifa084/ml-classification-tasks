import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Loading MNIST dataset...")
# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

# Convert labels to integers
y = y.astype(np.uint8)

# Split data into train and test sets (using the predefined split)
# Using a small subset for faster execution
X_train, X_test = X[:6000], X[60000:61000]
y_train, y_test = y[:6000], y[60000:61000]

# Further split training data to have a validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Preprocess data
print("Preprocessing data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define parameter grid for grid search
param_grid = {
    'alpha': [0.001, 0.01],  # Reduced alpha values for faster execution
    'penalty': ['l2'],  # Only using L2 penalty for faster execution
    'max_iter': [1000],
    'tol': [1e-3],
}

# Initialize SGDClassifier (used for linear regression classification)
sgd_clf = SGDClassifier(loss='log_loss', random_state=42, n_jobs=-1)

# Perform grid search with cross-validation
print("Performing grid search for hyperparameter tuning...")
grid_search = GridSearchCV(
    sgd_clf, param_grid, cv=3, scoring='accuracy', verbose=1
)

start_time = time.time()
grid_search.fit(X_train, y_train)
training_time = time.time() - start_time

# Get best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best parameters: {best_params}")
print(f"Training time: {training_time:.2f} seconds")

# Evaluate on validation set
val_accuracy = accuracy_score(y_val, best_model.predict(X_val))
print(f"Validation accuracy: {val_accuracy:.4f}")

# Evaluate on test set
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
print(f"Test accuracy: {test_accuracy:.4f}")

# If test accuracy < 0.97, train on full training set with best params
if test_accuracy < 0.97:
    print("Test accuracy below 0.97. Retraining on full training set...")
    
    # Combine training and validation data
    X_full_train = np.vstack((X_train, X_val))
    y_full_train = np.concatenate((y_train, y_val))
    
    # Train final model with best parameters
    final_model = SGDClassifier(
        loss='log_loss',
        alpha=best_params['alpha'],
        penalty=best_params['penalty'],
        max_iter=best_params['max_iter'],
        tol=best_params['tol'],
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    final_model.fit(X_full_train, y_full_train)
    final_training_time = time.time() - start_time
    
    # Evaluate final model on test set
    final_test_accuracy = accuracy_score(y_test, final_model.predict(X_test))
    print(f"Final model training time: {final_training_time:.2f} seconds")
    print(f"Final test accuracy: {final_test_accuracy:.4f}")
    
    if final_test_accuracy >= 0.97:
        print("✓ Successfully achieved test accuracy over 97%!")
    else:
        print("Failed to achieve test accuracy over 97%. Consider adjusting hyperparameters further.")
else:
    print("✓ Successfully achieved test accuracy over 97%!")

# Visualize some predictions
def plot_predictions(X, y_true, y_pred, n_samples=10):
    plt.figure(figsize=(20, 4))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_true[i]}\nPred: {y_pred[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    plt.close()

# Sample a few test examples to visualize
indices = np.random.choice(len(X_test), 10, replace=False)
X_sample = X_test[indices]
y_true_sample = y_test[indices]
y_pred_sample = best_model.predict(X_sample)

# Inverse transform to get original scale for visualization
X_sample_original = scaler.inverse_transform(X_sample)
plot_predictions(X_sample_original, y_true_sample, y_pred_sample)

print("Predictions visualization saved as 'mnist_predictions.png'")
