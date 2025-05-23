# -*- coding: utf-8 -*-
"""4

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JJPI5_VtxqMlRqu9eXHSwu_Gepa6YZyY
"""

# Logistic Regression Implementation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.datasets import make_classification, load_iris, load_breast_cancer, load_wine
import seaborn as sns

# Part 1: Standard Logistic Regression Implementation
# ===================================================

print("=== Logistic Regression Implementation ===")

# Create a synthetic dataset for binary classification
X, y = make_classification(
    n_samples=1000,            # Number of samples
    n_features=10,             # Number of features
    n_informative=5,           # Number of informative features
    n_redundant=2,             # Number of redundant features
    n_repeated=0,              # Number of repeated features
    n_classes=2,               # Number of classes (binary classification)
    random_state=42            # For reproducibility
)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a logistic regression model
# The 'liblinear' solver is efficient for small datasets
# C is the inverse of regularization strength; smaller values indicate stronger regularization
log_reg = LogisticRegression(solver='liblinear', C=1.0, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test_scaled)
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Get model coefficients and intercept
print("\nModel Coefficients (weights):")
print(log_reg.coef_)
print("\nIntercept (bias):")
print(log_reg.intercept_)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Part 2: Custom Implementation of Logistic Regression
# ===================================================

class CustomLogisticRegression:
    """
    Custom implementation of Logistic Regression with gradient descent optimization.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.costs = []

    def sigmoid(self, z):
        """Compute sigmoid activation."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the logistic regression model.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            
            # Compute and store cost
            cost = -np.mean(y * np.log(predictions + 1e-15) + 
                          (1-y) * np.log(1 - predictions + 1e-15))
            self.costs.append(cost)

    def predict_proba(self, X):
        """Predict class probabilities."""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def plot_decision_boundary(self, X, y, title="Decision Boundary"):
        """Plot the decision boundary."""
        h = 0.02  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.show()

# Train our custom logistic regression model
custom_log_reg = CustomLogisticRegression(learning_rate=0.01, n_iterations=1000)
custom_log_reg.fit(X_train_scaled, y_train)

# Make predictions using our custom model
custom_y_pred = custom_log_reg.predict(X_test_scaled)
custom_y_pred_prob = custom_log_reg.predict_proba(X_test_scaled)

# Evaluate our custom model
custom_accuracy = accuracy_score(y_test, custom_y_pred)
custom_conf_matrix = confusion_matrix(y_test, custom_y_pred)
custom_class_report = classification_report(y_test, custom_y_pred)

print("\n=== Custom Logistic Regression Results ===")
print(f"Accuracy: {custom_accuracy:.4f}")
print("\nConfusion Matrix:")
print(custom_conf_matrix)
print("\nClassification Report:")
print(custom_class_report)

# Plot training cost history
plt.figure(figsize=(10, 6))
plt.plot(range(custom_log_reg.n_iterations), custom_log_reg.costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.grid(True)
plt.show()

# Plot decision boundary for 2D case
def plot_decision_boundary(X, y, model, custom=False):
    # This only works for 2D data, so we'll take just the first two features
    X_subset = X[:, :2]

    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
    y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Create a mesh grid point matrix
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # For the remaining features, we'll use zeros (for visualization purposes)
    if X.shape[1] > 2:
        grid_points = np.hstack([grid_points, np.zeros((grid_points.shape[0], X.shape[1] - 2))])

    # Make predictions on the mesh grid
    if custom:
        Z = model.predict(grid_points)
    else:
        Z = model.predict(grid_points)

    # Reshape Z back to the grid shape
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y, s=40, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary' + (' (Custom Implementation)' if custom else ''))
    plt.colorbar()
    plt.show()

# Plot decision boundaries (only using the first 2 features for visualization)
plot_decision_boundary(X_scaled := scaler.transform(X), y, log_reg, custom=False)
plot_decision_boundary(X_scaled, y, custom_log_reg, custom=True)

# Part 3: Working with Custom Datasets
# ===================================

def load_data(use_synthetic=True, file_path=None):
    """
    Load data either from a file or generate synthetic data.
    
    Args:
        use_synthetic: If True, generate synthetic data
        file_path: Path to CSV file if use_synthetic is False
    
    Returns:
        X: Features
        y: Target values
    """
    if not use_synthetic and file_path:
        data = pd.read_csv(file_path)
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]
    else:
        X, y = make_classification(
            n_samples=1000, n_features=5, n_informative=3, n_redundant=1,
            n_classes=2, random_state=42
        )
    
    return X, y

# Example usage of the load_data function
print("\n=== Working with Custom Datasets ===")
print("Using synthetic data for demonstration:")
X_custom, y_custom = load_data(use_synthetic=True)

# Split the custom dataset
X_custom_train, X_custom_test, y_custom_train, y_custom_test = train_test_split(
    X_custom, y_custom, test_size=0.3, random_state=42
)

# Scale the features
custom_scaler = StandardScaler()
X_custom_train_scaled = custom_scaler.fit_transform(X_custom_train)
X_custom_test_scaled = custom_scaler.transform(X_custom_test)

# Train a logistic regression model on the custom dataset
custom_dataset_model = LogisticRegression(solver='liblinear', C=1.0, random_state=42)
custom_dataset_model.fit(X_custom_train_scaled, y_custom_train)

# Make predictions
custom_dataset_pred = custom_dataset_model.predict(X_custom_test_scaled)
custom_dataset_pred_prob = custom_dataset_model.predict_proba(X_custom_test_scaled)[:, 1]

# Evaluate the model
custom_dataset_accuracy = accuracy_score(y_custom_test, custom_dataset_pred)
custom_dataset_conf_matrix = confusion_matrix(y_custom_test, custom_dataset_pred)
custom_dataset_class_report = classification_report(y_custom_test, custom_dataset_pred)

print("\n=== Custom Dataset Results ===")
print(f"Accuracy: {custom_dataset_accuracy:.4f}")
print("\nConfusion Matrix:")
print(custom_dataset_conf_matrix)
print("\nClassification Report:")
print(custom_dataset_class_report)

# Plot ROC curve for the custom dataset model
custom_dataset_fpr, custom_dataset_tpr, _ = roc_curve(y_custom_test, custom_dataset_pred_prob)
custom_dataset_roc_auc = auc(custom_dataset_fpr, custom_dataset_tpr)

plt.figure(figsize=(10, 6))
plt.plot(custom_dataset_fpr, custom_dataset_tpr, color='darkorange', lw=2,
         label=f'ROC curve (area = {custom_dataset_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Custom Dataset)')
plt.legend(loc="lower right")
plt.show()

# Part 4: Real-world datasets examples
# ======================================

# Function to load and prepare some common real-world datasets
def load_real_world_dataset(dataset_name):
    """
    Load a real-world dataset from scikit-learn

    Parameters:
    dataset_name (str): Name of the dataset ('iris', 'breast_cancer', or 'wine')

    Returns:
    X (numpy.ndarray): Feature matrix
    y (numpy.ndarray): Target vector
    """
    from sklearn.datasets import load_iris, load_breast_cancer, load_wine

    if dataset_name == 'iris':
        # Iris dataset (multi-class classification)
        data = load_iris()
        # For binary classification, we'll only use two classes
        indices = np.where(data.target < 2)[0]
        X = data.data[indices]
        y = data.target[indices]
        print("Loaded Iris dataset (classes: setosa vs. versicolor)")

    elif dataset_name == 'breast_cancer':
        # Breast Cancer dataset (binary classification)
        data = load_breast_cancer()
        X = data.data
        y = data.target
        print("Loaded Breast Cancer dataset")

    elif dataset_name == 'wine':
        # Wine dataset (multi-class classification)
        data = load_wine()
        # For binary classification, we'll only use two classes
        indices = np.where(data.target < 2)[0]
        X = data.data[indices]
        y = data.target[indices]
        print("Loaded Wine dataset (binary classification)")

    else:
        raise ValueError("Dataset name not recognized. Choose from 'iris', 'breast_cancer', or 'wine'")

    return X, y

# Example of using a real-world dataset
print("\n=== Using Real-World Datasets ===")
print("Available datasets: 'iris', 'breast_cancer', 'wine'")
print("Example with Breast Cancer dataset:")

# Load breast cancer dataset
X_real, y_real = load_real_world_dataset('breast_cancer')

# Split the dataset
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    X_real, y_real, test_size=0.3, random_state=42
)

# Scale the features
real_scaler = StandardScaler()
X_real_train_scaled = real_scaler.fit_transform(X_real_train)
X_real_test_scaled = real_scaler.transform(X_real_test)

# Train with different regularization strengths
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
accuracies = []

plt.figure(figsize=(12, 6))

for i, C in enumerate(C_values):
    # Train model with different regularization strength
    real_model = LogisticRegression(solver='liblinear', C=C, random_state=42)
    real_model.fit(X_real_train_scaled, y_real_train)

    # Evaluate
    real_pred = real_model.predict(X_real_test_scaled)
    real_accuracy = accuracy_score(y_real_test, real_pred)
    accuracies.append(real_accuracy)

    # Plot decision regions (using first two features)
    if i < 4:  # Only plot first 4 regularization strengths to avoid clutter
        plt.subplot(2, 2, i+1)
        # Create a mesh grid for the first two features
        h = 0.02
        x_min, x_max = X_real_train_scaled[:, 0].min() - 1, X_real_train_scaled[:, 0].max() + 1
        y_min, y_max = X_real_train_scaled[:, 1].min() - 1, X_real_train_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Create prediction grid
        grid = np.c_[xx.ravel(), yy.ravel()]
        # Add zeros for remaining features
        if X_real.shape[1] > 2:
            grid = np.hstack([grid, np.zeros((grid.shape[0], X_real.shape[1] - 2))])

        Z = real_model.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X_real_train_scaled[:, 0], X_real_train_scaled[:, 1], c=y_real_train,
                   cmap=plt.cm.coolwarm, edgecolors='k')
        plt.title(f'C={C}, Accuracy={real_accuracy:.4f}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Plot regularization vs. accuracy
plt.figure(figsize=(10, 6))
plt.plot(C_values, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy')
plt.title('Effect of Regularization on Model Accuracy')
plt.grid(True)
plt.show()

# Print final results on the best model
best_C_index = np.argmax(accuracies)
best_C = C_values[best_C_index]
print(f"\nBest regularization parameter C: {best_C}")
print(f"Best accuracy: {accuracies[best_C_index]:.4f}")

# Train the final model with the best regularization
final_model = LogisticRegression(solver='liblinear', C=best_C, random_state=42)
final_model.fit(X_real_train_scaled, y_real_train)

# Final evaluation
final_pred = final_model.predict(X_real_test_scaled)
final_pred_prob = final_model.predict_proba(X_real_test_scaled)[:, 1]
final_accuracy = accuracy_score(y_real_test, final_pred)
final_conf_matrix = confusion_matrix(y_real_test, final_pred)
final_class_report = classification_report(y_real_test, final_pred)

print("\n=== Final Model Results (Best Regularization) ===")
print(f"Accuracy: {final_accuracy:.4f}")
print("\nConfusion Matrix:")
print(final_conf_matrix)
print("\nClassification Report:")
print(final_class_report)

# Plot final ROC curve
final_fpr, final_tpr, _ = roc_curve(y_real_test, final_pred_prob)
final_roc_auc = auc(final_fpr, final_tpr)

plt.figure(figsize=(10, 6))
plt.plot(final_fpr, final_tpr, color='darkorange', lw=2,
         label=f'ROC curve (area = {final_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Final ROC Curve (Best Model)')
plt.legend(loc="lower right")
plt.show()

print("\n=== Logistic Regression Implementation Complete ===")

