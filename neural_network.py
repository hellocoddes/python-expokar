# -*- coding: utf-8 -*-
"""8

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JJPI5_VtxqMlRqu9eXHSwu_Gepa6YZyY
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Set random seed for reproducibility
np.random.seed(42)

# ============================
# Single Layer Perceptron
# ============================

class SingleLayerPerceptron:
    def __init__(self, input_size, learning_rate=0.01, max_epochs=1000):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.errors = []

    def step_activation(self, x):
        return 1 if x >= 0 else 0

    def predict_single(self, x):
        # Add bias term
        x_with_bias = np.insert(x, 0, 1)
        # Calculate net input
        z = np.dot(x_with_bias, self.weights)
        # Apply activation function
        return self.step_activation(z)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    def fit(self, X, y, verbose=False):
        start_time = time.time()
        n_samples = X.shape[0]
        self.errors = []

        for epoch in range(self.max_epochs):
            errors = 0

            # Iterate through each training example
            for i in range(n_samples):
                x_i = X[i]
                # Add bias term
                x_i_with_bias = np.insert(x_i, 0, 1)

                # Make prediction
                y_pred = self.predict_single(x_i)

                # Calculate error
                error = y[i] - y_pred

                # Update weights only if there's an error
                if error != 0:
                    self.weights += self.learning_rate * error * x_i_with_bias
                    errors += 1

            # Store number of misclassifications for this epoch
            self.errors.append(errors)

            # Early stopping if perfect classification
            if errors == 0:
                if verbose:
                    print(f"Converged after {epoch+1} epochs")
                break

        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.4f} seconds")
            if epoch == self.max_epochs - 1:
                print(f"Warning: Maximum number of epochs ({self.max_epochs}) reached without convergence")

        return self

# ============================
# Multilayer Perceptron with Backpropagation
# ============================

class BackpropagationNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1,
                 max_epochs=1000, activation='sigmoid'):
        # Initialize weights with small random values
        self.weights_input_hidden = np.random.randn(input_size + 1, hidden_size) * 0.1  # +1 for bias
        self.weights_hidden_output = np.random.randn(hidden_size + 1, output_size) * 0.1  # +1 for bias

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.errors = []

        # Set activation function
        self.activation = activation
        if activation not in ['sigmoid', 'tanh']:
            raise ValueError("Activation function must be 'sigmoid' or 'tanh'")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clipping to avoid overflow

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    def activate(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        else:  # tanh
            return self.tanh(x)

    def activate_derivative(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        else:  # tanh
            return self.tanh_derivative(x)

    def forward_pass(self, X):
        # Add bias term to input
        X_with_bias = np.insert(X, 0, 1)

        # Hidden layer
        hidden_input = np.dot(X_with_bias, self.weights_input_hidden)
        hidden_output = self.activate(hidden_input)

        # Add bias to hidden layer output
        hidden_output_with_bias = np.insert(hidden_output, 0, 1)

        # Output layer
        output_input = np.dot(hidden_output_with_bias, self.weights_hidden_output)

        # For output layer, always use sigmoid for binary classification
        # This ensures output is between 0 and 1 for probability interpretation
        output = self.sigmoid(output_input)

        return hidden_output, hidden_output_with_bias, output

    def predict(self, X):
        predictions = []
        for x in X:
            _, _, output = self.forward_pass(x)
            # For binary classification
            predictions.append(1 if output >= 0.5 else 0)
        return np.array(predictions)

    def fit(self, X, y, verbose=False):
        start_time = time.time()
        n_samples = X.shape[0]
        self.errors = []

        # Convert y to array of shape (n_samples, 1) for vectorized operations
        y = np.array(y).reshape(-1, 1)

        for epoch in range(self.max_epochs):
            total_error = 0

            # Iterate through each training example
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]

                # Forward pass
                hidden_output, hidden_output_with_bias, output = self.forward_pass(x_i)

                # Calculate error
                output_error = y_i - output
                total_error += np.sum(output_error ** 2)

                # Backward pass
                # Output layer error
                delta_output = output_error * self.sigmoid_derivative(output)

                # Hidden layer error (removing bias)
                delta_hidden = np.dot(delta_output, self.weights_hidden_output[1:, :].T) * self.activate_derivative(hidden_output)

                # Update weights
                # Add bias to input
                x_i_with_bias = np.insert(x_i, 0, 1)

                # Update hidden->output weights
                for j in range(len(self.weights_hidden_output)):
                    for k in range(len(self.weights_hidden_output[j])):
                        self.weights_hidden_output[j, k] += self.learning_rate * delta_output[k] * hidden_output_with_bias[j]

                # Update input->hidden weights
                for j in range(len(self.weights_input_hidden)):
                    for k in range(len(self.weights_input_hidden[j])):
                        self.weights_input_hidden[j, k] += self.learning_rate * delta_hidden[k] * x_i_with_bias[j]

            # Store error for this epoch
            mean_squared_error = total_error / n_samples
            self.errors.append(mean_squared_error)

            # Early stopping if error is very small
            if mean_squared_error < 0.001:
                if verbose:
                    print(f"Converged after {epoch+1} epochs")
                break

        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.4f} seconds")
            if epoch == self.max_epochs - 1:
                print(f"Maximum number of epochs ({self.max_epochs}) reached")

        return self

# ============================
# Example usage for a linearly separable dataset
# ============================

def test_perceptron_linearly_separable():
    # Create a simple linearly separable dataset
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                              n_informative=2, random_state=42, n_clusters_per_class=1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the perceptron
    perceptron = SingleLayerPerceptron(input_size=2, learning_rate=0.01, max_epochs=100)
    perceptron.fit(X_train, y_train, verbose=True)

    # Make predictions
    y_pred = perceptron.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    # Plot decision boundary
    plt.figure(figsize=(10, 6))

    # Plot training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k')
    plt.title("Training Data")

    # Plot test data and decision boundary
    plt.subplot(1, 2, 2)

    # Create a grid to visualize the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Make predictions on the grid
    Z = np.array([perceptron.predict_single(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and test points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
    plt.title(f"Test Data and Decision Boundary\nAccuracy: {accuracy:.4f}")

    plt.tight_layout()
    plt.show()

    # Plot error convergence
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(perceptron.errors)), perceptron.errors)
    plt.xlabel('Epoch')
    plt.ylabel('Number of Misclassifications')
    plt.title('Perceptron Learning Convergence')
    plt.grid(True)
    plt.show()

# ============================
# Example usage for a non-linearly separable dataset with backpropagation (sigmoid vs tanh)
# ============================

def test_backpropagation_nonlinear(activation='sigmoid'):
    # Create a non-linearly separable dataset (moons)
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the neural network
    nn = BackpropagationNeuralNetwork(input_size=2, hidden_size=5, output_size=1,
                                     learning_rate=0.1, max_epochs=2000, activation=activation)
    nn.fit(X_train, y_train, verbose=True)

    # Make predictions
    y_pred = nn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy with {activation} activation: {accuracy:.4f}")

    # Plot decision boundary
    plt.figure(figsize=(10, 6))

    # Plot training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k')
    plt.title(f"Training Data ({activation} activation)")

    # Plot test data and decision boundary
    plt.subplot(1, 2, 2)

    # Create a grid to visualize the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Make predictions on the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and test points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
    plt.title(f"Test Data and Decision Boundary\nAccuracy: {accuracy:.4f}")

    plt.tight_layout()
    plt.show()

    # Plot error convergence
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(nn.errors)), nn.errors)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Backpropagation Learning Convergence ({activation} activation)')
    plt.grid(True)
    plt.show()

    return nn.errors

# ============================
# Compare sigmoid and tanh convergence
# ============================

def compare_activations():
    print("\nComparing sigmoid and tanh activation functions:")

    # Get errors for each activation function
    sigmoid_errors = test_backpropagation_nonlinear('sigmoid')
    tanh_errors = test_backpropagation_nonlinear('tanh')

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sigmoid_errors)), sigmoid_errors, label='Sigmoid')
    plt.plot(range(len(tanh_errors)), tanh_errors, label='Tanh')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Comparison of Activation Functions')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale for better visualization
    plt.show()

# Run the examples
print("Testing Single Layer Perceptron on linearly separable data:")
test_perceptron_linearly_separable()

print("\nTesting Backpropagation Neural Network with sigmoid activation:")
test_backpropagation_nonlinear('sigmoid')

print("\nTesting Backpropagation Neural Network with tanh activation:")
test_backpropagation_nonlinear('tanh')

# Compare sigmoid and tanh
compare_activations()

