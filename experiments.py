"""
Provides pre-built experiments and code examples for machine learning models.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
try:
    from . import hebbian_network
    from . import neural_network
    from . import mcculloch_pitts
    from . import logistic_regression
    from . import linear_regression
    from . import pca
    from . import svm
    HebbianNeuron = hebbian_network.HebbianNeuron
    BackpropagationNeuralNetwork = neural_network.BackpropagationNeuralNetwork
    McCullochPittsNeuron = mcculloch_pitts.McCullochPittsNeuron
    CustomLogisticRegression = logistic_regression.CustomLogisticRegression
    Ridge = linear_regression.Ridge
    Lasso = linear_regression.Lasso
    PCA = pca.PCA
    SVC = svm.SVC
except ImportError:
    # Direct imports for Colab compatibility
    from python_expokar.ml.hebbian_network import HebbianNeuron
    from python_expokar.ml.neural_network import BackpropagationNeuralNetwork
    from python_expokar.ml.mcculloch_pitts import McCullochPittsNeuron
    from python_expokar.ml.logistic_regression import CustomLogisticRegression
    from python_expokar.ml.linear_regression import Ridge, Lasso
    from python_expokar.ml.pca import PCA
    from python_expokar.ml.svm import SVC

class ExperimentWriter:
    """Provides access to pre-built experiments and example code."""
    
    def __init__(self, output_file=None):
        """Initialize the experiment writer."""
        self.output_file = output_file
        self.code_snippets = {
            'hebbian': self._get_hebbian_code(),
            'logistic': self._get_logistic_code(),
            'neural': self._get_neural_code(),
            'mcculloch': self._get_mcculloch_code(),
            'pca': self._get_pca_code(),
            'svm': self._get_svm_code(),
            'ridge': self._get_ridge_code(),
            'lasso': self._get_lasso_code()
        }
        self._experiments = {
            'hebbian_and': self._hebbian_and_gate,
            'decision_boundary': self._decision_region_perceptron,
            'neural_xor': self._neural_network_xor,
            'logistic_binary': self._logistic_regression,
            'svm_kernels': self._svm_kernels,
            'mcculloch_gates': self._mcculloch_pitts_gates,
            'pca_reduce': self._pca_dimension_reduction,
            'ridge_reg': self._ridge_regression,
            'lasso_reg': self._lasso_regression
        }

    @property
    def questions(self):
        """List all available code snippets."""
        return list(self.code_snippets.keys())

    @property
    def experiments(self):
        """List all available experiments."""
        return list(self._experiments.keys())

    def _get_decision_region_code(self):
        return '''
# Plot decision regions for any classifier
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.cm.RdYlBu

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0], X[y == cl, 1], 
                   alpha=0.8, c=[colors[idx]], marker=markers[idx], label=cl)'''

    def _get_hebbian_code(self):
        return '''
# Hebbian Learning Example
from python_expokar.ml import HebbianNeuron

# Create training data for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Create and train Hebbian neuron
neuron = HebbianNeuron(input_size=2, learning_rate=0.1)
neuron.train_hebbian(X, y, epochs=100)

# Make predictions
predictions = neuron.predict(X)
print("AND Gate Results:", predictions)'''

    def _get_logistic_code(self):
        return '''
# Logistic Regression Example
from python_expokar.ml.logistic_regression import CustomLogisticRegression
import numpy as np
from sklearn.datasets import make_classification

# Create sample data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)

# Create and train logistic regression model
model = CustomLogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print("Predictions:", predictions)'''

    def _get_neural_code(self):
        return '''
# Neural Network Example
from python_expokar.ml import BackpropagationNeuralNetwork

# Create XOR gate data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create and train neural network
nn = BackpropagationNeuralNetwork([2, 4, 1])
nn.train(X, y.reshape(-1, 1), epochs=1000)

# Make predictions
predictions = nn.predict(X)
print("XOR Gate Results:", predictions.flatten())'''

    def _get_mcculloch_code(self):
        return '''
# McCulloch-Pitts Neuron Example
from python_expokar.ml import McCullochPittsNeuron

# Create OR gate
neuron = McCullochPittsNeuron(weights=[1, 1], threshold=1)

# Test the neuron
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
results = [neuron.activate(x) for x in X]
print("OR Gate Results:", results)'''

    def _get_pca_code(self):
        return '''
# PCA Example
from python_expokar.ml import PCA

# Generate sample data
X = np.random.randn(100, 5)

# Create and fit PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print("Reduced shape:", X_reduced.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)'''

    def _get_svm_code(self):
        return '''
# SVM Example
from python_expokar.ml import SVC

# Create sample data
X, y = make_moons(n_samples=100, noise=0.15)

# Train SVM with different kernels
kernels = ['linear', 'poly', 'rbf']
models = {}
for kernel in kernels:
    svm = SVC(kernel=kernel)
    svm.fit(X, y)
    models[kernel] = svm.score(X, y)
print("Kernel accuracies:", models)'''

    def _get_ridge_code(self):
        return '''
# Ridge Regression Example
from python_expokar.ml.linear_regression import Ridge
import numpy as np

# Create sample data
X = np.random.randn(100, 5)
y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

# Train Ridge regression
ridge = Ridge(alpha=0.5)
ridge.fit(X, y)
print("Ridge coefficients:", ridge.weights)'''

    def _get_lasso_code(self):
        return '''
# Lasso Regression Example
from python_expokar.ml.linear_regression import Lasso
import numpy as np

# Create sample data
X = np.random.randn(100, 5)
y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

# Train Lasso regression
lasso = Lasso(alpha=0.5)
lasso.fit(X, y)
print("Lasso coefficients:", lasso.weights)'''

    def getCode(self, question):
        """Get code snippet by name."""
        if question in self.code_snippets:
            code = self.code_snippets[question]
            if self.output_file:
                with open(self.output_file, 'w') as f:
                    f.write(code)
            return code
        else:
            return f"No code found for '{question}'. Available: {self.questions}"

    def get_experiment(self, name):
        """Get a specific experiment by name."""
        if name not in self._experiments:
            raise ValueError(f"Unknown experiment: {name}. Available: {self.experiments}")
        return self._experiments[name]

    # Experiment implementations
    def _hebbian_and_gate(self):
        """AND gate using Hebbian learning."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        neuron = HebbianNeuron(input_size=2, learning_rate=0.1)
        neuron.train_hebbian(X, y, epochs=50)
        return neuron

    def _neural_network_xor(self):
        """XOR gate using neural network."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        nn = BackpropagationNeuralNetwork([2, 4, 1])
        nn.train(X, y.reshape(-1, 1), epochs=1000)
        return nn

    def _decision_region_perceptron(self):
        """Decision boundary visualization."""
        X, y = make_classification(n_samples=100, n_features=2, 
                                 n_redundant=0, n_informative=2)
        neuron = HebbianNeuron(input_size=2, learning_rate=0.1)
        neuron.train_perceptron(X, y, epochs=50)
        return neuron

    def _logistic_regression(self):
        """Binary classification with logistic regression."""
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)
        model = CustomLogisticRegression(learning_rate=0.01, n_iterations=1000)
        model.fit(X, y)
        return model

    def _svm_kernels(self):
        """SVM with different kernels."""
        X, y = make_moons(n_samples=100, noise=0.15)
        models = []
        for kernel in ['linear', 'poly', 'rbf']:
            svm = SVC(kernel=kernel)
            svm.fit(X, y)
            models.append(svm)
        return models

    def _mcculloch_pitts_gates(self):
        """Basic logic gates using McCulloch-Pitts neurons."""
        gates = {
            'AND': McCullochPittsNeuron(weights=[1, 1], threshold=2),
            'OR': McCullochPittsNeuron(weights=[1, 1], threshold=1),
            'NAND': McCullochPittsNeuron(weights=[-1, -1], threshold=-1),
            'NOR': McCullochPittsNeuron(weights=[-1, -1], threshold=0)
        }
        return gates

    def _pca_dimension_reduction(self):
        """PCA dimension reduction example."""
        X = np.random.randn(100, 5)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        return pca

    def _ridge_regression(self):
        """Ridge regression example."""
        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1
        ridge = Ridge(alpha=0.5)
        ridge.fit(X, y)
        return ridge

    def _lasso_regression(self):
        """Lasso regression example."""
        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1
        lasso = Lasso(alpha=0.5)
        lasso.fit(X, y)
        return lasso