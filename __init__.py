"""Machine learning implementations and utilities."""

from .linear_regression import Ridge, Lasso
from .logistic_regression import CustomLogisticRegression
from .svm import SVC
from .hebbian_network import HebbianNeuron
from .mcculloch_pitts import McCullochPittsNeuron
from .neural_network import BackpropagationNeuralNetwork
from .experiments import ExperimentWriter

__version__ = "0.1.2"

# Create a default writer instance for easy access
writer = ExperimentWriter()

__all__ = [
    'Ridge',
    'Lasso',
    'CustomLogisticRegression',
    'SVC', 
    'HebbianNeuron',
    'McCullochPittsNeuron',
    'BackpropagationNeuralNetwork',
    'ExperimentWriter',
    'writer'
]