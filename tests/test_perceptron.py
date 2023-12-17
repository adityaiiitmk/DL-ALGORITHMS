from Perceptron.Perceptron import Perceptron
import numpy as np

    
def test_perceptron_creation():
    perceptron = Perceptron()
    assert perceptron.learning_rate == 0.01
    assert perceptron.max_epochs == 100
    assert perceptron.activation_function == 'step'

def test_perceptron_activation_step():
    perceptron = Perceptron(activation_function='step')
    assert perceptron.activate(0) == 1
    assert perceptron.activate(1) == 1
    assert perceptron.activate(-1) == 0


def test_perceptron_activation_relu():
    perceptron = Perceptron(activation_function='relu')
    assert perceptron.activate(0) == 0
    assert perceptron.activate(1) == 1
    assert perceptron.activate(-1) == 0

def test_perceptron_fit():
    perceptron = Perceptron()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    perceptron.fit(X, y)
    assert perceptron.weights.shape == (2,)  
    assert perceptron.bias == 0


