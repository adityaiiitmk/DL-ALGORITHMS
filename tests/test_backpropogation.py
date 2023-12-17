import pytest
from BackPropogation.BackPropogation import BackPropogation
import numpy as np

@pytest.fixture
def sample_backpropagation():
    return BackPropogation(learning_rate=0.01, epochs=100, activation_function='step')

def test_backpropagation_creation(sample_backpropagation):
    assert sample_backpropagation.learning_rate == 0.01
    assert sample_backpropagation.max_epochs == 100
    assert sample_backpropagation.activation_function == 'step'

def test_backpropagation_activation_step(sample_backpropagation):
    assert sample_backpropagation.activate(0) == 1
    assert sample_backpropagation.activate(1) == 1
    assert sample_backpropagation.activate(-1) == 0

def test_backpropagation_activation_sigmoid():
    sample_backpropagation=BackPropogation(activation_function='sigmoid')
    assert sample_backpropagation.activation_function == 'sigmoid'
    assert sample_backpropagation.activate(0) == 1
    assert sample_backpropagation.activate(1) == 1
    assert sample_backpropagation.activate(-1) == 0

def test_backpropagation_activation_relu(sample_backpropagation):
    sample_backpropagation=BackPropogation(activation_function='relu')
    assert sample_backpropagation.activation_function == 'relu'
    assert sample_backpropagation.activate(0) == 0
    assert sample_backpropagation.activate(1) == 1
    assert sample_backpropagation.activate(-1) == 0

def test_backpropagation_fit(sample_backpropagation):
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    sample_backpropagation.fit(X, y)
    assert sample_backpropagation.weights.shape == (2,)  
    assert sample_backpropagation.bias >= 0

