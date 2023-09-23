from Perceptron import  Perceptron
import numpy as np



if __name__ == "__main__":
    
    # OR gate dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  # OR gate output

    perceptron = Perceptron(epochs=1)
    perceptron.fit(X, y)

    # Make predictions
    test_data = np.array([[0, 1], [1, 0], [0, 0], [1, 0]])
    predictions = perceptron.predict(test_data)

    print("Predictions :", predictions)
    



