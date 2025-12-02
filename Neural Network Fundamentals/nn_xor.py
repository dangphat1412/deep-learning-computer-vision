"""
XOR Neural Network Training Example

This script demonstrates training a simple neural network to learn the XOR function,
which is not linearly separable and requires a hidden layer to solve.
"""

import numpy as np
from utilities.nn import NeuralNetwork


def create_xor_dataset():
    """
    Create the XOR dataset.
    
    Returns:
        Tuple of (X, y) where X is input features and y is target labels
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return X, y


def train_network(X, y, architecture=[2, 2, 1], learning_rate=0.5, epochs=20000):
    """
    Train a neural network on the given data.
    
    Args:
        X: Training features
        y: Training labels
        architecture: List defining network layer sizes
        learning_rate: Learning rate for gradient descent
        epochs: Number of training iterations
        
    Returns:
        Trained NeuralNetwork instance
    """
    print('[INFO] Training network...')
    nn = NeuralNetwork(architecture, alpha=learning_rate)
    nn.fit(X, y, epochs=epochs)
    return nn


def test_network(nn, X, y, threshold=0.5):
    """
    Test the neural network and display predictions.
    
    Args:
        nn: Trained NeuralNetwork instance
        X: Test features
        y: True labels
        threshold: Classification threshold (default: 0.5)
    """
    print('\n[INFO] Testing network...')
    
    for (x, target) in zip(X, y):
        # Make prediction
        pred = nn.predict(x)[0][0]
        step = 1 if pred > threshold else 0
        
        # Display result
        print(f'[INFO] Data={x}, Ground Truth={target[0]}, '
              f'Prediction={pred:.4f}, Step={step}')


def main():
    """Main execution function."""
    # Create dataset
    X, y = create_xor_dataset()
    
    # Train network
    nn = train_network(X, y, architecture=[2, 2, 1], learning_rate=0.5, epochs=20000)
    
    # Test network
    test_network(nn, X, y)


if __name__ == '__main__':
    main()
