"""Perceptron implementation for learning the AND logical gate.

This script demonstrates how a simple perceptron can learn the AND logic gate,
which is a linearly separable problem.
"""

import numpy as np

from utilities.nn import Perceptron


def create_and_dataset():
    """Create the AND logic gate dataset.
    
    Returns:
        X: Input features (2D binary combinations)
        y: Target labels (AND logic results)
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # Flattened for compatibility
    return X, y


def train_perceptron(X, y, n_features=2, alpha=0.1, epochs=20):
    """Train a perceptron on the given dataset.
    
    Args:
        X: Training features
        y: Training labels
        n_features: Number of input features
        alpha: Learning rate
        epochs: Number of training epochs
    
    Returns:
        Trained perceptron model
    """
    print('[INFO] Training perceptron...')
    perceptron = Perceptron(n_features=n_features, alpha=alpha, seed=42)
    perceptron.fit(X, y, epochs=epochs)
    print(f'[INFO] Training complete after {epochs} epochs\n')
    return perceptron


def test_perceptron(perceptron, X, y):
    """Test the trained perceptron and display results.
    
    Args:
        perceptron: Trained perceptron model
        X: Test features
        y: True labels
    """
    print('[INFO] Testing perceptron...')
    print('-' * 50)
    
    correct = 0
    for x, target in zip(X, y):
        prediction = perceptron.predict(x)
        is_correct = '✓' if prediction == target else '✗'
        correct += (prediction == target)
        
        print(f'Input: {x} | Target: {target} | Prediction: {prediction} {is_correct}')
    
    accuracy = (correct / len(y)) * 100
    print('-' * 50)
    print(f'[INFO] Accuracy: {accuracy:.1f}% ({correct}/{len(y)})\n')


def main():
    """Main function to train and test perceptron on AND gate."""
    # Create dataset
    X, y = create_and_dataset()
    
    # Train perceptron
    perceptron = train_perceptron(X, y, n_features=X.shape[1], alpha=0.1, epochs=20)
    
    # Test perceptron
    test_perceptron(perceptron, X, y)


if __name__ == '__main__':
    main()