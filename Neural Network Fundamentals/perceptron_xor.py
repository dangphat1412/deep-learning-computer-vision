"""Perceptron implementation for learning the XOR logical gate.

This script demonstrates the limitations of a simple perceptron on the XOR problem,
which is NOT linearly separable. A single-layer perceptron cannot learn XOR logic
and will fail to achieve 100% accuracy, illustrating why multi-layer networks are needed.
"""

import numpy as np

from utilities.nn import Perceptron


def create_xor_dataset():
    """Create the XOR logic gate dataset.
    
    Returns:
        X: Input features (2D binary combinations)
        y: Target labels (XOR logic results)
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # Flattened for compatibility
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
    print('[WARNING] XOR is NOT linearly separable - perceptron will fail!')
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
    print(f'[INFO] Accuracy: {accuracy:.1f}% ({correct}/{len(y)})')
    
    if accuracy < 100:
        print('[NOTE] Perceptron failed to learn XOR (as expected).')
        print('[NOTE] XOR requires a multi-layer neural network to solve.\n')
    else:
        print('[UNEXPECTED] Perceptron learned XOR by chance!\n')


def main():
    """Main function to train and test perceptron on XOR gate."""
    # Create dataset
    X, y = create_xor_dataset()
    
    # Train perceptron (will fail to learn XOR)
    perceptron = train_perceptron(X, y, n_features=X.shape[1], alpha=0.1, epochs=20)
    
    # Test perceptron (will show poor accuracy)
    test_perceptron(perceptron, X, y)


if __name__ == '__main__':
    main()