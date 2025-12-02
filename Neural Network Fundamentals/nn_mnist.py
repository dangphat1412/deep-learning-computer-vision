"""
MNIST Digit Classification with Neural Network

This script demonstrates training a neural network on the MNIST dataset
for handwritten digit classification (0-9).
"""

import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from utilities.nn import NeuralNetwork


def load_and_preprocess_data():
    """
    Load MNIST dataset and apply min-max normalization.
    
    The digits dataset contains 8x8 pixel images (64-dimensional feature vectors)
    representing handwritten digits 0-9.
    
    Returns:
        Tuple of (data, labels) where data is normalized to [0, 1]
    """
    digits = datasets.load_digits()
    data = digits.data.astype('float')
    
    # Min-max normalization to scale pixel intensities to [0, 1]
    data = (data - data.min()) / (data.max() - data.min())
    
    print(f'[INFO] Loaded dataset: {data.shape[0]} samples, {data.shape[1]} features')
    
    return data, digits.target


def prepare_data_splits(data, labels, test_size=0.25, random_state=None):
    """
    Split data into training and testing sets and encode labels.
    
    Args:
        data: Feature matrix
        labels: Target labels
        test_size: Proportion of data to use for testing (default: 0.25)
        random_state: Random seed for reproducibility (default: None)
        
    Returns:
        Tuple of (train_x, test_x, train_y, test_y) with one-hot encoded labels
    """
    # Split data
    train_x, test_x, train_y, test_y = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    
    # One-hot encode labels
    lb = LabelBinarizer()
    train_y = lb.fit_transform(train_y)
    test_y = lb.transform(test_y)
    
    print(f'[INFO] Training samples: {train_x.shape[0]}')
    print(f'[INFO] Testing samples: {test_x.shape[0]}')
    
    return train_x, test_x, train_y, test_y


def build_and_train_network(train_x, train_y, architecture=None, 
                            learning_rate=0.1, epochs=1000, display_update=100):
    """
    Build and train a neural network.
    
    Args:
        train_x: Training features
        train_y: Training labels (one-hot encoded)
        architecture: List defining network layer sizes (default: [64, 32, 16, 10])
        learning_rate: Learning rate for gradient descent (default: 0.1)
        epochs: Number of training iterations (default: 1000)
        display_update: Frequency of displaying loss updates (default: 100)
        
    Returns:
        Trained NeuralNetwork instance
    """
    if architecture is None:
        architecture = [train_x.shape[1], 32, 16, 10]
    
    print(f'\n[INFO] Training network...')
    nn = NeuralNetwork(architecture, alpha=learning_rate)
    print(f'[INFO] Architecture: {nn}')
    
    nn.fit(train_x, train_y, epochs=epochs, display_update=display_update)
    
    return nn


def evaluate_network(nn, test_x, test_y):
    """
    Evaluate trained network and display classification report.
    
    Args:
        nn: Trained NeuralNetwork instance
        test_x: Test features
        test_y: Test labels (one-hot encoded)
    """
    print(f'\n[INFO] Evaluating network...')
    
    # Make predictions
    predictions = nn.predict(test_x)
    predictions = predictions.argmax(axis=1)
    
    # Convert one-hot encoded labels back to class indices
    true_labels = test_y.argmax(axis=1)
    
    # Display classification report
    print('\n' + '='*60)
    print('CLASSIFICATION REPORT')
    print('='*60)
    print(classification_report(true_labels, predictions))


def main():
    """Main execution function."""
    # Load and preprocess data
    data, labels = load_and_preprocess_data()
    
    # Prepare train/test splits
    train_x, test_x, train_y, test_y = prepare_data_splits(
        data, labels, test_size=0.25, random_state=42
    )
    
    # Build and train network
    nn = build_and_train_network(
        train_x, train_y, 
        architecture=[train_x.shape[1], 32, 16, 10],
        learning_rate=0.1,
        epochs=1000,
        display_update=100
    )
    
    # Evaluate network
    evaluate_network(nn, test_x, test_y)


if __name__ == '__main__':
    main()
