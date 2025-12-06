"""
MNIST Digit Classification with Keras Neural Network

This script demonstrates training a deep neural network using Keras on the full MNIST dataset
for handwritten digit classification (0-9).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        dict: Parsed arguments
    """
    ap = argparse.ArgumentParser(description='Train a neural network on MNIST dataset')
    ap.add_argument('-o', '--output', required=True,
                    help='Path to the output loss/accuracy plot')
    ap.add_argument('-e', '--epochs', type=int, default=100,
                    help='Number of training epochs (default: 100)')
    ap.add_argument('-b', '--batch-size', type=int, default=128,
                    help='Batch size for training (default: 128)')
    ap.add_argument('-l', '--learning-rate', type=float, default=0.01,
                    help='Learning rate for SGD optimizer (default: 0.01)')
    return vars(ap.parse_args())


def load_and_preprocess_data():
    """
    Load MNIST dataset and apply normalization.
    
    Returns:
        tuple: (data, labels) where data is normalized to [0, 1]
    """
    print('[INFO] Loading MNIST (full) dataset...')
    dataset = datasets.fetch_mldata('MNIST Original')
    
    # Normalize pixel intensities to [0, 1]
    data = dataset.data.astype('float32') / 255.0
    
    print(f'[INFO] Loaded {data.shape[0]} samples with {data.shape[1]} features each')
    
    return data, dataset.target


def prepare_data_splits(data, labels, test_size=0.25, random_state=42):
    """
    Split data into training and testing sets with one-hot encoded labels.
    
    Args:
        data: Feature matrix
        labels: Target labels
        test_size: Proportion of data for testing (default: 0.25)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (train_x, test_x, train_y, test_y, label_binarizer)
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
    
    return train_x, test_x, train_y, test_y, lb


def build_model(input_dim=784, architecture=None):
    """
    Build a sequential neural network model.
    
    Args:
        input_dim: Number of input features (default: 784 for 28x28 images)
        architecture: List of hidden layer sizes (default: [256, 128])
        
    Returns:
        Sequential: Compiled Keras model
    """
    if architecture is None:
        architecture = [256, 128]
    
    model = Sequential()
    
    # First hidden layer
    model.add(Dense(architecture[0], input_shape=(input_dim,), activation='sigmoid'))
    
    # Additional hidden layers
    for units in architecture[1:]:
        model.add(Dense(units, activation='sigmoid'))
    
    # Output layer (10 classes for digits 0-9)
    model.add(Dense(10, activation='softmax'))
    
    print(f'[INFO] Model architecture: {input_dim} -> {"->".join(map(str, architecture))} -> 10')
    
    return model


def train_model(model, train_x, train_y, test_x, test_y, 
                learning_rate=0.01, epochs=100, batch_size=128):
    """
    Compile and train the neural network.
    
    Args:
        model: Keras Sequential model
        train_x: Training features
        train_y: Training labels (one-hot encoded)
        test_x: Testing features
        test_y: Testing labels (one-hot encoded)
        learning_rate: Learning rate for SGD (default: 0.01)
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size for training (default: 128)
        
    Returns:
        History: Training history object
    """
    print('[INFO] Training network...')
    
    # Compile model
    sgd = SGD(learning_rate=learning_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_x, train_y,
        validation_data=(test_x, test_y),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_x, test_y, label_binarizer):
    """
    Evaluate model performance and display classification report.
    
    Args:
        model: Trained Keras model
        test_x: Testing features
        test_y: Testing labels (one-hot encoded)
        label_binarizer: LabelBinarizer instance used for encoding
    """
    print('\n[INFO] Evaluating network...')
    
    # Make predictions
    predictions = model.predict(test_x, batch_size=128, verbose=0)
    
    # Convert one-hot to class indices
    true_labels = test_y.argmax(axis=1)
    pred_labels = predictions.argmax(axis=1)
    
    # Display classification report
    print('\n' + '='*70)
    print('CLASSIFICATION REPORT')
    print('='*70)
    print(classification_report(
        true_labels, pred_labels,
        target_names=[str(x) for x in label_binarizer.classes_]
    ))


def plot_training_history(history, output_path, epochs):
    """
    Plot and save training history (loss and accuracy).
    
    Args:
        history: Keras History object from model.fit()
        output_path: Path to save the plot
        epochs: Number of epochs trained
    """
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, epochs), history.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epochs), history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, epochs), history.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, epochs), history.history['val_accuracy'], label='val_acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f'\n[INFO] Training plot saved to: {output_path}')


def main():
    """
    Main execution function.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load and preprocess data
    data, labels = load_and_preprocess_data()
    
    # Prepare data splits
    train_x, test_x, train_y, test_y, lb = prepare_data_splits(data, labels)
    
    # Build model
    model = build_model(input_dim=train_x.shape[1], architecture=[256, 128])
    
    # Train model
    history = train_model(
        model, train_x, train_y, test_x, test_y,
        learning_rate=args['learning_rate'],
        epochs=args['epochs'],
        batch_size=args['batch_size']
    )
    
    # Evaluate model
    evaluate_model(model, test_x, test_y, lb)
    
    # Plot training history
    plot_training_history(history, args['output'], args['epochs'])


if __name__ == '__main__':
    main()