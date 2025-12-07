"""
CIFAR-10 Image Classification with Keras Neural Network

This script demonstrates training a deep neural network using Keras on the CIFAR-10 dataset
for object classification across 10 categories.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

# CIFAR-10 class names
LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        dict: Parsed arguments
    """
    ap = argparse.ArgumentParser(description='Train a neural network on CIFAR-10 dataset')
    ap.add_argument('-o', '--output', required=True,
                    help='Path to the output loss/accuracy plot')
    ap.add_argument('-e', '--epochs', type=int, default=100,
                    help='Number of training epochs (default: 100)')
    ap.add_argument('-b', '--batch-size', type=int, default=32,
                    help='Batch size for training (default: 32)')
    ap.add_argument('-l', '--learning-rate', type=float, default=0.01,
                    help='Learning rate for SGD optimizer (default: 0.01)')
    return vars(ap.parse_args())


def load_and_preprocess_data():
    """
    Load CIFAR-10 dataset and apply normalization.
    
    CIFAR-10 contains 32x32 RGB images across 10 classes.
    Images are flattened from (32, 32, 3) to (3072,) for Dense layers.
    
    Returns:
        tuple: (train_x, train_y, test_x, test_y) normalized and flattened
    """
    print('[INFO] Loading CIFAR-10 dataset...')
    ((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
    
    # Normalize pixel intensities to [0, 1]
    train_x = train_x.astype('float32') / 255.0
    test_x = test_x.astype('float32') / 255.0
    
    # Flatten 3D images to 1D vectors (32x32x3 = 3072)
    train_x = train_x.reshape((train_x.shape[0], 3072))
    test_x = test_x.reshape((test_x.shape[0], 3072))
    
    print(f'[INFO] Training samples: {train_x.shape[0]} images, {train_x.shape[1]} features each')
    print(f'[INFO] Testing samples: {test_x.shape[0]} images')
    
    return train_x, train_y, test_x, test_y


def encode_labels(train_y, test_y):
    """
    One-hot encode the labels.
    
    Args:
        train_y: Training labels
        test_y: Testing labels
        
    Returns:
        tuple: (train_y_encoded, test_y_encoded, label_binarizer)
    """
    lb = LabelBinarizer()
    train_y = lb.fit_transform(train_y)
    test_y = lb.transform(test_y)
    
    return train_y, test_y, lb


def build_model(input_dim=3072, architecture=None):
    """
    Build a sequential neural network model.
    
    Args:
        input_dim: Number of input features (default: 3072 for 32x32x3 images)
        architecture: List of hidden layer sizes (default: [1024, 512])
        
    Returns:
        Sequential: Keras model
    """
    if architecture is None:
        architecture = [1024, 512]
    
    model = Sequential()
    
    # First hidden layer with input shape
    model.add(Dense(architecture[0], input_shape=(input_dim,), activation='relu'))
    
    # Additional hidden layers
    for units in architecture[1:]:
        model.add(Dense(units, activation='relu'))
    
    # Output layer (10 classes)
    model.add(Dense(10, activation='softmax'))
    
    print(f'[INFO] Model architecture: {input_dim} -> {" -> ".join(map(str, architecture))} -> 10')
    
    return model


def train_model(model, train_x, train_y, test_x, test_y,
                learning_rate=0.01, epochs=100, batch_size=32):
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
        batch_size: Batch size for training (default: 32)
        
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


def evaluate_model(model, test_x, test_y, label_names=None):
    """
    Evaluate model performance and display classification report.
    
    Args:
        model: Trained Keras model
        test_x: Testing features
        test_y: Testing labels (one-hot encoded)
        label_names: List of class names for display
    """
    if label_names is None:
        label_names = LABEL_NAMES
    
    print('\n[INFO] Evaluating network...')
    
    # Make predictions
    predictions = model.predict(test_x, batch_size=32, verbose=0)
    
    # Convert one-hot to class indices
    true_labels = test_y.argmax(axis=1)
    pred_labels = predictions.argmax(axis=1)
    
    # Display classification report
    print('\n' + '='*70)
    print('CLASSIFICATION REPORT')
    print('='*70)
    print(classification_report(true_labels, pred_labels, target_names=label_names))


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
    train_x, train_y, test_x, test_y = load_and_preprocess_data()
    
    # Encode labels
    train_y, test_y, lb = encode_labels(train_y, test_y)
    
    # Build model
    model = build_model(input_dim=3072, architecture=[1024, 512])
    
    # Train model
    history = train_model(
        model, train_x, train_y, test_x, test_y,
        learning_rate=args['learning_rate'],
        epochs=args['epochs'],
        batch_size=args['batch_size']
    )
    
    # Evaluate model
    evaluate_model(model, test_x, test_y, LABEL_NAMES)
    
    # Plot training history
    plot_training_history(history, args['output'], args['epochs'])


if __name__ == '__main__':
    main()