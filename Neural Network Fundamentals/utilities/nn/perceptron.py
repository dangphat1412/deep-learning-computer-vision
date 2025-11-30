"""Perceptron implementation for binary classification."""

from typing import Union

import numpy as np


class Perceptron:
    """A simple perceptron classifier for binary classification.
    
    The perceptron is a linear binary classifier that learns a decision
    boundary by updating weights based on misclassified examples.
    
    Attributes:
        W: Weight vector including bias term (shape: n_features + 1)
        alpha: Learning rate for weight updates
    """

    def __init__(self, n_features: int, alpha: float = 0.1, seed: int = None) -> None:
        """Initialize the perceptron with random weights.
        
        Args:
            n_features: Number of input features (excluding bias).
            alpha: Learning rate for gradient descent updates.
                Typical values range from 0.001 to 0.1.
            seed: Random seed for reproducibility. If None, results will vary.
        """
        # Initialize weights using Xavier/He initialization
        # +1 for bias term
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal(n_features + 1) / np.sqrt(n_features)
        self.alpha = alpha

    def step(self, x: Union[float, np.ndarray]) -> int:
        """Apply step activation function.
        
        Args:
            x: Input value or array.
        
        Returns:
            1 if x > 0, else 0.
        """
        return 1 if x > 0 else 0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> None:
        """Train the perceptron on the provided dataset.
        
        Updates weights using the perceptron learning rule:
        w = w - alpha * error * x (only when prediction != target)
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,). Should be binary (0 or 1).
            epochs: Number of complete passes through the training dataset.
        
        Raises:
            ValueError: If X and y have incompatible shapes.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )
        
        # Add bias column (column of 1's) to feature matrix
        # This treats bias as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones(X.shape[0])]

        # Training loop over epochs
        for _ in range(epochs):
            # Iterate over each training example
            for x, target in zip(X, y):
                # Forward pass: compute prediction
                prediction = self.step(np.dot(x, self.W))

                # Update weights only if prediction is incorrect
                if prediction != target:
                    # Compute error
                    error = prediction - target

                    # Perceptron weight update rule
                    self.W += -self.alpha * error * x

    def predict(self, X: np.ndarray, add_bias: bool = True) -> np.ndarray:
        """Predict class labels for samples in X.
        
        Args:
            X: Input data of shape (n_samples, n_features) or (n_features,).
            add_bias: Whether to add bias column. Set to False if X already
                includes bias term.
        
        Returns:
            Predicted class labels (0 or 1) of shape (n_samples,) or scalar.
        
        Raises:
            ValueError: If X has incorrect number of features.
        """
        # Ensure input is at least 2D
        X = np.atleast_2d(X)
        
        # Validate feature dimensions
        expected_features = len(self.W) - 1 if add_bias else len(self.W)
        if X.shape[1] != expected_features:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {expected_features}"
            )

        # Add bias column if needed
        if add_bias:
            X = np.c_[X, np.ones(X.shape[0])]

        # Compute predictions for all samples
        predictions = np.array([self.step(np.dot(x, self.W)) for x in X])
        
        # Return scalar if single sample, otherwise return array
        return predictions[0] if len(predictions) == 1 else predictions