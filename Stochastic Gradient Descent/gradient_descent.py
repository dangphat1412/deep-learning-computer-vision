import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class LogisticRegressionGD:
    """Logistic Regression with Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, epochs=100):
        """
        Initialize the model
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            epochs (int): Number of training epochs
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.losses = []
    
    @staticmethod
    def sigmoid(x):
        """
        Compute sigmoid activation
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Sigmoid activated output
        """
        return 1.0 / (1 + np.exp(-x))
    
    def _add_bias(self, X):
        """
        Add bias column to feature matrix
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Feature matrix with bias column
        """
        return np.c_[X, np.ones(X.shape[0])]
    
    def fit(self, X, y, verbose=True):
        """
        Train the model using gradient descent
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            verbose (bool): Print training progress
        """
        # Add bias term
        X = self._add_bias(X)
        
        # Initialize weights
        self.weights = np.random.randn(X.shape[1], 1)
        self.losses = []
        
        if verbose:
            print('[INFO] Training started...')
        
        # Training loop
        for epoch in range(self.epochs):
            # Forward pass
            predictions = self.sigmoid(X.dot(self.weights))
            
            # Compute loss
            error = predictions - y
            loss = np.sum(error ** 2)
            self.losses.append(loss)
            
            # Backward pass - compute gradient
            gradient = X.T.dot(error)
            
            # Update weights
            self.weights -= self.learning_rate * gradient
            
            # Print progress
            if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
                print(f'[INFO] Epoch: {epoch + 1}/{self.epochs}, Loss: {loss:.4f}')
        
        if verbose:
            print('[INFO] Training completed!')
    
    def predict_proba(self, X):
        """
        Predict probability scores
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Probability predictions
        """
        X = self._add_bias(X)
        return self.sigmoid(X.dot(self.weights))
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels
        
        Args:
            X (np.ndarray): Feature matrix
            threshold (float): Decision threshold
            
        Returns:
            np.ndarray: Binary predictions
        """
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).astype(int)
    
    def get_loss_history(self):
        """Get training loss history"""
        return self.losses


def generate_data(n_samples=1000, n_features=2, centers=2, 
                  cluster_std=1.5, test_size=0.5, random_state=42):
    """
    Generate synthetic classification dataset
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        centers (int): Number of class centers
        cluster_std (float): Standard deviation of clusters
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    y = y.reshape(-1, 1)
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def plot_results(X_test, y_test, losses, epochs):
    """
    Plot classification data and training loss
    
    Args:
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        losses (list): Training loss history
        epochs (int): Number of epochs
    """
    plt.style.use('ggplot')
    
    # Plot 1: Classification data
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Test Data Distribution')
    plt.scatter(X_test[:, 0], X_test[:, 1], 
                c=y_test.ravel(), cmap='viridis', 
                marker='o', s=30, alpha=0.6, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    
    # Plot 2: Training loss
    plt.subplot(1, 2, 2)
    plt.title('Training Loss Over Time')
    plt.plot(range(1, epochs + 1), losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('-a', '--alpha', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples to generate')
    args = parser.parse_args()
    
    # Generate dataset
    print('[INFO] Generating dataset...')
    X_train, X_test, y_train, y_test = generate_data(n_samples=args.samples)
    print(f'[INFO] Training samples: {len(X_train)}, Test samples: {len(X_test)}')
    
    # Train model
    model = LogisticRegressionGD(learning_rate=args.alpha, epochs=args.epochs)
    model.fit(X_train, y_train, verbose=True)
    
    # Evaluate model
    print('\n[INFO] Evaluating model...')
    y_pred = model.predict(X_test)
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    
    # Plot results
    plot_results(X_test, y_test, model.get_loss_history(), args.epochs)


if __name__ == '__main__':
    main()