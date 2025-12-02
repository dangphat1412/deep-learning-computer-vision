import numpy as np


class NeuralNetwork:
    """
    A feedforward neural network with backpropagation.
    
    Attributes:
        layers: List of integers representing the number of nodes in each layer
        alpha: Learning rate for gradient descent
        W: List of weight matrices for each layer connection
    """
    
    def __init__(self, layers, alpha=0.1):
        """
        Initialize the neural network.
        
        Args:
            layers: List of integers defining network architecture (e.g., [2, 2, 1])
            alpha: Learning rate (default: 0.1)
        """
        self.W = []
        self.layers = layers
        self.alpha = alpha
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weight matrices using Xavier/He initialization."""
        # Initialize weights for all layers
        for i in range(len(self.layers) - 1):
            # Xavier initialization: weights scaled by sqrt(fan_in)
            # Add 1 to account for bias term in input
            fan_in = self.layers[i] + 1
            w = np.random.randn(self.layers[i] + 1, self.layers[i + 1])
            self.W.append(w / np.sqrt(fan_in))

    def __repr__(self):
        """Return string representation of the network architecture."""
        return 'NeuralNetwork: {}'.format('-'.join(str(l) for l in self.layers))
    
    @staticmethod
    def sigmoid(x):
        """
        Compute sigmoid activation function with numerical stability.
        
        Args:
            x: Input value or array
            
        Returns:
            Sigmoid activation: 1 / (1 + exp(-x))
        """
        # Clip values to prevent overflow in exp()
        # For x < -500, sigmoid is effectively 0
        # For x > 500, sigmoid is effectively 1
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_deriv(x):
        """
        Compute derivative of sigmoid function.
        
        Args:
            x: Input value that has already been passed through sigmoid
            
        Returns:
            Derivative: x * (1 - x)
        """
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update=100):
        """
        Train the neural network using backpropagation.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)
            epochs: Number of training iterations (default: 1000)
            display_update: Frequency of displaying loss updates (default: 100)
        """
        # Add bias column to feature matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        
        # Training loop
        for epoch in range(epochs):
            # Train on each data point (stochastic gradient descent)
            for (x, target) in zip(X, y):
                self._fit_partial(x, target)
            
            # Display training progress
            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print(f'[INFO] epoch={epoch + 1}, loss={loss:.7f}')

    def _fit_partial(self, x, y):
        """
        Perform a single training step using backpropagation.
        
        Args:
            x: Single training example with bias
            y: Target value for this example
        """
        # Forward pass
        activations = self._forward_pass(x)
        
        # Backward pass
        deltas = self._backward_pass(activations, y)
        
        # Update weights
        self._update_weights(activations, deltas)
    
    def _forward_pass(self, x):
        """
        Perform forward propagation through the network.
        
        Args:
            x: Input features (already includes bias)
            
        Returns:
            List of activations for each layer (with bias added to hidden layers)
        """
        activations = [np.atleast_2d(x)]
        
        for layer in range(len(self.W)):
            # Compute net input (weighted sum)
            net_input = activations[layer].dot(self.W[layer])
            
            # Apply activation function
            net_output = self.sigmoid(net_input)
            
            # Add bias column to hidden layer outputs (but not final output)
            if layer < len(self.W) - 1:
                net_output = np.c_[net_output, np.ones((net_output.shape[0], 1))]
            
            activations.append(net_output)
        
        return activations
    
    def _backward_pass(self, activations, target):
        """
        Perform backward propagation to compute gradients.
        
        Args:
            activations: List of activations from forward pass (with bias in hidden layers)
            target: True target value
            
        Returns:
            List of deltas (gradients) for each layer
        """
        # Compute output layer error
        error = activations[-1] - target
        
        # Initialize deltas with output layer gradient
        deltas = [error * self.sigmoid_deriv(activations[-1])]
        
        # Backpropagate error through hidden layers
        for layer in range(len(self.W) - 1, 0, -1):
            # Propagate delta backward through weights
            delta = deltas[-1].dot(self.W[layer].T)
            
            # CRITICAL FIX: Remove bias term from delta before applying derivative
            # The last column corresponds to bias weights, which shouldn't affect previous layer
            delta = delta[:, :-1]
            
            # Get activation without bias for this layer
            act = activations[layer][:, :-1]  # Remove bias column
            
            # Apply sigmoid derivative
            delta = delta * self.sigmoid_deriv(act)
            deltas.append(delta)
        
        # Reverse to match layer order
        return deltas[::-1]
    
    def _update_weights(self, activations, deltas):
        """
        Update weight matrices using computed gradients.
        
        Args:
            activations: List of activations from forward pass (with bias)
            deltas: List of gradients from backward pass (without bias for hidden layers)
        """
        for layer in range(len(self.W)):
            # Gradient descent weight update
            # activations[layer] shape: (1, n_in + 1) with bias
            # deltas[layer] shape: (1, n_out) without bias
            # W[layer] shape: (n_in + 1, n_out)
            # Result: (n_in + 1, 1) @ (1, n_out) = (n_in + 1, n_out) ✓
            self.W[layer] += -self.alpha * activations[layer].T.dot(deltas[layer])

    def predict(self, X, add_bias=True):
        """
        Make predictions for input data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            add_bias: Whether to add bias column (default: True)
            
        Returns:
            Predictions of shape (n_samples, n_outputs)
        """
        p = np.atleast_2d(X)
        
        # Add bias column if needed
        if add_bias:
            p = np.c_[p, np.ones((p.shape[0], 1))]
        
        # Forward propagate through all layers
        for layer in range(len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
            
            # Add bias to hidden layer outputs (but not final output)
            if layer < len(self.W) - 1:
                p = np.c_[p, np.ones((p.shape[0], 1))]
        
        return p
    
    def calculate_loss(self, X, targets):
        """
        Calculate binary cross-entropy loss (better for sigmoid outputs).
        
        Args:
            X: Input data (with bias column already added)
            targets: True target values
            
        Returns:
            Binary cross-entropy loss
        """
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        # Binary cross-entropy: -sum(y*log(p) + (1-y)*log(1-p))
        loss = -np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        
        return loss