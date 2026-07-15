import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))

        # Initialize gradient buffers immediately
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.input_cache = None

    def zero_grad(self):
        self.dW.fill(0)
        self.db.fill(0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Y = XW + b
        x shape: (batch_size, in_features)
        """
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass with gradient accumulation.
        """
        # 1. Accumulate gradients with respect to weights
        # Instead of self.dW = ..., use self.dW += ...
        self.dW += np.dot(self.x.T, grad_output)

        # 2. Accumulate gradients with respect to bias
        self.db += np.sum(grad_output, axis=0, keepdims=True)

        # 3. Gradient with respect to input
        grad_input = np.dot(grad_output, self.W.T)

        return grad_input
