import numpy as np


class Embedding:
    def __init__(self, vocab_size: int, embed_dim: int):
        # Initialize weights with small random values to break symmetry
        self.weights = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(
            np.float32
        )
        # Gradient buffer for the optimizer
        self.dW = np.zeros_like(self.weights)
        # Cache for forward indices
        self.indices = None

    def forward(self, input_indices: np.ndarray) -> np.ndarray:
        # Ensure indices are within valid bounds [0, vocab_size - 1]
        # This prevents the IndexError
        clipped_indices = np.clip(input_indices, 0, self.weights.shape[0] - 1)

        self.indices = clipped_indices
        return self.weights[clipped_indices]

    def backward(self, grad_output: np.ndarray) -> None:
        """
        grad_output: Gradient from the LSTM (batch_size, embed_dim)

        Uses np.add.at to correctly accumulate gradients when the same
        word index appears multiple times in a batch.
        """
        # Add grad_output to the rows of dW specified by self.indices
        np.add.at(self.dW, self.indices, grad_output)

    def zero_grad(self) -> None:
        """Clears the gradient buffer."""
        self.dW.fill(0)
