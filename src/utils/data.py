import numpy as np


def generate_toy_data(num_samples=100, seq_len=5, vocab_size=10):
    """Generates simple sequences to 'copy'."""
    # Data: (seq_len, num_samples)
    data = np.random.randint(2, vocab_size, size=(seq_len, num_samples))
    return data
