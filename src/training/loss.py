import numpy as np


class CrossEntropyLoss:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.softmax_out = None
        self.target = None
        self.mask = None
        self.seq_len = 0
        self.batch_size = 0

    def log_softmax(self, x):
        # Stable log-softmax
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return (x - np.max(x, axis=-1, keepdims=True)) - np.log(
            np.sum(e_x, axis=-1, keepdims=True)
        )

    def forward(self, logits, targets):
        self.seq_len, self.batch_size, vocab_size = logits.shape
        self.target = targets
        self.mask = (targets != self.pad_token_id).astype(float)

        # Softmax for backward pass
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        self.softmax_out = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Calculate loss
        log_probs = self.log_softmax(logits)
        target_flat = targets.reshape(-1)
        log_probs_flat = log_probs.reshape(-1, vocab_size)

        # Gather relevant log probabilities
        indices = np.arange(len(target_flat))
        selected_log_probs = log_probs_flat[indices, target_flat]

        # Apply mask and normalize
        loss = -selected_log_probs * self.mask.reshape(-1)
        return np.sum(loss) / (np.sum(self.mask) + 1e-8)  # Avoid div by zero

    def backward(self) -> np.ndarray:
        """
        Gradient of CE w.r.t logits: (softmax_out - target_one_hot) * mask
        """
        # 1. Create one-hot targets
        vocab_size = self.softmax_out.shape[-1]
        target_flat = self.target.reshape(-1)
        one_hot = np.zeros_like(self.softmax_out.reshape(-1, vocab_size))
        one_hot[np.arange(len(target_flat)), target_flat] = 1

        # 2. Gradient of Softmax-CE: (P - y)
        grad = self.softmax_out.reshape(-1, vocab_size) - one_hot

        # 3. Apply mask to gradients so we don't backprop through padding
        grad *= self.mask.reshape(-1, 1)

        # 4. Normalize by total non-padding count
        return (grad / (np.sum(self.mask) + 1e-8)).reshape(
            self.seq_len, self.batch_size, vocab_size
        )
