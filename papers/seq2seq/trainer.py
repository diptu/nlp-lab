import numpy as np

from src.training.optimizer import Adam


class Trainer:
    def __init__(self, model, lr=0.001, clip_value=5.0):
        self.model = model
        # Assuming your Adam optimizer takes initial params and learns the structure
        self.optimizer = Adam(model.get_params(), lr=lr)
        self.clip_value = clip_value

    def clip_grads(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Prevents exploding gradients by capping values."""
        for k in grads:
            grads[k] = np.clip(grads[k], -self.clip_value, self.clip_value)
        return grads

    def train_step(
        self, input_seq: np.ndarray, target_seq: np.ndarray, criterion
    ) -> float:
        """
        Executes one complete Forward -> Backward -> Update cycle.
        """
        # 1. Clear gradients from previous step
        self.model.zero_grad()

        # 2. Forward Pass
        # logits shape: (seq_len, batch_size, vocab_size)
        logits = self.model.forward(input_seq, target_seq)

        # 3. Loss Calculation
        # target_seq[1:] aligns with the next-token prediction logic
        loss = criterion.forward(logits, target_seq[1:])

        # 4. Backward Pass (BPTT)
        grad_logits = criterion.backward()
        self.model.backward(grad_logits)

        # 5. Extract, Clip, and Update
        grads = self.model.get_grads()
        grads = self.clip_grads(grads)

        # Update model parameters via Adam optimizer
        # Note: Ensure self.optimizer.update takes (params, grads)
        self.optimizer.update(self.model.get_params(), grads)

        return float(loss)
