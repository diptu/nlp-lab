import numpy as np


class Metrics:
    @staticmethod
    def calculate_accuracy(
        predictions: np.ndarray, targets: np.ndarray, pad_token: int = 0
    ) -> float:
        """
        Calculates token-level accuracy, ignoring padding tokens.
        predictions: (batch_size, seq_len)
        targets: (batch_size, seq_len)
        """
        mask = targets != pad_token
        correct = (predictions == targets) & mask
        return np.sum(correct) / np.sum(mask)

    @staticmethod
    def perplexity(loss: float) -> float:
        """
        Calculates perplexity (exp of the average negative log-likelihood).
        """
        return np.exp(loss)

    @staticmethod
    def bleu_score_mock(prediction: list, reference: list) -> float:
        """
        A placeholder for a BLEU implementation.
        Note: True BLEU is complex to implement from scratch.
        Start with n-gram precision here.
        """
        # Simplistic implementation: 1-gram precision
        common = set(prediction) & set(reference)
        return len(common) / len(prediction) if len(prediction) > 0 else 0.0
