import numpy as np


class Tokenizer:
    def __init__(self, special_tokens: list[str] | None = None) -> None:
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def normalize_english(self, sentence: str) -> str:
        return sentence.lower().strip().replace(".", " .")

    def normalize_bengali(self, sentence: str) -> str:
        return sentence.replace("।", " ।").strip()

    def tokenize(self, sentence: str) -> list[str]:
        return sentence.split()

    def build_vocab(self, tokenized_sentences: list[list[str]]) -> None:
        vocab = set(token for sent in tokenized_sentences for token in sent)
        sorted_vocab = self.special_tokens + sorted(list(vocab))
        self.token_to_id = {token: idx for idx, token in enumerate(sorted_vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, tokens: list[str]) -> np.ndarray:
        ids = [self.token_to_id["<SOS>"]]
        ids.extend(
            [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]
        )
        ids.append(self.token_to_id["<EOS>"])
        return np.array(ids, dtype=np.int32)

    def pad_sequences(self, sequences: list[np.ndarray]) -> np.ndarray:
        max_length = max(len(seq) for seq in sequences)
        pad_id = self.token_to_id["<PAD>"]
        padded_array = np.full((len(sequences), max_length), pad_id, dtype=np.int32)
        for i, seq in enumerate(sequences):
            padded_array[i, : len(seq)] = seq
        return padded_array

    def one_hot_encode(self, sequence: np.ndarray) -> np.ndarray:
        return np.eye(len(self.token_to_id), dtype=np.float32)[sequence]

    def create_mask(self, padded_sequences: np.ndarray) -> np.ndarray:
        return (padded_sequences != self.token_to_id["<PAD>"]).astype(np.float32)


class EmbeddingLayer:
    def __init__(self, vocab_size: int, embed_dim: int):
        self.weights = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.1

    def forward(self, input_indices: np.ndarray) -> np.ndarray:
        return self.weights[input_indices]
