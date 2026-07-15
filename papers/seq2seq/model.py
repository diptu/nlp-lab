import numpy as np

from src.layers.embedding import Embedding
from src.layers.linear import Linear
from src.layers.lstm import LSTMCell


class Seq2Seq:
    def __init__(
        self, input_vocab_size, output_vocab_size, embed_dim, hidden_dim, input_seq_len
    ):
        self.encoder = LSTMCell(embed_dim, hidden_dim)
        self.decoder = LSTMCell(embed_dim, hidden_dim)
        self.embedding = Embedding(input_vocab_size, embed_dim)
        self.output_layer = Linear(hidden_dim, output_vocab_size)
        self.hidden_dim = hidden_dim
        self.input_seq_len = input_seq_len

    def get_grads(self):
        return {
            "enc_w": self.encoder.dW,
            "enc_b": self.encoder.db,
            "dec_w": self.decoder.dW,
            "dec_b": self.decoder.db,
            "out_w": self.output_layer.dW,
            "out_b": self.output_layer.db,
            "embed_w": self.embedding.dW,
        }

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.embedding.zero_grad()
        self.output_layer.zero_grad()

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        """
        input_seq: (seq_len, batch_size)
        target_seq: (seq_len, batch_size)
        teacher_forcing_ratio: Probability of using ground truth during decoding.
        """
        batch_size = input_seq.shape[1]

        # 1. Encode
        h_enc, c_enc = (
            np.zeros((batch_size, self.hidden_dim)),
            np.zeros((batch_size, self.hidden_dim)),
        )
        for t in range(input_seq.shape[0]):
            x_emb = self.embedding.forward(input_seq[t])
            h_enc, c_enc = self.encoder.forward(x_emb, h_enc, c_enc)

        # 2. Decode
        h_dec, c_dec = h_enc, c_enc
        logits_list = []
        curr_input = target_seq[0]  # <SOS> token

        for t in range(target_seq.shape[0] - 1):
            x_emb = self.embedding.forward(curr_input)
            h_dec, c_dec = self.decoder.forward(x_emb, h_dec, c_dec)
            logits = self.output_layer.forward(h_dec)
            logits_list.append(logits)

            # --- Scheduled Sampling Logic ---
            # Decide whether to use Teacher Forcing or the model's own prediction
            if np.random.rand() < teacher_forcing_ratio:
                # Use ground truth
                curr_input = target_seq[t + 1]
            else:
                # Use model's own prediction (argmax of logits)
                curr_input = np.argmax(logits, axis=1)

        return np.stack(logits_list)

    def backward(self, grad_logits):
        """
        grad_logits: (seq_len, batch_size, vocab_size)
        """
        batch_size = grad_logits.shape[1]

        # Initialize gradients based on hidden dimension, not layer attributes
        grad_h_dec = np.zeros((batch_size, self.hidden_dim))
        grad_c_dec = np.zeros((batch_size, self.hidden_dim))

        # 1. Backprop through Decoder
        for t in reversed(range(len(grad_logits))):
            grad_h_dec += self.output_layer.backward(grad_logits[t])
            grad_x, grad_h_dec, grad_c_dec = self.decoder.backward(
                grad_h_dec, grad_c_dec
            )
            self.embedding.backward(grad_x)

        # 2. Backprop through Encoder
        # grad_h_dec/c_dec now contain the gradients to pass into the Encoder
        for t in reversed(range(self.input_seq_len)):
            grad_x, grad_h_dec, grad_c_dec = self.encoder.backward(
                grad_h_dec, grad_c_dec
            )
            self.embedding.backward(grad_x)

    def get_params(self):
        return {
            "enc_w": self.encoder.W,
            "dec_w": self.decoder.W,
            "out_w": self.output_layer.W,
        }
