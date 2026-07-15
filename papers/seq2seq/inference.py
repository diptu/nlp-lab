import numpy as np


class Inference:
    def __init__(self, model):
        self.model = model

    def generate(self, input_seq, start_token, max_len=20):
        """
        Greedy decoding: predict one token at a time.
        input_seq: (seq_len, 1) - single example
        """
        # 1. Encode the full input sequence
        h_enc, c_enc = (
            np.zeros((1, self.model.hidden_dim)),
            np.zeros((1, self.model.hidden_dim)),
        )

        for t in range(input_seq.shape[0]):
            x_emb = self.model.embedding.forward(input_seq[t])
            h_enc, c_enc = self.model.encoder.forward(x_emb, h_enc, c_enc)

        # 2. Decode greedily
        generated_indices = []
        h_dec, c_dec = h_enc, c_enc
        curr_input = np.array([start_token])

        for _ in range(max_len):
            x_emb = self.model.embedding.forward(curr_input)
            h_dec, c_dec = self.model.decoder.forward(x_emb, h_dec, c_dec)
            logits = self.model.output_layer.forward(h_dec)

            # Predict the most likely next token
            next_token = np.argmax(logits, axis=1)[0]
            generated_indices.append(next_token)

            # Update input for next iteration
            curr_input = np.array([next_token])

            # Stop if we hit an <EOS> token (assuming index 1 for EOS)
            if next_token == 1:
                break

        return generated_indices
