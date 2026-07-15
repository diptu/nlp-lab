from papers.seq2seq.inference import Inference
from papers.seq2seq.model import Seq2Seq
from papers.seq2seq.trainer import Trainer
from src.training.loss import CrossEntropyLoss
from src.utils.tokenizer import Tokenizer


def decode_prediction(prediction, tokenizer):
    """Convert token IDs to readable tokens."""
    indices = [int(token) for token in prediction]

    tokens = [tokenizer.id_to_token.get(idx, f"<UNK:{idx}>") for idx in indices]

    return tokens


def print_readable_prediction(prediction, tokenizer, label="Prediction"):
    tokens = decode_prediction(prediction, tokenizer)
    print(f"{label}: {tokens}")


def train():

    # ==========================================================
    # 1. Raw Dataset
    # ==========================================================

    raw_english = [
        "I like cats.",
        "I like dogs.",
        "I love math.",
        "I love learning.",
        "I am learning AI.",
        "I like machine learning.",
        "I study computer science.",
        "I read books.",
        "I write code.",
        "I build models.",
    ]

    raw_bengali = [
        "আমি বিড়াল পছন্দ করি।",
        "আমি কুকুর পছন্দ করি।",
        "আমি গণিত ভালোবাসি।",
        "আমি শেখা ভালোবাসি।",
        "আমি এআই শিখছি।",
        "আমি মেশিন লার্নিং পছন্দ করি।",
        "আমি কম্পিউটার বিজ্ঞান পড়ি।",
        "আমি বই পড়ি।",
        "আমি কোড লিখি।",
        "আমি মডেল তৈরি করি।",
    ]

    # ==========================================================
    # 2. Tokenizers
    # ==========================================================

    eng_tok = Tokenizer()
    ben_tok = Tokenizer()

    tokenized_eng = [
        eng_tok.tokenize(eng_tok.normalize_english(sentence))
        for sentence in raw_english
    ]

    tokenized_ben = [
        ben_tok.tokenize(ben_tok.normalize_bengali(sentence))
        for sentence in raw_bengali
    ]

    # ==========================================================
    # 3. Build Vocabulary
    # ==========================================================

    eng_tok.build_vocab(tokenized_eng)
    ben_tok.build_vocab(tokenized_ben)

    # ==========================================================
    # 4. Encode and Pad
    # ==========================================================

    eng_encoded = [eng_tok.encode(tokens) for tokens in tokenized_eng]

    ben_encoded = [ben_tok.encode(tokens) for tokens in tokenized_ben]

    X = eng_tok.pad_sequences(eng_encoded).T
    Y = ben_tok.pad_sequences(ben_encoded).T

    # Shape:
    # X -> (sequence_length, batch_size)
    # Y -> (sequence_length, batch_size)

    seq_len = X.shape[0]

    print(f"English X shape: {X.shape}")
    print(f"Bengali Y shape: {Y.shape}")
    print(f"English vocabulary size: {len(eng_tok.token_to_id)}")
    print(f"Bengali vocabulary size: {len(ben_tok.token_to_id)}")

    # ==========================================================
    # 5. Initialize Model
    # ==========================================================

    model = Seq2Seq(
        input_vocab_size=len(eng_tok.token_to_id),
        output_vocab_size=len(ben_tok.token_to_id),
        embed_dim=32,
        hidden_dim=64,
        input_seq_len=seq_len,
    )

    trainer = Trainer(model, lr=0.01)
    criterion = CrossEntropyLoss()

    # ==========================================================
    # 6. Training
    # ==========================================================

    for epoch in range(5000):
        # Train on actual English -> Bengali data
        loss = trainer.train_step(X, Y, criterion)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss:.4f}")

    # ==========================================================
    # 7. Inference
    # ==========================================================

    inference = Inference(model)

    # Select one real sentence from the dataset
    sample_index = 0

    test_seq = X[:, sample_index : sample_index + 1]

    # ==========================================================
    # 8. Print Actual English Input Sentence
    # ==========================================================

    print("\n" + "=" * 60)
    print("INFERENCE")
    print("=" * 60)

    print(f"Actual English: {raw_english[sample_index]}")

    print(f"Input Token IDs: {test_seq.flatten()}")

    print_readable_prediction(test_seq.flatten(), eng_tok, label="Input Tokens")

    # ==========================================================
    # 9. Generate Bengali Translation
    # ==========================================================

    predicted = inference.generate(test_seq, start_token=test_seq[0, 0])

    print(f"Predicted Token IDs: {predicted}")

    print_readable_prediction(predicted, ben_tok, label="Predicted Bengali")

    print("=" * 60)


if __name__ == "__main__":
    train()
