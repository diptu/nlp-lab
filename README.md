# nlp-lab

A laboratory for implementing influential NLP papers from scratch using NumPy. Designed to bridge the gap between theoretical research and low-level engineering mastery.

## Current Focus: Seq2Seq
*   **Architecture:** Custom LSTM cells, Encoder-Decoder framework.
*   **Techniques:** Teacher Forcing, Masked Cross-Entropy, Gradient Clipping.
*   **Optimization:** Adaptive Adam Optimizer with BPTT.

## Project Structure
```text
nlp-lab/
├── papers/             # Implementation modules categorized by research paper
│   └── seq2seq/        # [Sutskever et al., 2014] implementation
├── src/                # Shared utilities
│   ├── layers/         # Reusable mathematical primitives
│   ├── training/       # Engine, optimizers, and schedulers
│   └── utils/          # Tokenization and I/O
├── tests/              # Gradient checks and unit tests
└── main.py             # Entry point for running experiments
```

## Setup
- Built with uv for fast, isolated development.


# Initialize and sync environment
```bash
uv sync
```
# Execute a specific paper implementation
```bash
uv run main.py --paper seq2seq
```

## Contributing
- Each implementation follows a "first principles" approach:

- Paper Review: Theoretical breakdown of the core algorithm.

- Implementation: Build components using only NumPy.

- Verification: Gradient checking and reproduction of reported benchmarks.

## License
- MIT
