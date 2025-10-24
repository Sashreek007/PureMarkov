# PureMarkov
PureMarkov is a from-scratch implementation of next-word prediction using Markov chains, written in Python with a functional programming approach. Instead of relying on external NLP libraries, the project constructs every component manually — from tokenization and transition mapping to probability normalization and sampling.
## Quick Start

### 1. Download Training Data
```bash
python3 download_data.py
```
This will download ~10 million words from Project Gutenberg books and create `gutenberg_combined.txt` in the `MarkovData` folder.

### 2. Run the CLI
```bash
python3 markovCLI.py
```
Then:
- Select an order (1, 2, or 3)
- The program will automatically load the training data
- Start predicting words or generating text!

## Features

- **Variable-order Markov chains** (order 1, 2, 3+)
- **Fast training** on large datasets (9.7M words in ~8 seconds)
- **Interactive CLI** for predictions and text generation
- **Comprehensive unit tests** (30 tests, all passing)
- **Minimal dependencies** - pure Python implementation
