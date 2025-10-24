# PureMarkov
PureMarkov is a from-scratch implementation of **statistical language modeling** using variable-order Markov chains. Built entirely in Python without external NLP libraries, it demonstrates the core principles of probabilistic text generation and next-word prediction.

### 1. Download Training Data  
```bash
chmod +x  markovData.sh
./markovData.sh
```
This will download words from Project Gutenberg books and create `gutenberg_combined.txt` in the `MarkovData` folder.
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
- **Interactive CLI** for predictions and text generation
- **Comprehensive unit tests** (30 tests, all passing)
- **Minimal dependencies** - pure Python implementation

## What It Does
Given a corpus of text, PureMarkov learns transition probabilities between word sequences and uses them to:
- **Predict the next word** given a context
- **Generate coherent text sequences** probabilistically
- **Analyze text patterns** through statistical transitions
## How It Works
The implementation follows the classical approach to language modeling:
1. **Tokenization** - Normalize and split text into words
2. **Context Extraction** - Build n-gram contexts (order 1, 2, 3, etc.)
3. **Transition Counting** - Track what words follow each context
4. **Probability Normalization** - Convert counts to probability distributions
5. **Text Generation** - Sample from distributions to generate text
## Key Features
- **Variable-order modeling** (unigram, bigram, trigram, etc.)
- **Interactive CLI** for exploration and experimentation
- **Pure Python** - No ML frameworks, just probability and statistics
- **Fully tested** - 30 comprehensive unit tests
- **Educational** - Clear, readable code demonstrating fundamental NLP concepts
## The Math
For order-N Markov chains:
```
P(word_t+1 | word_{t-n+1}...word_{t}) = count(word_{t-n+1}...,word_{t},word_{t+1}) / SUM(count(word_{t-n+1}...word_{t},word'))
```
## Limitations & Trade-offs
- **Limited context** - No long-range dependencies (unlike neural networks)
- **No semantic understanding** - Pure statistical pattern matching
- **Memoryless** - Each prediction only considers the immediate context
- **But**: Fast, interpretable, and resource-efficient

  
