# Byte Pair Encoding (BPE) Tokenizer

This directory contains implementations of Byte Pair Encoding (BPE) tokenizers, a fundamental component for text processing in Large Language Models (LLMs).

## Overview

Byte Pair Encoding is a data compression technique adapted for tokenization in natural language processing. It works by iteratively merging the most frequent pairs of bytes (or characters) in the training data to build a vocabulary of subword units. This approach allows the tokenizer to:

- Handle out-of-vocabulary words by breaking them into known subword units
- Achieve a balance between character-level and word-level tokenization
- Compress text efficiently while maintaining the ability to reconstruct the original input

## Implementations

This repository provides two BPE tokenizer implementations:

### 1. Basic BPE (`BPE` class)

A straightforward implementation of the BPE algorithm that operates on raw UTF-8 encoded bytes.

**Key Features:**
- Simple byte-level tokenization
- Builds vocabulary by merging most frequent byte pairs
- Suitable for understanding the core BPE algorithm

**Limitations:**
- Merges can occur across word boundaries
- May produce suboptimal tokens for natural language

### 2. Regex-Based BPE (`BPE_REGEX` class)

An enhanced implementation that uses regex pattern matching to split text into meaningful chunks before applying BPE.

**Key Features:**
- Pre-splits text using a regex pattern to identify:
  - Common contractions (e.g., 's, 't, 'll, 've, 're)
  - Words with optional leading non-alphanumeric characters
  - Numbers (up to 3 digits)
  - Whitespace patterns
  - Special characters and punctuation
- Performs BPE independently on each text chunk
- Prevents merges across natural text boundaries
- More aligned with production tokenizers (e.g., GPT-2/GPT-3)

**Regex Pattern:**
```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

## Usage

### Training a Tokenizer

```python
# Basic BPE
bpe_simple = BPE()
bpe_simple.train_bpe(vocab_size=320, inp=training_text)

# Regex-based BPE
bpe_regex = BPE_REGEX()
bpe_regex.train_bpe(vocab_size=320, inp=training_text)
```

The `vocab_size` parameter must be greater than 256 (the number of possible byte values). The tokenizer will perform `vocab_size - 256` merge operations to build the final vocabulary.

### Encoding Text

```python
# Encode text to token IDs
token_ids = bpe_regex.encode("hello world!!!? (안녕하세요!) lol123 😉")
```

### Decoding Tokens

```python
# Decode token IDs back to text
decoded_text = bpe_regex.decode(token_ids)
```

### Checking Compression

```python
# View compression statistics
bpe_regex.print_compression()
# Output:
# Original: 1000
# Processed: 450
# Compression ratio: 2.22x
```

## Implementation Details

### Core Components

Both implementations share common core methods:

1. **`__get_byte_freq()`**: Computes frequency of consecutive byte pairs
2. **`__replace_top_pair()`**: Replaces all occurrences of a byte pair with a new token
3. **`__create_vocab()`**: Builds the vocabulary mapping from token IDs to byte sequences
4. **`train_bpe()`**: Main training loop that performs iterative merging
5. **`encode()`**: Converts text to token IDs
6. **`decode()`**: Converts token IDs back to text

### Training Process

1. **Initialization**: Start with 256 base tokens (one for each possible byte value)
2. **Iteration**: For each merge operation:
   - Calculate frequency of all byte pairs in the current encoding
   - Find the most frequent pair
   - Create a new token to represent this pair
   - Replace all occurrences of the pair with the new token
   - Update the merge dictionary
3. **Vocabulary Creation**: Build final vocabulary mapping token IDs to byte sequences

### Encoding Process

1. Convert input text to UTF-8 bytes
2. For regex-based BPE: split text into chunks using regex pattern
3. Iteratively merge byte pairs following the learned merge order
4. Return final sequence of token IDs

### Decoding Process

1. Look up byte sequence for each token ID in vocabulary
2. Concatenate all byte sequences
3. Decode to UTF-8 text (with error handling for invalid sequences)

## Learning Resources

For a deeper understanding of BPE tokenization:

- **Basic Intuition**: [Let's build the GPT Tokenizer](https://youtu.be/fKd8s29e-l4?si=TSbl0E6pHgWIXHnb)
- **In-Depth Understanding**: [Andrej Karpathy - Let's build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=EOAf_ey5bXR_1Hw8)

## File Structure

- `bpe_basic_regex.ipynb`: Jupyter notebook containing both BPE implementations and usage examples

## Dependencies

- `regex`: For advanced regex pattern matching in the BPE_REGEX implementation
  ```bash
  pip install regex
  ```

## Example Notebook

The `bpe_basic_regex.ipynb` notebook can be run in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BilalAsifB/MyLLM/blob/main/tokenizer/bpe_basic_regex.ipynb)

## Key Differences: BPE vs BPE_REGEX

| Feature | BPE | BPE_REGEX |
|---------|-----|-----------|
| **Text Splitting** | No pre-splitting | Regex-based chunking |
| **Merge Boundaries** | Can cross word boundaries | Respects text structure |
| **Token Quality** | Less optimal for natural language | Better for natural language |
| **Speed** | Faster | Slightly slower |
| **Use Case** | Learning/experimentation | Production-like tokenization |

## Notes

- Both implementations use UTF-8 encoding as the base representation
- The `decode()` method includes error handling (`errors="replace"`) for invalid UTF-8 sequences
- The regex pattern in BPE_REGEX is designed to handle multiple languages and special characters
- Vocabulary size should be chosen based on the use case (common choices: 8K, 16K, 32K, 50K tokens)
