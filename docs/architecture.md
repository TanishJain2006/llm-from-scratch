# LLM Architecture Overview

This project follows the architecture of a decoder-only Transformer (similar to GPT) and builds every component step by step.

## High-Level Pipeline

```text
Input Text
      │
      ▼
Tokenization
      │
      ▼
Vocabulary Mapping
      │
      ▼
Token Embeddings
      │
      ▼
Positional Embeddings
      │
      ▼
Self-Attention
      │
      ▼
Feed Forward Network
      │
      ▼
Transformer Block
      │
      ▼
Stack Multiple Transformer Blocks
      │
      ▼
Layer Normalization
      │
      ▼
Linear Output Layer
      │
      ▼
Next Token Prediction
```

## Components

### 1. Tokenization
Converts raw text into tokens.

### 2. Vocabulary
Maps every token to a unique integer ID.

### 3. Embeddings
Transforms token IDs into dense vector representations.

### 4. Positional Embeddings
Provides information about the order of tokens.

### 5. Self-Attention
Allows each token to attend to other relevant tokens.

### 6. Feed Forward Network
Processes token representations independently after attention.

### 7. Transformer Block
Combines Self-Attention, Feed Forward Network, Residual Connections, and Layer Normalization.

### 8. GPT Model
Stacks multiple Transformer blocks to perform next-token prediction.
