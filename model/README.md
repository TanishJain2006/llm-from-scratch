# GPT Model

This module combines all previously implemented components to form a simplified GPT-style decoder-only Transformer architecture.

The objective is to understand how individual modules interact to process text and generate contextual representations.

---

## Pipeline

```text
Input Text
      │
      ▼
Tokenizer
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
Transformer Block × N
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

---

## Components

- Tokenizer
- Embeddings
- Self-Attention
- Feed Forward Network
- Layer Normalization
- Residual Connections
- Transformer Block
- Output Projection

---

## Learning Objectives

- Understand how all Transformer components work together.
- Build a simplified GPT model from scratch.
- Learn the complete forward pass of a decoder-only Transformer.
- Prepare the architecture for future training and text generation.

---

## Files

### gpt_model.py

Implements the complete GPT model by integrating all previously built modules.

---

## Future Improvements

- Multi-Head Attention
- Positional Encoding
- Causal Masking
- Training Loop
- Loss Function
- Text Generation
- Model Inference
