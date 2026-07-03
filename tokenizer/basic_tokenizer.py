"""
Basic Tokenizer Implementation

Author: Tanish Jain

Objective:
Understand how raw text is converted into tokens
and token IDs before entering an LLM.
"""

# ----------------------------
# Input Sentence
# ----------------------------

sentence = "I love AI"

print("Sentence:")
print(sentence)

# ----------------------------
# Word Tokenization
# ----------------------------

tokens = sentence.split()

print("\nTokens:")
print(tokens)

# ----------------------------
# Vocabulary Creation
# ----------------------------

vocabulary = {}

for index, token in enumerate(sorted(set(tokens))):
    vocabulary[token] = index

print("\nVocabulary:")
print(vocabulary)

# ----------------------------
# Token Encoding
# ----------------------------

token_ids = [vocabulary[token] for token in tokens]

print("\nToken IDs:")
print(token_ids)
