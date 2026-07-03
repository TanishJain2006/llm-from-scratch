"""
Basic Embeddings Implementation

Objective:
Understand how token IDs are converted into embedding vectors.
"""

# Step 1: Vocabulary
vocabulary = {
    "I": 0,
    "love": 1,
    "AI": 2
}

print("Vocabulary:")
print(vocabulary)

# Step 2: Input Sentence
sentence = "I love AI"

print("\nSentence:")
print(sentence)

# Step 3: Tokenization
tokens = sentence.split()

print("\nTokens:")
print(tokens)

# Step 4: Convert Tokens to IDs
token_ids = [vocabulary[token] for token in tokens]

print("\nToken IDs:")
print(token_ids)

# Step 5: Embedding Matrix
embedding_matrix = {
    0: [0.70, 0.80, 0.90],
    1: [0.40, 0.50, 0.60],
    2: [0.10, 0.20, 0.30]
}

print("\nEmbedding Matrix:")
print(embedding_matrix)

# Step 6: Convert IDs into Embeddings
embeddings = [embedding_matrix[token] for token in token_ids]

print("\nEmbedding Vectors:")

for token, vector in zip(tokens, embeddings):
    print(f"{token} --> {vector}")
