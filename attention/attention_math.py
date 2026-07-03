import numpy as np

# ------------------------------------
# Input Embeddings
# ------------------------------------

embeddings = np.array([
    [0.7, 0.8, 0.9],
    [0.4, 0.5, 0.6],
    [0.1, 0.2, 0.3]
])

print("Embeddings:")
print(embeddings)

# ------------------------------------
# Query, Key and Value
# ------------------------------------

Q = embeddings
K = embeddings
V = embeddings

# ------------------------------------
# Attention Scores
# ------------------------------------

scores = np.matmul(Q, K.T)

print("\nAttention Scores:")
print(scores)

# ------------------------------------
# Scale the Scores
# ------------------------------------

dk = K.shape[1]

scaled_scores = scores / np.sqrt(dk)

print("\nScaled Scores:")
print(scaled_scores)

# ------------------------------------
# Softmax
# ------------------------------------

def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)

print("\nAttention Weights:")
print(attention_weights)

# ------------------------------------
# Final Output
# ------------------------------------

output = np.matmul(attention_weights, V)

print("\nFinal Attention Output:")
print(output)
