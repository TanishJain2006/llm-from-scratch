import numpy as np

# ---------------------------------------------
# Layer Normalization
# ---------------------------------------------

# Input from Feed Forward Network
ffn_output = np.array([
    [1.28, 1.46, 1.65],
    [1.15, 1.31, 1.48],
    [0.79, 0.90, 1.02]
])

print("=== Feed Forward Output ===")
print(ffn_output)

# ---------------------------------------------
# Calculate Mean
# ---------------------------------------------

mean = np.mean(ffn_output, axis=1, keepdims=True)

print("\n=== Mean ===")
print(mean)

# ---------------------------------------------
# Calculate Variance
# ---------------------------------------------

variance = np.var(ffn_output, axis=1, keepdims=True)

print("\n=== Variance ===")
print(variance)

# ---------------------------------------------
# Layer Normalization
# ---------------------------------------------

epsilon = 1e-5

normalized_output = (ffn_output - mean) / np.sqrt(variance + epsilon)

print("\n=== Layer Normalized Output ===")
print(normalized_output)

# ---------------------------------------------
# Shape Information
# ---------------------------------------------

print("\n=== Shape Summary ===")
print("Input Shape :", ffn_output.shape)
print("Output Shape:", normalized_output.shape)
