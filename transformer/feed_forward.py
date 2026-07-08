import numpy as np

# ---------------------------------------------
# Feed Forward Network (FFN)
# ---------------------------------------------

# Input from Self-Attention
attention_output = np.array([
    [0.52, 0.61, 0.70],
    [0.48, 0.55, 0.63],
    [0.30, 0.38, 0.42]
])

print("=== Self Attention Output ===")
print(attention_output)

# ---------------------------------------------
# First Linear Layer
# ---------------------------------------------

W1 = np.array([
    [0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7],
    [0.8, 0.9, 1.0]
])

print("\n=== Weight Matrix 1 ===")
print(W1)

linear_output = np.matmul(attention_output, W1)

print("\n=== Linear Layer 1 Output ===")
print(linear_output)

# ---------------------------------------------
# ReLU Activation
# ---------------------------------------------

relu_output = np.maximum(0, linear_output)

print("\n=== ReLU Activation Output ===")
print(relu_output)

# ---------------------------------------------
# Second Linear Layer
# ---------------------------------------------

W2 = np.array([
    [0.3, 0.2, 0.1],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
])

print("\n=== Weight Matrix 2 ===")
print(W2)

final_output = np.matmul(relu_output, W2)

print("\n=== Feed Forward Network Output ===")
print(final_output)

# ---------------------------------------------
# Shape Information
# ---------------------------------------------

print("\n=== Shape Summary ===")
print("Input Shape :", attention_output.shape)
print("After Linear Layer 1 :", linear_output.shape)
print("After ReLU :", relu_output.shape)
print("Final Output Shape :", final_output.shape)
