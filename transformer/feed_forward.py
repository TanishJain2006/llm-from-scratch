import numpy as np

# Input from Self Attention

attention_output = np.array([
    [0.52, 0.61, 0.70],
    [0.48, 0.55, 0.63],
    [0.30, 0.38, 0.42]
])

print("Attention Output:")
print(attention_output)

# Feed Forward Weight Matrix

W = np.array([
    [0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7],
    [0.8, 0.9, 1.0]
])

print("\nWeight Matrix:")
print(W)

# Linear Transformation

ffn_output = np.matmul(attention_output, W)

print("\nLinear Output:")
print(ffn_output)

# ReLU Activation

relu_output = np.maximum(0, ffn_output)

print("\nReLU Output:")
print(relu_output)
