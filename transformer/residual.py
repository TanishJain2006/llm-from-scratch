import numpy as np

# ---------------------------------------------
# Residual Connection
# ---------------------------------------------

# Input to Transformer Block
input_tensor = np.array([
    [0.52, 0.61, 0.70],
    [0.48, 0.55, 0.63],
    [0.30, 0.38, 0.42]
])

print("=== Original Input ===")
print(input_tensor)

# ---------------------------------------------
# Output from Layer Normalization
# ---------------------------------------------

layer_norm_output = np.array([
    [-1.22, 0.00, 1.22],
    [-1.20, 0.05, 1.15],
    [-1.18, 0.08, 1.10]
])

print("\n=== Layer Normalization Output ===")
print(layer_norm_output)

# ---------------------------------------------
# Residual Connection
# ---------------------------------------------

residual_output = input_tensor + layer_norm_output

print("\n=== Residual Connection Output ===")
print(residual_output)

# ---------------------------------------------
# Shape Information
# ---------------------------------------------

print("\n=== Shape Summary ===")
print("Input Shape :", input_tensor.shape)
print("LayerNorm Shape :", layer_norm_output.shape)
print("Output Shape :", residual_output.shape)
