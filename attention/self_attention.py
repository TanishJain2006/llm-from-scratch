import numpy as np

class SelfAttention:
    """
    Basic implementation of Scaled Dot-Product Self-Attention.
    """

    def __init__(self):
        pass

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def forward(self, Q, K, V):
        d_k = K.shape[-1]

        scores = np.matmul(Q, K.T) / np.sqrt(d_k)

        attention_weights = self.softmax(scores)

        output = np.matmul(attention_weights, V)

        return output, attention_weights
