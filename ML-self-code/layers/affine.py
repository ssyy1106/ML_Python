import numpy as np

class Affine:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.cache = None
        self.dW = None
        self.db = None  # Gradient of weights
    
    def forward(self, x):
        out = x.dot(self.W) + self.b
        self.cache = x
        return out
    
    def backward(self, dout):
        x = self.cache
        self.dW = x.T.dot(dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = dout.dot(self.W.T)
        return dx

