import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.out = y
        return y

    def backward(self, dout):
        return self.out * (1.0 - self.out) * dout
