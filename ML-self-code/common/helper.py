import numpy as np
import matplotlib.pylab as plt

def softmax(x):
    # 数值稳定性处理，减去最大值，并指定 axis
    maximum = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - maximum)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_exp_x

def sigmoid(x):
    """Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_derivative(x):
    """Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds

def relu(x):
    """Compute the ReLU function for the input here.

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0, x)
    return s

def relu_derivative(x):
    """Compute the gradient (also called the slope or derivative) of the ReLU function with respect to its input x.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    ds = np.where(x > 0, 1, 0)
    return ds

