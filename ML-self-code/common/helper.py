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

# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交叉熵误差
# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     batch_size = y.shape[0]
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta)) / batch_size

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
        it.iternext()   

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=10):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
