import numpy as np

from layers.affine import Affine
from common.helper import softmax, sigmoid, gradient_descent, numerical_gradient, cross_entropy_error
from mnist.mnist import load_mnist
from PIL import Image
import pickle


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

def loss(W):
    x, t = np.array(x_train[: 10]), np.array(t_train[:10])
    y = predict(network, x)
    return cross_entropy_error(y, t)

(x_train, t_train), (x, t) = load_mnist(flatten=True,normalize=True, one_hot_label=True)
batch_size = 100
network = {}
def generate_network():
    network['W1'] = np.random.randn(784, 50)
    network['W2'] = np.random.randn(50, 100)
    network['W3'] = np.random.randn(100, 10)
    network['b1'] = np.random.randn(50)
    network['b2'] = np.random.randn(100)
    network['b3'] = np.random.randn(10)

    network['W1'] = gradient_descent(loss, network['W1'])
    network['W2'] = gradient_descent(loss, network['W2'])
    network['W3'] = gradient_descent(loss, network['W3'])

    print(network)
    with open("mnist\sample_weight1.pkl", "wb") as f:  # wb = write binary
        pickle.dump(network, f)

try:
    with open("mnist\sample_weight1.pkl", 'rb') as f:
        network = pickle.load(f)
except:
    generate_network()


accuracy_cnt = 0
batch_size = 100
for i in range(0, len(x), batch_size):
    y = predict(network, x[i: i + batch_size])

    p = np.argmax(y, axis = 1) # 获取概率最高的元素的索引 第一维就是取每行的最大值的索引
    accuracy_cnt += sum(y == t[i: i + batch_size])
print(accuracy_cnt)
print("Accuracy:" + str(float(sum(accuracy_cnt)) / len(x)))
