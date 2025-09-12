import numpy as np

from layers.affine import Affine
from common.helper import softmax, sigmoid
from mnist.mnist import load_mnist
from PIL import Image
import pickle

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

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

(x_train, t_train), (x, t) = load_mnist(flatten=True,normalize=True)

with open("mnist\sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

print(f"w1: {network['W1'].shape} w2: {network['W2'].shape} w3: {network['W3'].shape} x shape: {t.shape}")

accuracy_cnt = 0
batch_size = 100
for i in range(0, len(x), batch_size):
    y = predict(network, x[i: i + batch_size])

    p = np.argmax(y, axis = 1) # 获取概率最高的元素的索引 第一维就是取每行的最大值的索引
    accuracy_cnt += sum(p == t[i: i + batch_size])
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

