from layers.twolayer import TwoLayerNet
import numpy as np
from mnist.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=True, one_hot_label=True)
batch_size = 100
learning_rate = 0.1
iters_num = 10000
two_layer_net = TwoLayerNet()

res = []
res_train = []
res_test = []

for i in range(iters_num):
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x, t = x_train[batch_mask], t_train[batch_mask]
    grad = two_layer_net.gradient(x, t)
    for key in ['W1', 'b1', 'W2', 'b2']:
        two_layer_net.network[key] -= grad[key] * learning_rate
    loss = two_layer_net.loss(x, t)
    res.append(loss)


    if i % 100 == 0:
        train_acc = two_layer_net.accuracy(x_train, t_train)
        test_acc = two_layer_net.accuracy(x_test, t_test)
        res_train.append(train_acc)
        res_test.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(res_train))
plt.plot(x, res_train, label='train acc')
plt.plot(x, res_test, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()