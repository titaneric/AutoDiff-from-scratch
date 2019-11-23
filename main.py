#pylint: disable=no-member
import warnings

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cbook
import numpy as _np

import autodiff as ad
from autodiff import value_and_grad, value, grad
from autodiff.utils.datasets import Dataset, DataLoader
from autodiff.nn.optimizer import GradientDescent
from autodiff.nn.criterion import MSE

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def model(W, feed_dict={}):
    return ad.dot(ad.Placeholder(x=feed_dict['x']), ad.Variable(W=W))

data_size = 100
X = 2 * _np.random.rand(data_size, 1)
aug_X = _np.c_[_np.ones(data_size), X]
theta = _np.array([[5], [10]])
Y = aug_X @ theta + _np.random.randn(data_size, 1)
print("training shape", X.shape, Y.shape)

W = _np.random.random((2, 1))
parameters = [W]

dataset = Dataset(aug_X, Y)
dataloader = DataLoader(dataset)

# print(W)
lr = 0.001
opt = GradientDescent(lr, parameters, ["W"])
loss_func = MSE()
epoch = 100
for i in range(epoch):
    x, y = dataloader.next_batch(50)
    predicted_y = value(model)(W=W, feed_dict={'x': x})
    v, loss_grad = loss_func.calc_loss(y, predicted_y)
    model_grad = grad(model, upstream=loss_grad)(W=W, feed_dict={'x': x})
    opt.step(model_grad)
    print(i, v)

print(W)
print(theta)
predict = value(model)(W=W, feed_dict={'x': aug_X})

plt.scatter(
    X,
    Y,
)
plt.plot(X, predict, 'r')
# plt.axis('off')
plt.savefig('result.png')
plt.show()
