#pylint: disable=no-member
import warnings

import numpy as _np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import matplotlib.cbook

import autodiff as ad
from autodiff import value_and_grad, value, grad
from autodiff.utils.datasets import Dataset, DataLoader
from autodiff.nn.optimizer import GradientDescent
from autodiff.nn.criterion import MSE

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def model(W, feed_dict={}):
    return ad.dot(ad.Placeholder(x=feed_dict['x']), ad.Variable(W=W))


X, Y = load_boston(return_X_y=True)
Y = Y.reshape(-1, 1)
data_size, num_features = X.shape

W = _np.random.random((num_features, 1))

dataset = Dataset(X, Y)
dataloader = DataLoader(dataset)

lr = 1e-8
opt = GradientDescent(lr, W)
loss_func = MSE()
epoch = 500
loss_list = []
for i in range(epoch):
    x, y = dataloader.next_batch(5)
    predicted_y, grad_value = value_and_grad(model, 'W')(W=W,
                                                         feed_dict={
                                                             'x': x
                                                         })
    v, loss_grad = loss_func.calc_loss(y, predicted_y)
    opt.step(grad_value, loss_grad)
    print(i, v)
    loss_list.append(v)

plt.plot(range(epoch), loss_list)
plt.savefig('Boston.png')
plt.show()