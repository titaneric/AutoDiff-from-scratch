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


def model(W1, W2, feed_dict={}):
    output1 = ad.dot(ad.Placeholder(x=feed_dict['x']), ad.Variable(W1=W1))
    return ad.dot(output1, ad.Variable(W2=W2))


X, Y = load_boston(return_X_y=True)
Y = Y.reshape(-1, 1)
data_size, num_features = X.shape

W1 = _np.random.random((num_features, 5))
W2 = _np.random.random((5, 1))
parameters = [W1, W2]

dataset = Dataset(X, Y)
dataloader = DataLoader(dataset)

lr = 1e-8
opt = GradientDescent(lr, parameters, ["W1", "W2"])
loss_func = MSE()
epoch = 200
batch_size = 16
loss_list = []
for i in range(epoch):
    x, y = dataloader.next_batch(batch_size)
    predicted_y = value(model)(W1=W1, W2=W2, feed_dict={'x': x})
    v, loss_grad = loss_func.calc_loss(y, predicted_y)
    model_grad = grad(model, upstream=loss_grad)(W1=W1, W2=W2, feed_dict={'x': x})
    opt.step(model_grad)
    loss_list.append(v)

plt.plot(range(epoch), loss_list)
plt.savefig('Boston.png')
plt.show()