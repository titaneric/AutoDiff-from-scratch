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
from autodiff.nn.layer import Module, Linear

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# def model(W1, W2, feed_dict={}):
#     output1 = ad.dot(ad.Placeholder(x=feed_dict['x']), ad.Variable(W1=W1))
#     return ad.dot(output1, ad.Variable(W2=W2))


class SimpleModel(Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear1 = Linear(num_features, 5)
        self.linear2 = Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x, bridge=False)
        x = self.linear2(x)
        return x


X, Y = load_boston(return_X_y=True)
Y = Y.reshape(-1, 1)
data_size, num_features = X.shape

model = SimpleModel(num_features)
dataset = Dataset(X, Y)
dataloader = DataLoader(dataset)

lr = 1e-8
opt = GradientDescent(lr, model.parameters())
loss_func = MSE()
epoch = 200
batch_size = 2
loss_list = []
for i in range(epoch):
    x, y = dataloader.next_batch(batch_size)
    predicted_y = model(x)
    v, loss_grad = loss_func(y, predicted_y)
    model_grad = model.backward(loss_grad)
    opt.step(model_grad)
    model.zero_grad()
    loss_list.append(v)
    print(i, v)

plt.plot(range(epoch), loss_list)
plt.savefig('Boston.png')
plt.show()