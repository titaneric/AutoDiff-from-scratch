#pylint: disable=no-member
import warnings

import matplotlib.pyplot as plt
import matplotlib.cbook
import numpy as _np

import autodiff as ad
from autodiff.utils.model_utils import Dataset, DataLoader, train_procedure
from autodiff.nn.optimizer import GradientDescent
from autodiff.nn.criterion import MSE
from autodiff.nn.layer import Module, Linear

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# def model(W, feed_dict={}):
#     return ad.dot(ad.Placeholder(x=feed_dict['x']), ad.Variable(W=W))

class SimpleModel(Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x, bridge=False)

data_size = 300
X = 2 * _np.random.rand(data_size, 1)
aug_X = _np.c_[_np.ones(data_size), X]

# True parameters
theta = _np.array([[5], [10]])
Y = aug_X @ theta + _np.random.randn(data_size, 1)
print("training shape", X.shape, Y.shape)

model = SimpleModel(2)
dataset = Dataset(aug_X, Y)
dataloader = DataLoader(dataset)

lr = 0.001
opt = GradientDescent(lr, model.parameters())
loss_func = MSE()
epoch = 300
for i in train_procedure(epoch):
    x, y = dataloader.next_batch(10)
    predicted_y = model(x)
    v, loss_grad = loss_func(y, predicted_y)
    model_grad = model.backward(loss_grad)
    opt.step(model_grad)
    model.zero_grad()

predict = model(aug_X)

plt.scatter(
    X,
    Y,
)
plt.plot(X, predict, 'r')
# plt.axis('off')
plt.savefig('regression.png')
plt.show()
