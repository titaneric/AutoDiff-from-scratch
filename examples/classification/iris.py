#pylint: disable=no-member
import warnings

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib.cbook

import autodiff as ad
from autodiff.utils.model_utils import Dataset, DataLoader, train_procedure
from autodiff.nn.optimizer import GradientDescent
from autodiff.nn.criterion import CrossEntropy
from autodiff.nn.layer import Module, Linear

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SimpleModel(Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear1 = Linear(num_features, 5)
        self.linear2 = Linear(5, num_classes)

    def forward(self, x):
        x = self.linear1(x, bridge=False)
        x = self.linear2(x)
        return x


X, Y = load_iris(return_X_y=True)
Y = Y.reshape(-1, 1)
data_size, num_features = X.shape
num_classes = 3

model = SimpleModel(num_features, num_classes)
dataset = Dataset(X, Y)
dataloader = DataLoader(dataset)

lr = 1e-5
opt = GradientDescent(lr, model.parameters())
loss_func = CrossEntropy()
epoch = 300
batch_size = 32
loss_list = []
for i in train_procedure(epoch):
    x, y = dataloader.next_batch(batch_size)
    predicted_y = model(x)
    v, loss_grad = loss_func(y, predicted_y)
    model_grad = model.backward(loss_grad)
    opt.step(model_grad)
    model.zero_grad()
    loss_list.append(v)

plt.plot(range(epoch), loss_list)
plt.savefig('Iris.png')
plt.show()