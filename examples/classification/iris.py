#pylint: disable=no-member
import warnings

import numpy as _np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import matplotlib.cbook

import autodiff as ad
from autodiff import value_and_grad, value, grad
from autodiff.utils.datasets import Dataset, DataLoader
from autodiff.nn.optimizer import GradientDescent
from autodiff.nn.criterion import CrossEntropy
from classifier import grad_cross_entropy, cross_entropy

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def softmax(feed_dict={}):
    """
        X size is (batch_size, num_class)
    """
    exps = ad.Constant(ad.exp(ad.Placeholder(X=feed_dict["X"])))
    return ad.true_divide(
        exps, ad.reshape(ad.sum(exps, axis=1), ad.Constant((-1, 1))))


def model(W, feed_dict={}):
    output1 = ad.dot(ad.Placeholder(x=feed_dict['x']), ad.Variable(W=W))
    # output2 = ad.dot(output1, ad.Variable(W2=W2))
    exps = ad.Constant(ad.exp(output1))
    return ad.true_divide(
        exps, ad.reshape(ad.sum(exps, axis=1), ad.Constant((-1, 1))))



X, Y = load_iris(return_X_y=True)
num_classes = 3

# enc = OneHotEncoder(categories="auto")
Y = Y.reshape(-1, 1)
# Y = enc.fit_transform(Y).toarray()

data_size, num_features = X.shape

# W1 = _np.random.random((num_features, 5))
# W2 = _np.random.random((5, num_classes))

# parameters = [W1, W2]
W = _np.random.random((num_features, num_classes))
parameters = [W]

dataset = Dataset(X, Y)
dataloader = DataLoader(dataset)

lr = 1e-5
opt = GradientDescent(lr, parameters, ["W"])
loss_func = CrossEntropy()
epoch = 300
batch_size = 20
loss_list = []
for i in range(epoch):
    x, y = dataloader.next_batch(batch_size)
    # print(x, y)
    predicted_y = value(model)(W=W, feed_dict={'x': x})
    # print(y, predicted_y)
    # v, loss_grad = loss_func(y, predicted_y)
    v = cross_entropy(predicted_y, y)
    loss_grad = grad_cross_entropy(predicted_y, y)
    # print(loss_grad)
    model_grad = grad(model, upstream=loss_grad)(W=W,
                                                 feed_dict={
                                                     'x': x
                                                 })
    opt.step(model_grad)
    loss_list.append(v)
    # break

plt.plot(range(epoch), loss_list)
plt.savefig('Iris.png')
plt.show()