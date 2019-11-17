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


def known_model(feed_dict={}):
    return ad.dot(ad.Placeholder(x=feed_dict['x']), ad.Constant([[2], [3]]))


# def mse(feed_dict={}):
# diff = ad.subtract(ad.Placeholder(predicted_y=feed_dict['predicted_y']), ad.Placeholder(true_y=feed_dict['true_y']))
# return ad.multiply(ad.dot(diff, diff), ad.Constant(1/2))


def mse(feed_dict={}):
    diff = ad.subtract(ad.Placeholder(predicted_y=feed_dict['predicted_y']),
                       ad.Placeholder(true_y=feed_dict['true_y']))
    return ad.multiply(ad.power(diff, ad.Constant(2)), ad.Constant(1 / 2))


def train_model():
    data_size = 100
    X = 2 * _np.random.rand(data_size, 1)
    aug_X = _np.c_[_np.ones(data_size), X]
    theta = _np.array([[5], [10]])
    Y = aug_X @ theta + _np.random.randn(data_size, 1)
    print("training shape", X.shape, Y.shape)

    W = _np.random.random((2, 1))

    dataset = Dataset(aug_X, Y)
    dataloader = DataLoader(dataset)

    # print(W)
    lr = 0.001
    opt = GradientDescent(lr, W)
    loss_func = MSE()
    epoch = 100
    for i in range(epoch):
        x, y = dataloader.next_batch(50)
        predicted_y, grad_value = value_and_grad(model, 'W')(W=W,
                                                             feed_dict={
                                                                 'x': x
                                                             })
        v, loss_grad = loss_func.calc_loss(y, predicted_y)
        opt.step(grad_value, loss_grad)
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


def test_known_model():
    data_size = 10
    x = ad.linspace(-7, 7, data_size)
    x = _np.c_[_np.ones(data_size), x]
    # print(x)
    print(grad(known_model, 'x')(feed_dict={'x': x}))


def test_loss():
    data_size = 100
    X = 2 * _np.random.rand(data_size, 1)
    aug_X = _np.c_[_np.ones(data_size), X]
    theta = _np.array([[5], [10]])
    Y = aug_X @ theta + _np.random.randn(data_size, 1)
    predict = value(known_model)(feed_dict={'x': aug_X})
    print(predict)
    v, l = value_and_grad(mse, 'predicted_y')(feed_dict={
        'predicted_y': predict,
        'true_y': Y
    })
    print(v, l)


def test_train():
    data_size = 100
    X = 2 * _np.random.rand(data_size, 1)
    aug_X = _np.c_[_np.ones(data_size), X]
    theta = _np.array([[5], [10]])
    Y = aug_X @ theta + _np.random.randn(data_size, 1)

    W = _np.random.random((2, 1))
    predicted_y, grad_value = value_and_grad(model)(W=W,
                                                    feed_dict={
                                                        'x': aug_X
                                                    })
    loss_grad = grad(mse, 'predicted_y')(feed_dict={
        'predicted_y': predicted_y,
        'true_y': Y
    })
    print(predicted_y.shape, Y.shape, X.shape)
    print(ad.array_equal(loss_grad, predicted_y - Y))
    print(ad.array_equal(aug_X.T, grad_value['W']))


if __name__ == "__main__":
    # print(grad(test2)(x=1, y=2, z=0))
    # print(grad(test)(z=2, y=-4, x=3, w=-1))
    train_model()