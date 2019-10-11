#pylint: disable=no-member
import warnings

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cbook
import numpy as _np

import autodiff
from autodiff.diff import value_and_grad, value, grad
import autodiff.numpy_grad.wrapper as np
from utils.datasets import Dataset, DataLoader

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def tanh(x):
    return np.divide(
        np.subtract(np.Constant(1), np.exp(np.negative(np.Variable(x)))),
        np.add(np.Constant(1), np.exp(np.negative(np.Variable(x))))
    )

def test(x, y, z, w):
    return np.multiply(
                np.add(np.multiply(np.Variable(x=x), np.Variable(y=y)),
                    np.maximum(np.Variable(z=z), np.Variable(w=w))), 
                np.Constant(2)
            )

def test2(x, y, z):
    return np.multiply(
        np.add(np.Variable(x=x), np.Variable(y=y)),
        np.maximum(np.Variable(y=y),np.Variable(z=z))
    )


def test3(x):
    return np.power(np.Variable(x=x), np.Constant(3))

def test4(x):
    return np.multiply(np.power(np.Variable(x=x), np.Constant(2)), np.Constant(1/2))

def model(W, feed_dict={}):
    return np.dot(np.Placeholder(x=feed_dict['x']), np.Variable(W=W))

def known_model(feed_dict={}):
    return np.dot(np.Placeholder(x=feed_dict['x']), np.Constant([2, 3]))

# def mse(feed_dict={}):
    # diff = np.subtract(np.Placeholder(predicted_y=feed_dict['predicted_y']), np.Placeholder(true_y=feed_dict['true_y']))
    # return np.multiply(np.dot(diff, diff), np.Constant(1/2))

def mse(feed_dict={}):
    diff = np.subtract(np.Placeholder(predicted_y=feed_dict['predicted_y']), np.Placeholder(true_y=feed_dict['true_y']))
    return np.multiply(np.power(diff, np.Constant(2)), np.Constant(1/2))

def power_demo():
    x_list = np.linspace(-7, 7, 200)
    # x_list = 5
    y_list, dy_list = value_and_grad(test4, 'x')(x=x_list)
    # print(y_list, dy_list)

    plt.plot(
        x_list, y_list,
        x_list, dy_list
    )
    # plt.axis('off')
    plt.savefig('x**3.png')
    plt.show()

def train_model():
    data_size = 100
    X = np.linspace(-7, 7, data_size)
    aug_X = _np.c_[_np.ones(data_size), X]
    noise = _np.random.randint(-3, 3, (data_size))    
    theta = _np.array([10, 5])
    Y = _np.dot(aug_X, theta) + noise
    # print(Y.shape, aug_X.shape)

    W = _np.random.random((2, 1)) 

    dataset = Dataset(aug_X, Y)
    dataloader = DataLoader(dataset)

    lr = 0.001
    epoch = 300
    for _ in range(epoch):
        x, y = dataloader.next_batch(10)
        predicted_y, grad_value= value_and_grad(model)(W=W, feed_dict={'x': x})
        # print(predicted_y.shape, y.shape)
        # print((predicted_y-y[:,None]).shape)
        loss_grad = grad(mse, 'predicted_y')(feed_dict={'predicted_y': predicted_y, 'true_y': y})
        W -= lr * _np.dot(grad_value['W'],  (predicted_y-y[:,None]))   # x.T * (y_hat - y)
        
    print(W, theta)
    predict = value(model)(W=W, feed_dict={'x': aug_X})
    
    plt.scatter(
        X, Y,
    )
    plt.plot(
        X, predict, 'r'
    )
    # plt.axis('off')
    plt.savefig('result.png')
    plt.show()

    
def test_known_model():
    data_size = 10
    x = np.linspace(-7, 7, data_size)
    x = _np.c_[_np.ones(data_size), x]
    # print(x)
    print(grad(known_model, 'x')(feed_dict={'x': x}))

def test_loss():
    data_size = 10
    x = np.linspace(-7, 7, data_size)
    x = _np.c_[_np.ones(data_size), x]
    noise = _np.random.random((data_size))
    theta = _np.array([2, 5])
    y = _np.dot(x, theta) + noise
    predict = value(known_model)(feed_dict={'x': x})
    print(predict)
    v, l = value_and_grad(mse, 'predicted_y')(feed_dict={'predicted_y': predict, 'true_y': y})
    print(v, l)

def test_train():
    data_size = 10
    x = np.linspace(-7, 7, data_size)
    x = _np.c_[_np.ones(data_size), x]
    
    noise = _np.random.random((data_size))
    theta = _np.array([2, 5])
    y = _np.dot(x, theta) + noise

    W = _np.random.random((2, 1))    
    predicted_y, grad_value= value_and_grad(model)(W=W, feed_dict={'x': x})
    loss_grad = grad(mse, 'predicted_y')(feed_dict={'predicted_y': predicted_y, 'true_y': y})
    print(predicted_y.shape, y.shape, x.shape)
    print(np.array_equal(loss_grad, predicted_y - y))
    print(np.array_equal(x.T, grad_value['W']))
    

if __name__ == "__main__":
    # print(grad(test2)(x=1, y=2, z=0))
    # print(grad(test)(z=2, y=-4, x=3, w=-1))
    train_model()