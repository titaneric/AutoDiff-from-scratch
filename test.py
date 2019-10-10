#pylint: disable=no-member
import warnings

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cbook
import numpy as _np

import autodiff
from autodiff.diff import value_and_grad, value, grad
import autodiff.numpy_grad.wrapper as np

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

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

def model(W, b, feed_dict={}):
    return np.add(np.multiply(np.Variable(W=W), np.Placeholder(x=feed_dict['x'])), np.Variable(b=b))

def known_model(feed_dict={}):
    return np.add(np.multiply(np.Constant(2), np.Placeholder(x=feed_dict['x'])), np.Constant(1))

def mse(feed_dict={}):
    return np.multiply(np.dot(np.Placeholder(predicted_y=feed_dict['predicted_y']), np.Placeholder(true_y=feed_dict['true_y'])), np.Constant(1/2))


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
    data_size = 10
    x = np.linspace(-7, 7, data_size)
    # noise = _np.random.randint(-3, 3, (data_size))
    noise = _np.random.random((data_size))
    y = 2 * x + 5 + noise


    lr = 0.001
    epoch = 10
    W = _np.random.random()
    b = _np.random.random()

    for _ in range(epoch):
        _, grad = value_and_grad(mse)(W=W, b=b, feed_dict={'x': x, 'true_y': y})
        print(grad)
        W -= lr * grad['W']
        b -= lr * grad['b']
    

    predict, _ = value_and_grad(mse)(W=W, b=b, feed_dict={'x': x, 'true_y': y})
    
    plt.scatter(
        x, y,
    )
    plt.plot(
        x, predict
    )
    # plt.axis('off')
    plt.savefig('result.png')
    plt.show()

    
def test_known_model():
    data_size = 10
    x = np.linspace(-7, 7, data_size)
    print(value(known_model)(feed_dict={'x': x}))

def test_loss():
    data_size = 10
    x = np.linspace(-7, 7, data_size)
    noise = _np.random.random((data_size))
    y = 2 * x + 5 + noise
    predict = value(known_model)(feed_dict={'x': x})
    print(predict)
    l = grad(mse, 'predicted_y')(feed_dict={'predicted_y': predict, 'true_y': y})
    print(l)

def test_train():
    data_size = 10
    x = np.linspace(-7, 7, data_size)
    # noise = _np.random.randint(-3, 3, (data_size))
    noise = _np.random.random((data_size))
    y = 2 * x + 5 + noise


    lr = 0.001
    epoch = 10
    W = _np.random.random()
    b = _np.random.random()
    print(W, b)
    predicted_y= value(model)(W=W, b=b, feed_dict={'x': x})
    loss_grad = grad(mse, 'predicted_y')(feed_dict={'predicted_y': predicted_y, 'true_y': y})
    print(loss_grad)

if __name__ == "__main__":
    # print(grad(test2)(x=1, y=2, z=0))
    # print(grad(test)(z=2, y=-4, x=3, w=-1))
    power_demo()