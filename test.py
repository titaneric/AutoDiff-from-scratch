#pylint: disable=no-member
import warnings

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cbook
import numpy as _np

import autodiff
from autodiff.autodiff.diff import value_and_grad, value, grad
import autodiff.autodiff.numpy_grad.wrapper as np

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

def power_demo():
    x_list = np.linspace(-7, 7, 200)
    # x_list = 5
    y_list, dy_list = value_and_grad(test3, 'x')(x=x_list)
    # print(y_list, dy_list)

    plt.plot(
        x_list, y_list,
        x_list, dy_list
    )
    # plt.axis('off')
    plt.savefig('x**3.png')
    plt.show()

if __name__ == "__main__":
    power_demo()