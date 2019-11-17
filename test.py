#pylint: disable=no-member
import warnings

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cbook
import numpy as _np

import autodiff as ad
from autodiff import value_and_grad, value, grad

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def tanh(x):
    return ad.divide(
        ad.subtract(ad.Constant(1), ad.exp(ad.negative(ad.Variable(x)))),
        ad.add(ad.Constant(1), ad.exp(ad.negative(ad.Variable(x)))))


def test(x, y, z, w):
    return ad.multiply(
        ad.add(ad.multiply(ad.Variable(x=x), ad.Variable(y=y)),
               ad.maximum(ad.Variable(z=z), ad.Variable(w=w))), ad.Constant(2))


def test2(x, y, z):
    return ad.multiply(ad.add(ad.Variable(x=x), ad.Variable(y=y)),
                       ad.maximum(ad.Variable(y=y), ad.Variable(z=z)))


def test3(x):
    return ad.power(ad.Variable(x=x), ad.Constant(3))


def test4(x):
    return ad.multiply(ad.power(ad.Variable(x=x), ad.Constant(2)),
                       ad.Constant(1 / 2))


def power_demo():
    x_list = ad.linspace(-7, 7, 200)
    # x_list = 5
    y_list, dy_list = value_and_grad(test3, 'x')(x=x_list)
    # print(y_list, dy_list)

    plt.plot(x_list, y_list, x_list, dy_list)
    # plt.axis('off')
    plt.savefig('x**3.png')
    plt.show()


if __name__ == "__main__":
    power_demo()