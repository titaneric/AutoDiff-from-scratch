#pylint: disable=no-member
import warnings

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cbook

import autodiff
from autodiff.diff import value_and_grad
import autodiff.numpy_grad.wrapper as np

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

def tanh(x):
    return np.divide(
        np.subtract(np.const(1), np.exp(np.negative(np.var(x)))),
        np.add(np.const(1), np.exp(np.negative(np.var(x))))
    )

def test(x, y, z, w):
    return np.multiply(
                np.add(np.multiply(np.var(x=x), np.var(y=y)),
                    np.maximum(np.var(z=z), np.var(w=w))), 
                np.const(2)
            )

def test2(x, y, z):
    return np.multiply(
        np.add(np.var(x=x), np.var(y=y)),
        np.maximum(np.var(y=y),np.var(z=z))
    )


def test3(x):
    return np.power(np.var(x=x), np.const(3))

if __name__ == "__main__":
    # print(grad(test2)(x=1, y=2, z=0))
    # print(grad(test)(z=2, y=-4, x=3, w=-1))
    x_list = np.linspace(-7, 7, 200)
    y_list, dy_list = value_and_grad(test3, 'x')(x=x_list)

    plt.plot(
        x_list, y_list,
        x_list, dy_list
    )
    # plt.axis('off')
    plt.savefig('x**3.png')
    plt.show()
