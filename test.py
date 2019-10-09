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
        np.subtract(np.constant(1), np.exp(np.negative(np.variable(x)))),
        np.add(np.constant(1), np.exp(np.negative(np.variable(x))))
    )

def test(x, y, z, w):
    return np.multiply(
                np.add(np.multiply(np.variable(x=x), np.variable(y=y)),
                    np.maximum(np.variable(z=z), np.variable(w=w))), 
                np.constant(2)
            )

def test2(x, y, z):
    return np.multiply(
        np.add(np.variable(x=x), np.variable(y=y)),
        np.maximum(np.variable(y=y),np.variable(z=z))
    )


def test3(x):
    return np.power(np.variable(x=x), np.constant(3))

# def model(W, x, b):
#     return np.multiply(np.variable(W), np.place)

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
