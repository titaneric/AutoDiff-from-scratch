#pylint: disable=no-member

import networkx as nx

import autodiff
from autodiff.core import forward_prop
import autodiff.numpy_grad.wrapper as np

def tanh(x):
    return (1.0 - np.exp(-x))  / (1.0 + np.exp(-x)) 

def tanh2(x, **kwargs):
    return np.divide(np.subtract(1, np.exp(np.negative(x))), np.add(1, np.exp(np.negative(x))))

if __name__ == "__main__":
    # print(tanh(0))
    print(forward_prop(tanh2, 0)())
    # grad(tanh2)
    # print(wnp.add(1,2))