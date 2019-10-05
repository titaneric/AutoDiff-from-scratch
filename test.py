#pylint: disable=no-member

import networkx as nx
import matplotlib.pyplot as plt

import autodiff
from autodiff.core import forward_prop, graph_stack
import autodiff.numpy_grad.wrapper as np

def tanh(x):
    return (1.0 - np.exp(-x))  / (1.0 + np.exp(-x)) 

def tanh2(x):
    return np.divide(
        np.subtract(np.const(1), np.exp(np.negative(np.var(x)))),
        np.add(np.const(1), np.exp(np.negative(np.var(x))))
    )

if __name__ == "__main__":
    # print(tanh(0))
    print(forward_prop(tanh2, 0)())
    graph: nx.DiGraph = graph_stack.pop()
    nx.draw(graph)
    plt.show()
    plt.savefig('graph.png', format='PNG')
    # print(graph.nodes[1])
    # grad(tanh2)
    # print(graph.edges())
    # print(wnp.add(1,2))