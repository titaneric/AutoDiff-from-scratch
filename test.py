#pylint: disable=no-member
import warnings

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cbook

import autodiff
from autodiff.core import forward_prop, backward_prop, graph_stack
import autodiff.numpy_grad.wrapper as np

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
def tanh(x):
    return (1.0 - np.exp(-x))  / (1.0 + np.exp(-x)) 

def tanh2(x):
    return np.divide(
        np.subtract(np.const(1), np.exp(np.negative(np.var(x)))),
        np.add(np.const(1), np.exp(np.negative(np.var(x))))
    )

def test(x, y, z, w):
    return np.multiply(
                np.add(np.multiply(np.var(x), np.var(y)),
                    np.maximum(np.var(z), np.var(w))), 
                np.const(2)
            )
if __name__ == "__main__":
    print(forward_prop(test, z=2, y=-4, x=3, w=-1)())
    graph: nx.DiGraph = graph_stack.pop()
    nx.draw(graph)
    plt.show()
    plt.savefig('graph.png', format='PNG')
    # print(list(nx.topological_sort(graph)))
    # print(graph.edges())
    backward_prop(graph)
    for node in nx.topological_sort(graph):
        print(node, graph.nodes[node]['node'].gradient)