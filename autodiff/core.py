from collections import defaultdict
from queue import Queue
import inspect

import numpy as np
import networkx as nx

primitive_vhp = defaultdict(dict)
graph_queue = Queue()

def build_graph(func, x):
    print(x)
    return

def forward_prop(func, *x):
    def forward_wrap(*args, **kwargs):
        graph = nx.DiGraph()
        print(id(graph))
        graph_queue.put(graph)
        return func(*x, *args, **kwargs)
    return forward_wrap

def backward_prop(upstream, graph: nx.DiGraph):
    for node in reversed(list(nx.topological_sort(graph))):
        for parent in graph.predecessors:
            pass

def register_vjp(func, vhp_list):
    for i, downstream in enumerate(vhp_list):
        primitive_vhp[func][i] = downstream

if __name__ == "__main__":
    pass