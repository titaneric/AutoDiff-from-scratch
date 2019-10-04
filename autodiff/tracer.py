from .core import graph_queue

import networkx as nx

def primitive(func):
    def func_wrapped(*args, **kwargs):
        print(func.__name__, args, kwargs)
        graph = graph_queue.get()
        print(id(graph), graph_queue.qsize())
        graph_queue.put(graph)
        # sub_graph = nx.DiGraph()
        # sub_graph.add_node(func)
        # sub_graph
        return func(*args, **kwargs)
    return func_wrapped