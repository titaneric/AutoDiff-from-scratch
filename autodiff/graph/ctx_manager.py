import networkx as nx

from autodiff.core import op_stack, graph_stack

class Graph:
    def __init__(self):
        pass

    def __enter__(self):
        self.graph: nx.DiGraph = op_stack.pop()
        self.stack = graph_stack[self.graph]
        return self.graph, self.stack

    def __exit__(self, *args):
        op_stack.append(self.graph)        
        pass