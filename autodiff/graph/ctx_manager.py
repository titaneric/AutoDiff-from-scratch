import networkx as nx

from autodiff.core import graph_stack, graph_info_dict, GraphInfo


class GraphManager:
    def __init__(self):
        pass

    def __enter__(self):
        self.graph: nx.DiGraph = graph_stack.pop()
        graph_info = graph_info_dict.get(self.graph, GraphInfo([], {}))
        return self.graph, graph_info.stack, graph_info.vars

    def __exit__(self, *args):
        graph_stack.append(self.graph)        
        pass