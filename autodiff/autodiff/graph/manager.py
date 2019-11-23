import networkx as nx

from ..core import global_vars, graph_info_dict, GraphInfo


class GraphManager:
    def __init__(self):
        pass

    def __enter__(self):
        self.graph: nx.DiGraph = global_vars._default_graph
        self.graph_info = graph_info_dict.get(self.graph,
                                              GraphInfo([], dict(), dict()))
        return self.graph, self.graph_info

    def __exit__(self, *args):
        _default_graph = self.graph
        graph_info_dict[_default_graph] = self.graph_info


def add_node(graph, node):
    node_index = len(graph.nodes()) + 1
    graph.add_node(node_index, node=node)
    return node_index