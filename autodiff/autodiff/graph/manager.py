import networkx as nx

from ..core import global_vars, GraphInfo, register_graph, pop_graph


class GraphManager:
    def __init__(self):
        pass

    def __enter__(self):
        self.graph: nx.DiGraph = pop_graph()
        self.graph_info = global_vars._graph_info_dict.get(
            self.graph, GraphInfo())
        return self.graph, self.graph_info

    def __exit__(self, *args):
        register_graph(self.graph)
        global_vars._graph_info_dict[self.graph] = self.graph_info


def add_node(graph, node):
    node_index = len(graph.nodes()) + 1
    graph.add_node(node_index, node=node)
    return node_index