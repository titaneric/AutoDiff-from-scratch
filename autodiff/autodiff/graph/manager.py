import networkx as nx

from ..global_vars import get_graph_info, update_graph_info, register_graph, pop_graph


class GraphManager:
    def __init__(self):
        pass

    def __enter__(self):
        self.graph: nx.DiGraph = pop_graph()
        self.graph_info = get_graph_info(self.graph)
        return self.graph, self.graph_info

    def __exit__(self, *args):
        register_graph(self.graph)
        update_graph_info(self.graph, self.graph_info)


def add_node(graph, node):
    node_index = len(graph.nodes()) + 1
    graph.add_node(node_index, node=node)
    return node_index