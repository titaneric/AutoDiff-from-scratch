import networkx as nx

from autodiff.core import graph_stack, graph_info_dict, GraphInfo


class GraphManager:
    def __init__(self):
        pass

    def __enter__(self):
        self.graph: nx.DiGraph = graph_stack.pop()
        self.graph_info = graph_info_dict.get(self.graph, GraphInfo([], dict(), dict()))
        return self.graph, self.graph_info

    def __exit__(self, *args):
        graph_stack.append(self.graph)
        graph_info_dict[self.graph] = self.graph_info

def add_node(graph, node):
    node_index = len(graph.nodes())+1
    graph.add_node(node_index, node=node)
    return node_index