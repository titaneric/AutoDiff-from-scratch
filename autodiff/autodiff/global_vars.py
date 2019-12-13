from collections import defaultdict


class GraphInfo:
    def __init__(self):
        self.stack = []
        self.vars = dict()
        self.places = dict()
        self.forwarded = False


class var:
    pass


global_vars = var()
global_vars._graph_stack = []
global_vars._graph_info_dict = defaultdict(GraphInfo)


def register_graph(graph):
    global_vars._graph_stack.append(graph)


def pop_graph():
    return global_vars._graph_stack.pop()


def set_forwarded(graph):
    global_vars._graph_info_dict[graph].forwarded = True


def set_parameters(parameters, graph):
    for array_id, node_index in global_vars._graph_info_dict[graph].vars.items(
    ):
        parameters.append({
            "array_id": array_id,
            "variables": graph.nodes[node_index]['node'].content
        })


def get_graph_info(graph):
    return global_vars._graph_info_dict.get(graph, GraphInfo())


def update_graph_info(graph, graph_info):
    global_vars._graph_info_dict[graph] = graph_info
