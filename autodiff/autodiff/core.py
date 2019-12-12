from collections import defaultdict, namedtuple

import networkx as nx
import numpy as onp

from .graph.node import OperationNode, VariableNode, PlaceholderNode

primitive_vhp = defaultdict(dict)


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


def is_wrt(node):
    return type(node) in [VariableNode, PlaceholderNode]


def forward_prop(func, **assignd):
    def forward_wrap(*args, **kwargs):
        global global_vars
        register_graph(nx.DiGraph())
        return func(*args, **assignd)

    return forward_wrap


def backward_prop(upstream):
    graph = pop_graph()
    graph.nodes[len(graph.nodes())]['node'].gradient = upstream
    # print("Set gradient to ", upstream, len(graph.nodes()))
    gradient_dict = {}
    for node in reversed(list(nx.topological_sort(graph))):
        child_node: OperationNode = graph.nodes[node]['node']
        if isinstance(child_node, OperationNode):
            func, args, kwargs, result, arg_num = child_node.recipe
            upstream = child_node.gradient
            # print(func.__name__, node, upstream, args, arg_num)

            for i, parent in zip(range(arg_num), graph.predecessors(node)):
                vhp = primitive_vhp[func.__name__][i]
                downstream = vhp(upstream, result, *args, **kwargs)
                # print(i, "downstream size is", downstream.shape)
                graph.nodes[parent]['node'].gradient += downstream

        elif is_wrt(child_node):
            gradient_dict[child_node.var] = child_node.gradient

    return gradient_dict


def zero_grad(graph):
    for node_index in graph.nodes:
        graph.nodes[node_index]['node'].gradient = 0


def register_vjp(func, vhp_list):
    for i, downstream in enumerate(vhp_list):
        primitive_vhp[func.__name__][i] = downstream


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
