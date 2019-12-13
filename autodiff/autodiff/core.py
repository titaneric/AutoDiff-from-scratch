from collections import defaultdict

import networkx as nx

from .graph.node import OperationNode, VariableNode, PlaceholderNode
from .global_vars import register_graph, pop_graph

primitive_vhp = defaultdict(dict)


def is_wrt(node):
    return type(node) in [VariableNode, PlaceholderNode]


def forward_prop(func, **assignd):
    def forward_wrap(*args, **kwargs):
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


# TODO restructure this part
def zero_grad(graph):
    for node_index in graph.nodes:
        graph.nodes[node_index]['node'].gradient = 0


def register_vjp(func, vhp_list):
    for i, downstream in enumerate(vhp_list):
        primitive_vhp[func.__name__][i] = downstream
