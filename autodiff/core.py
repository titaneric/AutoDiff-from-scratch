from collections import defaultdict, namedtuple

import networkx as nx
import numpy as onp

from autodiff.graph.node import OperationNode, VariableNode, PlaceholderNode

primitive_vhp = defaultdict(dict)

graph_stack = []
GraphInfo = namedtuple('GraphInfo', 'stack, vars, places')
graph_info_dict = defaultdict(GraphInfo)

def is_wrt(node):
    return type(node) in [VariableNode, PlaceholderNode]

def forward_prop(func, **assignd):
    def forward_wrap(*args, **kwargs):
        graph = nx.DiGraph()
        graph_stack.append(graph)
        return func(*args, **assignd)

    return forward_wrap

def backward_prop(graph: nx.DiGraph):
    graph.nodes[len(graph.nodes())]['node'].gradient = onp.array(1)

    gradient_dict = {}
    for node in reversed(list(nx.topological_sort(graph))):
        child_node: OperationNode = graph.nodes[node]['node']
        if isinstance(child_node, OperationNode):
            func, args, kwargs, result, arg_num = child_node.recipe
            upstream = child_node.gradient
            # print(func.__name__, upstream)
            for i, parent in zip(range(arg_num), graph.predecessors(node)):
                vhp = primitive_vhp[func.__name__][i]
                downstream = vhp(upstream, result, *args)
                graph.nodes[parent]['node'].gradient += downstream
                # tmp = graph.nodes[parent]['node']
                # print("\t:", "" if not is_wrt(tmp) else tmp.var, downstream)
        elif is_wrt(child_node):
            gradient_dict[child_node.var] = child_node.gradient
    
    return graph, gradient_dict


def register_vjp(func, vhp_list):
    for i, downstream in enumerate(vhp_list):
        primitive_vhp[func.__name__][i] = downstream
