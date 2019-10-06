from collections import defaultdict, namedtuple

import networkx as nx

from autodiff.graph.node import OperationNode, VariableNode

primitive_vhp = defaultdict(dict)

graph_stack = []
GraphInfo = namedtuple('GraphInfo', 'stack, vars')
graph_info_dict = defaultdict(GraphInfo)

def forward_prop(func, **assignd):
    def forward_wrap(*args, **kwargs):
        graph = nx.DiGraph()
        graph_stack.append(graph)
        return func(*args, **assignd)

    return forward_wrap

def backward_prop(graph: nx.DiGraph):
    graph.nodes[len(graph.nodes())]['node'].gradient = 1

    gradient_dict = {}
    for node in reversed(list(nx.topological_sort(graph))):
        child_node: OperationNode = graph.nodes[node]['node']
        if isinstance(child_node, OperationNode):
            func, args, kwargs, result, arg_num = child_node.recipe
            upstream = child_node.gradient
            # print(node, func.__name__, upstream)
            for i, parent in zip(range(arg_num), graph.predecessors(node)):
                parent_node: VariableNode = graph.nodes[parent]['node']
                if isinstance(parent_node, VariableNode):
                    vhp = primitive_vhp[func.__name__][i]
                    downstream = vhp(upstream, result, *args)
                    graph.nodes[parent]['node'].gradient += downstream
                    # print(parent,":", downstream)
        elif isinstance(child_node, VariableNode):
            gradient_dict[child_node.var] = child_node.gradient
    
    return graph, gradient_dict


def register_vjp(func, vhp_list):
    for i, downstream in enumerate(vhp_list):
        primitive_vhp[func.__name__][i] = downstream
