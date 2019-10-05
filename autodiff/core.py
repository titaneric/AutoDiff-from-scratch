from collections import defaultdict, namedtuple

import networkx as nx

from autodiff.graph.node import OperationNode

primitive_vhp = defaultdict(dict)

graph_stack = []
GraphInfo = namedtuple('GraphInfo', 'stack, vars')
graph_info_dict = defaultdict(GraphInfo)

def forward_prop(func, *x):
    def forward_wrap(*args, **kwargs):
        graph = nx.DiGraph()
        print(id(graph), x)
        graph_stack.append(graph)
        return func(*x, *args, **kwargs)

    return forward_wrap

def backward_prop(graph: nx.DiGraph):
    graph.nodes[len(graph.nodes())]['node'].gradient = 1
    for node in reversed(list(nx.topological_sort(graph))):
        child_node: OperationNode = graph.nodes[node]['node']
        if isinstance(child_node, OperationNode):
            func, args, kwargs, result, arg_num = child_node.recipe
            upstream = child_node.gradient
            for i, parent in zip(range(arg_num), graph.predecessors(node)):
                vhp = primitive_vhp[func.__name__][i]
                downstream = vhp(upstream, result, *args)
                graph.nodes[parent]['node'].gradient += downstream


def register_vjp(func, vhp_list):
    for i, downstream in enumerate(vhp_list):
        primitive_vhp[func.__name__][i] = downstream

if __name__ == "__main__":
    pass