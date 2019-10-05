from collections import defaultdict, namedtuple

import networkx as nx

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

def backward_prop(upstream, graph: nx.DiGraph):
    for node in reversed(list(nx.topological_sort(graph))):
        for parent in graph.predecessors:
            pass

def register_vjp(func, vhp_list):
    for i, downstream in enumerate(vhp_list):
        primitive_vhp[func][i] = downstream

if __name__ == "__main__":
    pass