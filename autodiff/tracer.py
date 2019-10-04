import networkx as nx

def primitive(func):
    def f_wrapped(*args, **kwargs):
        print(func.__name__, args, kwargs)#, id(kwargs['graph']))
        # sub_graph = nx.DiGraph()
        # sub_graph.add_node(func)
        # sub_graph
        return func(*args, **kwargs)
    return f_wrapped