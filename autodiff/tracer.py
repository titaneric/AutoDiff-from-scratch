import networkx as nx

def primitive(func):
    def f_wrapped(*args):
        print(1)
        return func(*args)
    return f_wrapped