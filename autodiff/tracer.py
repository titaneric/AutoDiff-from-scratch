import networkx as nx

def primitive(func):
    def f_wrapped(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)
    return f_wrapped