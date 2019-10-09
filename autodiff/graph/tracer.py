from functools import wraps

import networkx as nx

from autodiff.graph.node import ConstantNode, OperationNode, VariableNode, PlaceholderNode
from autodiff.graph.manager import GraphManager, add_node


def constant(array):
    def const_wrapped(*args, **kwargs):
        with GraphManager() as (graph, info):
            node = ConstantNode(args)
            node_index = add_node(graph, node)
            info.stack.append(node_index)
        # print('const', node_index, const, args)
        return array(*args, **kwargs)
    return const_wrapped

def variable(array):
    def var_wrapped(**kwargs):
        with GraphManager() as (graph, info):
            var_id = "".join(kwargs.keys())
            if var_id not in info.vars:
                node = VariableNode(var_id)
                node_index = add_node(graph, node)
                info.vars[var_id] = node_index
            else:
                node_index = info.vars[var_id]
            info.stack.append(node_index)
        # print('var', node_index, kwargs)
        return array(*kwargs.values())
    return var_wrapped

def placeholder(array):
    def place_wrapped(**kwargs):
        with GraphManager() as (graph, info):
            place_id = "".join(kwargs.keys())
            if place_id not in info.places:
                node = PlaceholderNode(place_id)
                node_index = add_node(graph, node)
                info.places[place_id] = node_index
            else:
                node_index = info.places[place_id]
            info.stack.append(node_index)
        # print('var', node_index, kwargs)
        return array(*kwargs.values())
    return place_wrapped

def primitive(func):
    @wraps(func)
    def func_wrapped(*args, **kwargs):
        with GraphManager() as (graph, info):
            result = func(*args, **kwargs)
            node = OperationNode(func, args, kwargs, result)
            node_index = add_node(graph, node)

            parents = info.stack[-len(args):]
            for parent in parents:
                graph.add_edge(parent, node_index)
                info.stack.pop()

            # print('fun', node_index, func.__name__, args, kwargs, )

            info.stack.append(node_index)
        return result
    return func_wrapped
