from functools import wraps

import networkx as nx

from .node import ConstantNode, OperationNode, VariableNode, PlaceholderNode
from .manager import GraphManager, add_node

#TODO refactor this part
def constant(array):
    def const_wrapped(content):
        with GraphManager() as (graph, info):
            if not info.forwarded:
                node = ConstantNode(content)
                node_index = add_node(graph, node)
                info.stack.append(node_index)
        # print('const', node_index, content)
        return content if isinstance(content, tuple) else array(content)

    return const_wrapped


def variable(array):
    def var_wrapped(content):
        with GraphManager() as (graph, info):
            if not info.forwarded:
                var_id = id(content)
                if var_id not in info.vars:
                    node = VariableNode(var_id, content)
                    node_index = add_node(graph, node)
                    info.vars[var_id] = node_index
                else:
                    node_index = info.vars[var_id]
                info.stack.append(node_index)
        # print('var', node_index, content.shape)
        return array(content)

    return var_wrapped


def placeholder(array):
    def place_wrapped(content):
        with GraphManager() as (graph, info):
            if not info.forwarded:
                place_id = id(content)
                if place_id not in info.places:
                    node = PlaceholderNode(place_id)
                    node_index = add_node(graph, node)
                    info.places[place_id] = node_index
                else:
                    node_index = info.places[place_id]
                info.stack.append(node_index)
        # print('place', node_index, content.shape)
        return array(content)

    return place_wrapped


def primitive(func):
    @wraps(func)
    def func_wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        with GraphManager() as (graph, info):
            if not info.forwarded:
                node = OperationNode(func, args, kwargs, result)
                node_index = add_node(graph, node)

                parents = info.stack[-len(args):]
                for parent in parents:
                    graph.add_edge(parent, node_index)
                    info.stack.pop()

                # print('fun', node_index, func.__name__)

                info.stack.append(node_index)
        return result

    return func_wrapped
