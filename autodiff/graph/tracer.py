from functools import wraps

import networkx as nx

from .node import ConstantNode, OperationNode, VariableNode
from .ctx_manager import GraphManager


def constant(array):
    def const_wrapped(*args, **kwargs):
        with GraphManager() as (graph, info):
            node_index = len(graph.nodes())+1
            node = ConstantNode(args)
            graph.add_node(node_index, node=node)

            info.stack.append(node_index)
        # print('const', node_index, const, args)
        return array(*args, **kwargs)
    return const_wrapped

def variable(array):
    def var_wrapped(**kwargs):
        with GraphManager() as (graph, info):
            var_id = tuple(kwargs.keys())
            # print(var_id, args, kwargs)
            if var_id not in info.vars:
                node_index = len(graph.nodes())+1
                node = VariableNode(var_id)
                graph.add_node(node_index, node=node)
                info.vars[var_id] = node_index
            else:
                node_index = info.vars[var_id]
            info.stack.append(node_index)
        # print('var', node_index, kwargs)
        return array(*kwargs.values())
    return var_wrapped

def primitive(func):
    @wraps(func)
    def func_wrapped(*args, **kwargs):
        with GraphManager() as (graph, info):
            result = func(*args, **kwargs)
            node_index = len(graph.nodes())+1
            node = OperationNode(func, args, kwargs, result)
            graph.add_node(node_index, node=node)

            parents = info.stack[-len(args):]
            for parent in parents:
                graph.add_edge(parent, node_index)
                info.stack.pop()

            # print('fun', node_index, func.__name__, args, kwargs, )

            info.stack.append(node_index)
        return result
    return func_wrapped
