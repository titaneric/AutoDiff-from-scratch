import networkx as nx

from autodiff.core import op_stack, graph_stack
from .node import ConstantNode, OperationNode, VariableNode


def constant(const):
    def const_wrapped(*args, **kwargs):
        graph: nx.DiGraph = op_stack.pop()
        stack = graph_stack[graph]

        node_index = len(graph.nodes())+1
        node = ConstantNode(const)
        graph.add_node(node_index)

        stack.append(node_index)
        op_stack.append(graph)
        print('const', node_index, const, args)
        return const(*args, **kwargs)
    return const_wrapped

def variable(var):
    def var_wrapped(*args, **kwargs):
        graph: nx.DiGraph = op_stack.pop()
        stack = graph_stack[graph]

        node_index = len(graph.nodes())+1
        node = VariableNode(var)
        graph.add_node(node_index)

        stack.append(node_index)
        op_stack.append(graph)
        print('var', node_index, var, args)
        return var(*args, **kwargs)
    return var_wrapped

def primitive(func):
    def func_wrapped(*args, **kwargs):
        graph: nx.DiGraph = op_stack.pop()
        stack = graph_stack[graph]

        result = func(*args, **kwargs)
        node_index = len(graph.nodes())+1
        node_info = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'result': result
        }
        graph.add_node(node_index, **node_info)

        parents = stack[-len(args):]
        for parent in parents:
            graph.add_edge(parent, node_index)
            stack.pop()

        print('fun', node_index, func.__name__, args, kwargs, )

        stack.append(node_index)
        op_stack.append(graph)
        return result
    return func_wrapped
