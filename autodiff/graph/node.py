from abc import ABC

class Node(ABC):
    pass

class OperationNode(Node):
    def __init__(self, func, args, kwargs, result):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = result

class VariableNode(Node):
    def __init__(self, var):
        self.var = var

class ConstantNode(Node):
    def __init__(self, constant):
        self.constant = constant