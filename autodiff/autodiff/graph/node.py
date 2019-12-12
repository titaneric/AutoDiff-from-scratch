only_take_lhs_grad = [
    "reshape",
    "__getitem__",
]

class Node:
    def __init__(self):
        self.gradient = 0


class OperationNode(Node):
    def __init__(self, func, args, kwargs, result):
        super().__init__()
        self.recipe = (func, args, kwargs, result,
                       len(args) if func.__name__ not in only_take_lhs_grad else 1)


class VariableNode(Node):
    def __init__(self, var_id, content):
        super().__init__()
        self.var = var_id
        self.content = content


class PlaceholderNode(Node):
    def __init__(self, var):
        super().__init__()
        self.var = var


class ConstantNode(Node):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant