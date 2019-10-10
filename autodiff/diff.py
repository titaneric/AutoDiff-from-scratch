from autodiff.core import forward_prop, backward_prop, graph_stack

def value(func):
    def valueWrapped(**kwargs):
        # print(kwargs)
        forward_func = forward_prop(func, **kwargs)
        return forward_func()
    
    return valueWrapped

def grad(func, wrt=None):
    def gradVal(**kwargs):
        # print(kwargs)
        forward_func = forward_prop(func, **kwargs)
        _ = forward_func()
        graph = graph_stack.pop()
        graph, grad = backward_prop(graph)
        return grad if wrt is None else grad[wrt]

    return gradVal

def value_and_grad(func, wrt=None):
    def gradVal(**kwargs):
        # print(kwargs)
        forward_func = forward_prop(func, **kwargs)
        end_value = forward_func()
        graph = graph_stack.pop()
        graph, grad = backward_prop(graph)
        return (end_value, grad) if wrt is None else (end_value, grad[wrt])

    return gradVal

