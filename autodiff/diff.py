from autodiff.core import forward_prop, backward_prop, graph_stack

def grad(func, wrt=None):
    def gradVal(**kwargs):
        # print(kwargs)
        forward_func = forward_prop(func, **kwargs)
        end_value = forward_func()
        graph = graph_stack.pop()
        graph, grad = backward_prop(graph)
        return (end_value, grad) if wrt is None else (end_value, grad[wrt])

    return gradVal

