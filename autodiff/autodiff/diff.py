import numpy as onp

from .core import forward_prop, backward_prop, global_vars


def value(func):
    def valueWrapped(**kwargs):
        forward_func = forward_prop(func, **kwargs)
        return forward_func()

    return valueWrapped


def grad(func, wrt=None, upstream=None):
    def gradVal(**kwargs):
        forward_func = forward_prop(func, **kwargs)
        end_value = forward_func()
        g = onp.ones_like(end_value) if upstream is None else upstream
        grad = backward_prop(g)
        return grad if wrt is None else grad[wrt]

    return gradVal


def value_and_grad(func, wrt=None):
    def gradVal(**kwargs):
        forward_func = forward_prop(func, **kwargs)
        end_value = forward_func()
        grad = backward_prop(onp.ones_like(end_value))
        return (end_value, grad) if wrt is None else (end_value, grad[wrt])

    return gradVal
