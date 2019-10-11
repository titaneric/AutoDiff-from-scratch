#pylint: disable=no-member
import numpy as onp

from autodiff.numpy_grad import wrapper as wnp
from autodiff.core import register_vjp

# print(__name__)


"""
    Unary operation
"""
register_vjp(wnp.negative, [
    lambda upstream, result, x: -upstream,  # w.r.t. x
])

register_vjp(wnp.reciprocal, [
    lambda upstream, result, x: upstream * (-1 / x ** 2),  # w.r.t. x
])

register_vjp(wnp.exp, [
    lambda upstream, result, x: upstream * result,  # w.r.t. x
])

register_vjp(wnp.log, [
    lambda upstream, result, x: upstream / x,  # w.r.t. x
])

register_vjp(wnp.sin, [
    lambda upstream, result, x: upstream * wnp.cos(x),  # w.r.t. x
])

register_vjp(wnp.cos, [
    lambda upstream, result, x: upstream * -wnp.sin(x),  # w.r.t. x
])

"""
    Binary operation
"""
register_vjp(wnp.add, [
    lambda upstream, result, x, y: unbroadcast(x, upstream),  # w.r.t. x
    lambda upstream, result, x, y: unbroadcast(y, upstream),  # w.r.t. y
])

register_vjp(wnp.subtract, [
    lambda upstream, result, x, y: unbroadcast(x, upstream, other=y),  # w.r.t. x
    lambda upstream, result, x, y: unbroadcast(y, -upstream, other=x),  # w.r.t. y
])

register_vjp(wnp.multiply, [
    lambda upstream, result, x, y: unbroadcast(x, upstream * y),  # w.r.t. x
    lambda upstream, result, x, y: unbroadcast(y, upstream * x),  # w.r.t. y
])

register_vjp(wnp.true_divide, [
    lambda upstream, result, x, y: unbroadcast(x, upstream / y),  # w.r.t. x
    lambda upstream, result, x, y: unbroadcast(y, upstream * (-x / y ** 2)),  # w.r.t. y
])

register_vjp(wnp.maximum, [
    lambda upstream, result, x, y:upstream * balanced_eq(x, result, y),  # w.r.t. x
    lambda upstream, result, x, y:upstream * balanced_eq(y, result, x),  # w.r.t. y
])

register_vjp(wnp.minimum, [
    lambda upstream, result, x, y: upstream * balanced_eq(x, result, y),  # w.r.t. x
    lambda upstream, result, x, y: upstream * balanced_eq(y, result, x),  # w.r.t. y
])

register_vjp(wnp.power, [
    lambda upstream, result, x, y: unbroadcast(x, upstream * (y * x ** (y - 1))),  # w.r.t. x
    lambda upstream, result, x, y: unbroadcast(y, upstream * (result * onp.log(replace_zero(x, 1.)))),  # w.r.t. y
])

# shamelessly taken from autograd
def replace_zero(x, val):
    return onp.where(x, x, val)

def unbroadcast(target, g, broadcast_idx=0, other=None):
    """
        Let downstream have the same shape as target
    """
    # if onp.ndim(g) == 2:
    #     g = g.diagonal()[:,None]
    # print("-" * 10)
    # print(onp.ndim(g) > onp.ndim(target), g.shape, target.shape)
    # print(other)
    # print(target)
    # print("Before", g)
    while onp.ndim(g) > onp.ndim(target):
        g = onp.sum(g, axis=broadcast_idx)
        # print("Sum", g)
    
    if onp.ndim(g) == onp.ndim(target):
        for axis, size in enumerate(onp.shape(target)):
            if size == 1:
                g = onp.sum(g, axis=axis, keepdims=True)
    # print("After", g)
    # print("-" * 10)
    return g

def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y)) 

"""
    Matrix calculation
"""

def dot_vjp_first(upstream, result, x, y):
    if max(wnp.ndim(x), wnp.ndim(y)) > 2:
        raise NotImplementedError("Not support dimension currently!")

    if wnp.ndim(x) == 0:
        return wnp.sum(upstream * y)
    
    if wnp.ndim(x) == 1 and wnp.ndim(y) == 1:
        return upstream * y
    
    if wnp.ndim(x) == 2 and wnp.ndim(y) == 1:
        return upstream[:, None] * y if wnp.ndim(upstream) == 1 else upstream * y
    if wnp.ndim(x) == 1 and wnp.ndim(y) == 2:
        return onp.dot(y, upstream)
    
    return onp.dot(upstream, y.T)
    
    

def dot_vjp_second(upstream, result, x, y):
    if max(wnp.ndim(x), wnp.ndim(y)) > 2:
        raise NotImplementedError("Not support dimension currently!")
    
    if wnp.ndim(x) == 0:
        return wnp.sum(upstream * x)
    
    if wnp.ndim(x) == 1 and wnp.ndim(y) == 1:
        return upstream * x
    
    if wnp.ndim(x) == 2 and wnp.ndim(y) == 1:
        return onp.dot(upstream, y)
    
    if wnp.ndim(x) == 1 and wnp.ndim(y) == 2:
        return x[:, None] * upstream
    
    return onp.dot(x.T, upstream)

register_vjp(wnp.dot, [
    dot_vjp_first ,  # w.r.t. x
    dot_vjp_second ,  # w.r.t. y
])
