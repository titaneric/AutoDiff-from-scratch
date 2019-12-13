#pylint: disable=no-member
import numpy as onp

from ..numpy_grad import wrapper as wnp
from ..core import register_vjp

# print(__name__)
"""
    Unary operation
"""
register_vjp(
    wnp.negative,
    [
        lambda upstream, result, x: -upstream,  # w.r.t. x
    ])

register_vjp(
    wnp.reciprocal,
    [
        lambda upstream, result, x: upstream * (-1 / x**2),  # w.r.t. x
    ])

register_vjp(
    wnp.exp,
    [
        lambda upstream, result, x: upstream * result,  # w.r.t. x
    ])

register_vjp(
    wnp.log,
    [
        lambda upstream, result, x: upstream / x,  # w.r.t. x
    ])

register_vjp(
    wnp.sin,
    [
        lambda upstream, result, x: upstream * onp.cos(x),  # w.r.t. x
    ])

register_vjp(
    wnp.cos,
    [
        lambda upstream, result, x: upstream * -onp.sin(x),  # w.r.t. x
    ])
"""
    Binary operation
"""
register_vjp(
    wnp.add,
    [
        lambda upstream, result, x, y: unbroadcast(x, upstream),  # w.r.t. x
        lambda upstream, result, x, y: unbroadcast(y, upstream),  # w.r.t. y
    ])

register_vjp(
    wnp.subtract,
    [
        lambda upstream, result, x, y: unbroadcast(x, upstream, other=y
                                                   ),  # w.r.t. x
        lambda upstream, result, x, y: unbroadcast(y, -upstream, other=x
                                                   ),  # w.r.t. y
    ])

register_vjp(
    wnp.multiply,
    [
        lambda upstream, result, x, y: unbroadcast(x, upstream * y
                                                   ),  # w.r.t. x
        lambda upstream, result, x, y: unbroadcast(y, upstream * x
                                                   ),  # w.r.t. y
    ])

register_vjp(
    wnp.true_divide,
    [
        lambda upstream, result, x, y: unbroadcast(x, upstream / y
                                                   ),  # w.r.t. x
        lambda upstream, result, x, y: unbroadcast(y, upstream *
                                                   (-x / y**2)),  # w.r.t. y
    ])

register_vjp(
    wnp.maximum,
    [
        lambda upstream, result, x, y: upstream * balanced_eq(x, result, y
                                                              ),  # w.r.t. x
        lambda upstream, result, x, y: upstream * balanced_eq(y, result, x
                                                              ),  # w.r.t. y
    ])

register_vjp(
    wnp.minimum,
    [
        lambda upstream, result, x, y: upstream * balanced_eq(x, result, y
                                                              ),  # w.r.t. x
        lambda upstream, result, x, y: upstream * balanced_eq(y, result, x
                                                              ),  # w.r.t. y
    ])

register_vjp(
    wnp.power,
    [
        lambda upstream, result, x, y: unbroadcast(
            x, upstream * (y * x**(y - 1))),  # w.r.t. x
        lambda upstream, result, x, y: unbroadcast(
            y, upstream * (result * onp.log(replace_zero(x, 1.)))),  # w.r.t. y
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
    # print(g, target)
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
    # print("first: ", upstream, result.shape, x.shape, y.shape)

    if not (onp.ndim(x) == onp.ndim(y) == 2):
        raise NotImplementedError("Only care about MM or MV product!")

    # Take the derivative of output respect to x (input)
    downstream = onp.dot(upstream, y.T)

    assert downstream.shape == x.shape

    return downstream


def dot_vjp_second(upstream, result, x, y):
    # print("second: ", upstream, result.shape, x.shape, y.shape)

    if not (onp.ndim(x) == onp.ndim(y) == 2):
        raise NotImplementedError("Only care about MM or MV product!")

    # Take the derivative of output respect to y (weight)
    downstream = onp.dot(x.T, upstream)

    assert downstream.shape == y.shape

    return downstream


register_vjp(
    wnp.dot,
    [
        dot_vjp_first,  # w.r.t. x
        dot_vjp_second,  # w.r.t. y
    ])

# Special operator

register_vjp(wnp.reshape, [
    lambda upstream, result, x, shape, order=None: onp.reshape(
        upstream, onp.shape(x), order=order)
])


def sum_vjp(upstream, result, x, axis=1, keepdims=False, dtype=None):
    shape = onp.shape(x)

    if not shape:
        return upstream

    # print(result, x, axis)
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = onp.array(shape)
    new_shape[axis] = 1
    # print(onp.reshape(upstream, new_shape))
    return onp.reshape(upstream, new_shape) + onp.zeros(shape, dtype=dtype)


register_vjp(wnp.sum, [sum_vjp])


def getitem_vjp(upstream, result, x, index):
    # print(x, index, upstream)
    onp.add.at(x, index, upstream)
    return x

register_vjp(wnp.__getitem__, [getitem_vjp])

