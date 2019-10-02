import numpy as np

from autodiff.core import register_vjp

# print(__name__)


"""
    Unary operation
"""
register_vjp(np.negative, [
    lambda upstream, result, x: -upstream,  # w.r.t. x
])

register_vjp(np.reciprocal, [
    lambda upstream, result, x: upstream * (-1 / x ** 2),  # w.r.t. x
])

register_vjp(np.exp, [
    lambda upstream, result, x: upstream * result,  # w.r.t. x
])

register_vjp(np.log, [
    lambda upstream, result, x: upstream / x,  # w.r.t. x
])

register_vjp(np.sin, [
    lambda upstream, result, x: upstream * np.cos(x),  # w.r.t. x
])

register_vjp(np.cos, [
    lambda upstream, result, x: upstream * -np.sin(x),  # w.r.t. x
])

"""
    Binary operation
"""
register_vjp(np.add, [
    lambda upstream, result, x, y: upstream,  # w.r.t. x
    lambda upstream, result, x, y: upstream,  # w.r.t. y
])

register_vjp(np.subtract, [
    lambda upstream, result, x, y: upstream,  # w.r.t. x
    lambda upstream, result, x, y: -upstream,  # w.r.t. y
])

register_vjp(np.multiply, [
    lambda upstream, result, x, y: upstream * y,  # w.r.t. x
    lambda upstream, result, x, y: upstream * -x,  # w.r.t. y
])

register_vjp(np.divide, [
    lambda upstream, result, x, y: upstream / y,  # w.r.t. x
    lambda upstream, result, x, y: upstream * (-x / y ** 2),  # w.r.t. y
])