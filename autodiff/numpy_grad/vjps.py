#pylint: disable=no-member
from math import log

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
    lambda upstream, result, x, y: upstream,  # w.r.t. x
    lambda upstream, result, x, y: upstream,  # w.r.t. y
])

register_vjp(wnp.subtract, [
    lambda upstream, result, x, y: upstream,  # w.r.t. x
    lambda upstream, result, x, y: -upstream,  # w.r.t. y
])

register_vjp(wnp.multiply, [
    lambda upstream, result, x, y: upstream * y,  # w.r.t. x
    lambda upstream, result, x, y: upstream * x,  # w.r.t. y
])

register_vjp(wnp.true_divide, [
    lambda upstream, result, x, y: upstream / y,  # w.r.t. x
    lambda upstream, result, x, y: upstream * (-x / y ** 2),  # w.r.t. y
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
    lambda upstream, result, x, y: upstream * (y * x ** (y - 1)),  # w.r.t. x
    lambda upstream, result, x, y: upstream * (result * log(x)),  # w.r.t. y
])

# shamelessly taken from autograd, brilliant!
def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y)) 