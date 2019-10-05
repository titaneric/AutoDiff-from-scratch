#pylint: disable=no-member

from . import wrapper as wnp
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
    lambda upstream, result, x, y: upstream * -x,  # w.r.t. y
])

register_vjp(wnp.true_divide, [
    lambda upstream, result, x, y: upstream / y,  # w.r.t. x
    lambda upstream, result, x, y: upstream * (-x / y ** 2),  # w.r.t. y
])