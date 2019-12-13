#pylint: disable=no-member
import unittest

from absl.testing import parameterized
import numpy as np

import autodiff as ad
from autodiff import value_and_grad, value, grad
"""
    This numerical VJP checking is greatly referenced from google/jax.
"""

epsilon = 1e-4

add = lambda wrt, argument: [
    np.add(arg, epsilon / 2) if i == wrt else arg
    for i, arg in enumerate(argument)
]
sub = lambda wrt, argument: [
    np.subtract(arg, epsilon / 2) if i == wrt else arg
    for i, arg in enumerate(argument)
]


def numerical_jvp(func, arguments, wrt):
    func_pos = func(*add(wrt, arguments))
    func_neg = func(*sub(wrt, arguments))
    return (func_pos - func_neg) / epsilon


def func_helper(func):
    def wrapped(*args, **kwargs):
        arg_list = (ad.Variable(arg) for arg in args)
        return ad.__dict__[func](*arg_list)

    return wrapped


def check_vjp(func, func_vjp, args):
    for i in range(len(args)):
        out = grad(func_vjp, wrt=id(args[i]))(*args)
        expected = numerical_jvp(func, args, i)
        np.testing.assert_allclose(out, expected, atol=1e-4, rtol=1e-4)


def check_value(func, func_value, args):
    out = value(func_value)(*args)
    expected = func_value(*args)
    np.testing.assert_allclose(out, expected)


class AutoDiffTestCase(parameterized.TestCase):
    def assert_all_close(self):
        pass