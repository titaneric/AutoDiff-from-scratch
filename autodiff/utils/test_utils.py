#pylint: disable=no-member
import unittest
from functools import partial
from collections import namedtuple

from absl.testing import parameterized
import numpy as np
import numpy.random as npr

import autodiff as ad
from autodiff import value_and_grad, value, grad
"""
    This numerical VJP checking is greatly referenced from google/jax.
"""
"""
    For random generator
"""


def _rand_type(rand, shape, dtype=np.float64, scale=1., post=lambda x: x):
    r = lambda: np.asarray(scale * rand(*shape), dtype=dtype)
    vals = r()
    return post(vals)


def rand_default():
    rand = npr.RandomState(0).randn
    return partial(_rand_type, rand, scale=3)


def rand_positive():
    post = lambda x: np.where(x < 0, -x, x)
    rand = npr.RandomState(0).randn
    return partial(_rand_type, rand, scale=2, post=post)


def rand_small():
    rand = npr.RandomState(0).randn
    return partial(_rand_type, rand, scale=1e-3)


def rand_not_small():
    post = lambda x: x + np.where(x > 0, 10.0, -10.0)
    rand = npr.RandomState(0).randn
    return partial(_rand_type, rand, scale=3., post=post)


"""
    Format string
"""


def format_test_name(func, shape):
    return f"{func}_{shape}"


"""
    For grad check
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


default_tol = 1e-2


def check_vjp(func, func_vjp, args):
    for i in range(len(args)):
        out = grad(func_vjp, wrt=id(args[i]))(*args)
        expected = numerical_jvp(func, args, i)
        np.testing.assert_allclose(out,
                                   expected,
                                   atol=default_tol,
                                   rtol=default_tol)


"""
    For value check
"""


def check_value(func, func_value, args):
    out = value(func_value)(*args)
    expected = func_value(*args)
    np.testing.assert_allclose(out, expected)


def func_helper(func):
    def wrapped(*args, **kwargs):
        arg_list = (ad.Variable(arg) for arg in args)
        return ad.__dict__[func](*arg_list)

    return wrapped


class AutoDiffTestCase(parameterized.TestCase):
    def assert_all_close(self):
        pass

"""
    Test cases
"""

tested_shapes = [(), (3, ), (3, 2)]
record = namedtuple("TestRecord",
                         ["name", "np_func", "ad_func", "nargs", "rng"])


def record_factory(name, nargs, rng):
    np_func = np.__dict__[name]
    ad_func = func_helper(name)
    return record(name, np_func, ad_func, nargs, rng)


OP_RECORDS = [
    record_factory("negative", 1, rand_default()),
    record_factory("reciprocal", 1, rand_positive()),
    record_factory("exp", 1, rand_small()),
    record_factory("log", 1, rand_positive()),
    record_factory("sin", 1, rand_default()),
    record_factory("cos", 1, rand_default()),
    record_factory("add", 2, rand_default()),
    record_factory("subtract", 2, rand_default()),
    record_factory("multiply", 2, rand_default()),
    record_factory("true_divide", 2, rand_not_small()),
    record_factory("maximum", 2, rand_default()),
    record_factory("minimum", 2, rand_default()),
    record_factory("power", 2, rand_positive()),
]
