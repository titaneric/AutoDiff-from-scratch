#pylint: disable=no-member
import unittest
import functools
import itertools
from collections import namedtuple

import numpy as np
import numpy.random as npr
from absl.testing import parameterized

import autodiff as ad
from autodiff.autodiff.core import primitive_vhp
from autodiff.utils.test_utils import AutoDiffTestCase, check_value, check_vjp, func_helper

special_function = {"sum", "dot", "reshape", "__getitem__"}

op_record = namedtuple("OpRecord", ["name", "np_func", "ad_func", "nargs"])

nargs_dict = {
    "negative": 1,
    "reciprocal": 1,
    "exp": 1,
    "log": 1,
    "sin": 1,
    "cos": 1,
    "add": 2,
    "subtract": 2,
    "multiply": 2,
    "true_divide": 2,
    "maximum": 2,
    "minimum": 2,
    "power": 2
}


def get_op_record(func_name):
    np_func = np.__dict__[func_name]
    ad_func = func_helper(func_name)
    nargs = nargs_dict[func_name]
    return op_record(func_name, np_func, ad_func, nargs)


op_records = [
    get_op_record(func_name) for func_name in primitive_vhp.keys()
    if func_name not in special_function
]


class TestJVPMethods(AutoDiffTestCase):
    @parameterized.named_parameters([{
        "testcase_name": record.name,
        "ad_func": record.ad_func,
        "np_func": record.np_func,
        "nargs": record.nargs
    } for record in op_records])
    def test_op(self, ad_func, np_func, nargs):
        args = [npr.rand(1) for _ in range(nargs)]
        check_vjp(np_func, ad_func, args)


class TestNumpyMethods(AutoDiffTestCase):
    @parameterized.named_parameters([{
        "testcase_name": record.name,
        "ad_func": record.ad_func,
        "np_func": record.np_func,
        "nargs": record.nargs
    } for record in op_records])
    def test_op(self, ad_func, np_func, nargs):
        args = [npr.randint(1, high=4) for _ in range(nargs)]
        check_value(np_func, ad_func, args)


if __name__ == "__main__":
    unittest.main()