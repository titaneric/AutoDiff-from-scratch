#pylint: disable=no-member
import unittest

import numpy as _np

import autodiff as ad
from autodiff import value_and_grad, value, grad


def binary_func_helper(func):
    def true_binary_func(*args, **kwargs):
        v1 = ad.Variable(kwargs["v1"])
        v2 = ad.Variable(kwargs["v2"])
        return ad.__dict__[func](v1, v2)

    return true_binary_func


class TestVHPMethods(unittest.TestCase):
    def test_add(self):
        v1, v2 = 1, 2

        v, g = value_and_grad(binary_func_helper("add"))(v1=v1, v2=v2)

        self.assertEqual(v, v1 + v2)
        self.assertEqual(g[id(v1)], 1)
        self.assertEqual(g[id(v2)], 1)


if __name__ == "__main__":
    unittest.main()