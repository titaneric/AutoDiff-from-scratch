#pylint: disable=no-member
import unittest

import numpy as np

import autodiff as ad
from autodiff.autodiff.core import primitive_vhp
from autodiff import value_and_grad, value, grad

numpy_function = {
    func_name: np.__dict__[func_name]
    if not func_name.startswith("__") else np.ndarray.__dict__[func_name]
    for func_name in primitive_vhp.keys()
}


def func_helper(func):
    def wrapped(*args, **kwargs):
        arg_list = (ad.Variable(arg) for arg in args)
        return ad.__dict__[func](*arg_list)

    return wrapped


class TestVHPMethods(unittest.TestCase):
    def test_add(self):
        v1, v2 = 1, 2
        args = (v1, v2)

        v, g = value_and_grad(func_helper("add"))(*args)

        self.assertEqual(v, numpy_function["add"](*args))
        self.assertEqual(g[id(v1)], 1)
        self.assertEqual(g[id(v2)], 1)


if __name__ == "__main__":

    print(numpy_function)

    unittest.main()