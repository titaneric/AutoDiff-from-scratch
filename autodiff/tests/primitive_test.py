#pylint: disable=no-member
import unittest

import numpy as np

import autodiff as ad
from autodiff.autodiff.core import primitive_vhp
from autodiff.utils.test_utils import check_value, check_vjp, func_helper

numpy_function = {
    func_name: np.__dict__[func_name]
    if not func_name.startswith("__") else np.ndarray.__dict__[func_name]
    for func_name in primitive_vhp.keys()
}


class TestJVPMethods(unittest.TestCase):
    def test_add(self):
        v1, v2 = 1, 2
        args = (v1, v2)
        check_vjp(numpy_function["add"], func_helper("add"), args)


class TestNumpyMethods(unittest.TestCase):
    def test_add(self):
        v1, v2 = 1, 2
        args = (v1, v2)
        check_value(numpy_function["add"], func_helper("add"), args)


if __name__ == "__main__":
    unittest.main()