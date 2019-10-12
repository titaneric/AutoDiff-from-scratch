#pylint: disable=no-member
import unittest


import numpy as _np

import autodiff
from autodiff.autodiff.diff import value_and_grad, value, grad
import autodiff.autodiff.numpy_grad.wrapper as np

# def binary_func_helper(func, v1, v2):
    # return func(np.Variable(v1=v1), np.Variable(v2=v2))

class TestStringMethods(unittest.TestCase):

    def test_add(self):
        v1, v2 = 1, 2
        def add(v1, v2):
            return np.add(np.Variable(v1=v1), np.Variable(v2=v2))

        v, g = value_and_grad(add)(v1=v1, v2=v2)
        self.assertEqual(v, v1 + v2)
        self.assertEqual(g['v1'], 1)
        self.assertEqual(g['v2'], 1)


if __name__ == "__main__":
    unittest.main()