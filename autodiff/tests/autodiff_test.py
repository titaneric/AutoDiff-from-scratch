#pylint: disable=no-member
import unittest

from absl.testing import parameterized

import autodiff as ad
from autodiff.autodiff.core import primitive_vhp
import autodiff.utils.test_utils as adu


class TestJVPMethods(adu.AutoDiffTestCase):
    @parameterized.named_parameters([{
        "testcase_name":
        adu.format_test_name(record.name, shape),
        "ad_func":
        record.ad_func,
        "np_func":
        record.np_func,
        "rng":
        record.rng,
        "shape":
        shape,
        "nargs":
        record.nargs
    } for shape in adu.tested_shapes for record in adu.OP_RECORDS])
    def test_op(self, ad_func, np_func, rng, shape, nargs):
        args = [rng(shape) for _ in range(nargs)]
        adu.check_vjp(np_func, ad_func, args)


if __name__ == "__main__":
    unittest.main()