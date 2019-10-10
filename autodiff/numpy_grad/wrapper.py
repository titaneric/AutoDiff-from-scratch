import types

import numpy as _np

from autodiff.graph.tracer import primitive, constant, variable, placeholder

nograd_functions = [
    _np.ndim, _np.shape, _np.iscomplexobj, _np.result_type, _np.zeros_like,
    _np.ones_like, _np.floor, _np.ceil, _np.round, _np.rint, _np.around,
    _np.fix, _np.trunc, _np.all, _np.any, _np.argmax, _np.argmin,
    _np.argpartition, _np.argsort, _np.argwhere, _np.nonzero, _np.flatnonzero,
    _np.count_nonzero, _np.searchsorted, _np.sign, _np.ndim, _np.shape,
    _np.floor_divide, _np.logical_and, _np.logical_or, _np.logical_not,
    _np.logical_xor, _np.isfinite, _np.isinf, _np.isnan, _np.isneginf,
    _np.isposinf, _np.allclose, _np.isclose, _np.array_equal, _np.array_equiv,
    _np.greater, _np.greater_equal, _np.less, _np.less_equal, _np.equal,
    _np.not_equal, _np.iscomplexobj, _np.iscomplex, _np.size, _np.isscalar,
    _np.isreal, _np.zeros_like, _np.ones_like, _np.result_type
]

excluded_function = [
    _np.linspace, _np.arange,
]

def wrap_func(numpy, local):
    function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in numpy.items():
        if obj in nograd_functions or obj in excluded_function:
            local[name] = obj
        elif type(obj) in function_types:
            local[name] = primitive(obj)
        elif isinstance(obj, type) and _np.issubdtype(obj, _np.integer):
            local[name] = obj

wrap_func(_np.__dict__, globals())
globals()['Constant'] = constant(_np.array)
globals()['Variable'] = variable(_np.array)
globals()['Placeholder'] = placeholder(_np.array)
