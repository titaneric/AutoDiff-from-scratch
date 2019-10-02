import numpy as _np

from autodiff.tracer import primitive

def wrap_func(numpy, local):
    for name, obj in numpy.items():
        if isinstance(obj, _np.ufunc):
            local[name] = primitive(obj)

wrap_func(_np.__dict__, globals())