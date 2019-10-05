import numpy as _np

from autodiff.graph.tracer import primitive, constant, variable

def wrap_func(numpy, local):
    for name, obj in numpy.items():
        if isinstance(obj, _np.ufunc):
            local[name] = primitive(obj)

wrap_func(_np.__dict__, globals())
globals()['const'] = constant(_np.array)
globals()['var'] = variable(_np.array)
