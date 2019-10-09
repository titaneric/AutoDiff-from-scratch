import numpy as _np

from autodiff.graph.tracer import primitive, constant, variable, placeholder

def wrap_func(numpy, local):
    for name, obj in numpy.items():
        if isinstance(obj, _np.ufunc):
            local[name] = primitive(obj)

wrap_func(_np.__dict__, globals())
globals()['constant'] = constant(_np.array)
globals()['variable'] = variable(_np.array)
globals()['placeholder'] = placeholder(_np.array)

globals()['linspace'] = _np.linspace
