from . import autodiff  # register the VHP of primitive function
import autodiff.autodiff.numpy_grad.wrapper as np  # export the primitive function
#TODO need refactor, may use __all__?
from autodiff.autodiff.core import backward_prop, zero_grad
from autodiff.autodiff.global_vars import register_graph, set_forwarded, set_parameters

# export the gradient-related function
from autodiff.autodiff.diff import *
globals().update(np.__dict__)

for func in [
        register_graph, set_parameters, backward_prop, zero_grad, set_forwarded
]:
    globals()[func.__name__] = func
