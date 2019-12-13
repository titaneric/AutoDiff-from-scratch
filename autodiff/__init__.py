from . import autodiff  # register the VHP of primitive function
import autodiff.autodiff.numpy_grad.wrapper as np  # export the primitive function
#TODO need refactor, may use __all__?
from autodiff.autodiff.core import forward_prop, backward_prop, zero_grad
from autodiff.autodiff.global_vars import set_forwarded, set_parameters

# export the gradient-related function
from autodiff.autodiff.diff import *
globals().update(np.__dict__)

for func in [
        set_parameters,
        set_forwarded,
        forward_prop,
        backward_prop,
        zero_grad,
]:
    globals()[func.__name__] = func
