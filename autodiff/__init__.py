from . import autodiff  # register the VHP of primitive function
import autodiff.autodiff.numpy_grad.wrapper as np  # export the primitive function

# export the gradient-related function
from autodiff.autodiff.diff import *
globals().update(np.__dict__)
