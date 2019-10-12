import numpy as _np

import autodiff
from autodiff.diff import value_and_grad, value, grad
import autodiff.numpy_grad.wrapper as np

class Optimizer:
    def __init__(self, lr, parameter):
        self.lr = lr
        self.parameter = parameter

    def step(self):
        pass

class GradientDescent(Optimizer):
    def __init__(self, lr, parameter):
        super().__init__(lr, parameter)
    
    def step(self, grad_value, loss_grad):
        self.parameter -= self.lr * _np.dot(grad_value, loss_grad)  #x.T * (y_hat - y)