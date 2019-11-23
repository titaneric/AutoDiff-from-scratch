import numpy as _np

import autodiff as ad
from autodiff import value_and_grad, value, grad


class Optimizer:
    def __init__(self, lr, parameters):
        self.lr = lr
        self.parameters = parameters

    def step(self):
        pass


class GradientDescent(Optimizer):
    def __init__(self, lr, parameter, variables):
        super().__init__(lr, parameter)
        self.variables = variables

    def step(self, model_grad):
        for i, var in zip(reversed(range(len(self.parameters))),
                          reversed(self.variables)):
            self.parameters[i] -= self.lr * model_grad[var]
