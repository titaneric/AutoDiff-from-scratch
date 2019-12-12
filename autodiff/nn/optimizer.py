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
    def __init__(self, lr, parameter):
        super().__init__(lr, parameter)

    def step(self, model_grad):
        for i, para_info in reversed(list(enumerate(self.parameters))):
            self.parameters[i]["variables"] -= self.lr * model_grad[para_info["array_id"]]
