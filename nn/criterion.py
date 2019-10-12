#pylint: disable=no-member

import numpy as _np

import autodiff
from autodiff.diff import value_and_grad, value, grad
import autodiff.numpy_grad.wrapper as np


class Criterion:
    def __init__(self):
        pass

    def calc_loss(self, true, predicted):
        pass

    def loss_func(self):
        pass


class MSE(Criterion):
    def __init__(self):
        super().__init__()
    
    def calc_loss(self, true, predicted):
        value, loss_grad = value_and_grad(self.loss_func, 'predicted_y')(feed_dict={'predicted_y': predicted, 'true_y': true})
        return value, loss_grad

    def loss_func(self, feed_dict={}):
        diff = np.subtract(np.Placeholder(predicted_y=feed_dict['predicted_y']), np.Placeholder(true_y=feed_dict['true_y']))
        return np.multiply(np.power(diff, np.Constant(2)), np.Constant(1/2))
