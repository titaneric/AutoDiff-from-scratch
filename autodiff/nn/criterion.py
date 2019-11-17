#pylint: disable=no-member

import numpy as _np

import autodiff as ad
from autodiff import value_and_grad, value, grad


class Criterion:
    def __init__(self):
        pass

    def calc_loss(self, true, predicted):
        value, loss_grad = value_and_grad(self.loss_func,
                                          'predicted_y')(feed_dict={
                                              'predicted_y': predicted,
                                              'true_y': true
                                          })
        return _np.average(value), loss_grad

    def loss_func(self):
        pass


class MSE(Criterion):
    def __init__(self):
        super().__init__()

    def loss_func(self, feed_dict={}):
        diff = ad.subtract(
            ad.Placeholder(predicted_y=feed_dict['predicted_y']),
            ad.Placeholder(true_y=feed_dict['true_y']))
        return ad.multiply(ad.power(diff, ad.Constant(2)), ad.Constant(1 / 2))
    