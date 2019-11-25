#pylint: disable=no-member
import numpy as np

import autodiff as ad
from autodiff import value_and_grad, value, grad
from autodiff.nn.criterion import CrossEntropy

def cross_entropy(predictions, targets):
    # batch_size = len(predictions)
    # classes = targets.argmax(axis=1)
    # log_likelihood = np.negative((predictions[range(batch_size), classes]))
    # result = np.add(log_likelihood, np.log(np.sum(np.exp(predictions),
    #                                               axis=1)))
    # return np.reshape(result, (-1, 1))
    return np.reshape(
        np.negative(np.sum(np.multiply(targets, np.log(predictions)), axis=1)),
        (-1, 1))


def cross_entropy_ad(feed_dict={}):
    prediction_place = ad.Placeholder(predictions=feed_dict["predictions"])
    targets_place = ad.Placeholder(targets=feed_dict["targets"])

    return ad.reshape(
        ad.negative(
            ad.sum(ad.multiply(targets_place, ad.log(prediction_place)),
                   axis=1)), (-1, 1))


def grad_cross_entropy(predictions, targets):
    batch_size = len(predictions)

    #
    one_hot_vector = np.zeros_like(predictions)
    one_hot_vector[range(batch_size), targets] = 1

    return np.divide(np.add(np.negative(one_hot_vector), softmax(predictions)),
                     batch_size)


def softmax(X):
    """
        X size is (batch_size, num_class)
    """
    exps = np.exp(X)
    return np.divide(exps, np.sum(exps, axis=1, keepdims=True))


def softmax_cross_entropy(X, y):
    predictions = softmax(X)

    return cross_entropy(predictions, y)


X = np.array([[1, 1, 1, 1], [1, 1, 1, 5]])

predictions = np.array([[0.25, 0.25, 0.25, 0.25], [0.01, 0.01, 0.01, 0.96]])

targets = np.array([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])
classes = np.array([3, 3])
# print(cross_entropy(predictions, targets))
loss_value, loss_grad = value_and_grad(cross_entropy_ad, 'predictions')(feed_dict={
    'predictions': predictions,
    'targets': targets
})
print(loss_value)
# print(grad_cross_entropy(predictions, classes))
print(loss_grad)

ce = CrossEntropy()
print(ce(targets, predictions))
