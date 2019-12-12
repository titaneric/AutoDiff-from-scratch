#pylint: disable=no-member
import numpy as np

import autodiff as ad
from autodiff import value_and_grad, value, grad
from autodiff.nn.criterion import CrossEntropy


# def cross_entropy(predictions, targets):
#     # batch_size = len(predictions)
#     # classes = targets.argmax(axis=1)
#     # log_likelihood = np.negative((predictions[range(batch_size), classes]))
#     # result = np.add(log_likelihood, np.log(np.sum(np.exp(predictions),
#     #                                               axis=1)))
#     # return np.reshape(result, (-1, 1))
#     return np.reshape(
#         np.negative(np.sum(np.multiply(targets, np.log(predictions)), axis=1)),
#         (-1, 1))


def cross_entropy(p,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss

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


def test_grad(predictions, targets):
    batch_size = predictions.shape[0]
    grad = predictions.copy()
    grad[range(batch_size), targets] -= 1
    return grad

def softmax(X):
    """
        X size is (batch_size, num_class)
    """
    exps = np.exp(X)
    return np.true_divide(exps, np.reshape(np.sum(exps, axis=1), (-1, 1)))


def softmax_ad(feed_dict={}):
    exps = ad.Constant(ad.exp(ad.Placeholder(X=feed_dict["X"])))
    return ad.true_divide(
        exps, ad.reshape(ad.sum(exps, axis=1), ad.Constant((-1, 1))))


def softmax_cross_entropy(X, y):
    predictions = softmax(X)

    return cross_entropy(predictions, y)


if __name__ == "__main__":
    X = np.array([[1, 1, 1, 1], [1, 1, 1, 5]])

    print(softmax(X))
    # print(softmax_ad({"X": X}))
    # loss_value, loss_grad = value_and_grad(softmax_ad)(feed_dict={"X": X})
    # print(loss_value)
    # print(loss_grad)
    predictions = np.array([[0.25, 0.25, 0.25, 0.25], [0.01, 0.01, 0.01, 0.96]])

    targets = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ])
    classes = np.array([3, 3])
    # print(cross_entropy(predictions, targets))
    # loss_value, loss_grad = value_and_grad(cross_entropy_ad,
    #                                        'predictions')(feed_dict={
    #                                            'predictions': predictions,
    #                                            'targets': targets
    #                                        })
    # print(loss_value)
    # print("-" * 10)
    # print(cross_entropy(predictions, classes))
    # print(grad_cross_entropy(predictions, classes))
    # print(test_grad(predictions, classes))
    # # print(loss_grad)

    ce = CrossEntropy()
    avg_v, grad = ce(targets, predictions)
    # print(avg_v)
    # print(grad)
