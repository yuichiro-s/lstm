import theano
import theano.tensor as T
import numpy as np


def adagrad(params, g_params, learning_rate, epsilon):
    """
    ADAGRAD optimizer.

    :type params: list of theano.tensor.TensorVariable
    :param params: parameters
    :type g_params: list of theano.tensor.TensorVariable
    :param g_params: gradients
    :param learning_rate: learning_rate shared among all dimensions
    :return: list of updates to be used when constructing a theano function
    """

    updates = []

    # initialize accumulator
    accs = []
    for param in params:
        # TODO fixed this not to use get_value() for retrieving the dimension of param
        acc = theano.shared(np.zeros_like(param.get_value()), borrow=True)
        accs.append(acc)

    for param, grad, acc in zip(params, g_params, accs):
        # accumulate gradient
        new_acc = acc + grad ** 2

        # update accumulator
        updates.append((acc, new_acc))

        # update parameter
        updates.append((param, param - learning_rate / T.sqrt(new_acc + epsilon) * grad))

    return updates


def adadelta(params, g_params, decay_rate, epsilon):
    """
    ADADELTA optimizer.

    :return: list of updates to be used when constructing a theano function
    """

    # initialize accumulators
    g_accs = []
    x_accs = []
    for param in params:
        # TODO fixed this not to use get_value() for retrieving the dimension of param
        g_acc = theano.shared(np.zeros_like(param.get_value()), borrow=True)
        x_acc = theano.shared(np.zeros_like(param.get_value()), borrow=True)
        g_accs.append(g_acc)
        x_accs.append(x_acc)

    updates = []
    for param, grad, g_acc, x_acc in zip(params, g_params, g_accs, x_accs):
        decay_rate_comp = np.asarray(1. - decay_rate, dtype="float32")  # 1 - decay_rate

        # accumulate gradient
        new_g_acc = decay_rate * g_acc + decay_rate_comp * (grad ** 2)

        # RMS
        g_rms = T.sqrt(new_g_acc + epsilon)
        x_rms = T.sqrt(x_acc + epsilon)     # use RMS at t-1

        # compute update
        update = -x_rms/g_rms * grad

        # accumulate update
        new_x_acc = decay_rate * x_acc + decay_rate_comp * (update ** 2)

        # update accumulator
        updates.append((g_acc, new_g_acc))
        updates.append((x_acc, new_x_acc))

        # update parameter
        updates.append((param, param + update))

    return updates


def sgd(params, g_params, learning_rate):
    return [(w, w - learning_rate * dw) for w, dw in zip(params, g_params)]
