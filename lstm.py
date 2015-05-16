import theano
import theano.scan_module
import theano.tensor as T
import theano.tensor.shared_randomstreams
import theano.printing
import numpy as np
import util


def get_param(name, n_in, n_out, params, rng):
    w_name = "w_" + name
    b_name = "b_" + name
    if params is not None and w_name in params:
        assert b_name in params
        init_w = params[w_name]
        init_b = params[b_name]
    else:
        init_w, init_b = util.init_weight(rng, n_in, n_out)
    w = theano.shared(name=w_name, borrow=True, value=init_w)
    b = theano.shared(name=b_name, borrow=True, value=init_b)
    return w, b


class Lstm(object):
    """LSTM sequential classifier"""

    def __init__(self, n_in, n_hidden, n_out, rng=np.random.RandomState(), params=None, init_forget=True):
        """
        :type n_in: int
        :type n_hidden: int
        :type n_out: int

        :param n_in: size of input layer
        :param n_hidden: size of hidden layer (and memory cells)
        :param n_out: size of output layer

        :param params: dict of numpy arrays of parameter names and values
        :param init_forget: initialize forget gate biases to high values

        Make sure that all samples in a batch have the same length.
        """

        # parameters
        w_xi, b_xi = get_param("xi", n_in, n_hidden, params, rng)
        w_xf, b_xf = get_param("xf", n_in, n_hidden, params, rng)
        w_xo, b_xo = get_param("xo", n_in, n_hidden, params, rng)
        w_xc, b_xc = get_param("xc", n_in, n_hidden, params, rng)
        w_mi, b_mi = get_param("mi", n_hidden, n_hidden, params, rng)
        w_mf, b_mf = get_param("mf", n_hidden, n_hidden, params, rng)
        w_mo, b_mo = get_param("mo", n_hidden, n_hidden, params, rng)
        w_mc, b_mc = get_param("mc", n_hidden, n_hidden, params, rng)
        w_my, b_my = get_param("my", n_hidden, n_out, params, rng)

        # initialize forget gate biases to high values
        if init_forget and params is not None and not "b_xf" in params:
            assert not "b_mf" in params
            b_xf = theano.shared(name="b_xf", borrow=True, value=np.empty(n_hidden).fill(5.))
            b_mf = theano.shared(name="b_mf", borrow=True, value=np.empty(n_hidden).fill(5.))

        def step(x, m_prev, c_prev):
            # x: float32 matrix of size (batch_size x n_in)
            # m_prev: float32 matrix of size (batch_size x n_hidden)
            # c_prev: float32 matrix of size (batch_size x n_hidden)

            # gates
            input_gate = T.nnet.sigmoid(
                T.dot(x, w_xi) + b_xi +
                T.dot(m_prev, w_mi) + b_mi)
            forget_gate = T.nnet.sigmoid(
                T.dot(x, w_xf) + b_xf +
                T.dot(m_prev, w_mf) + b_mf)
            output_gate = T.nnet.sigmoid(
                T.dot(x, w_xo) + b_xo +
                T.dot(m_prev, w_mo) + b_mo)

            input = T.tanh(
                T.dot(x, w_xc) + b_xc +
                T.dot(m_prev, w_mc) + b_mc)

            # new CEC value
            c = forget_gate * c_prev + input_gate * input

            # new output value
            m = output_gate * c

            # prediction
            y = T.nnet.softmax(T.dot(m, w_my) + b_my)

            return [m, c, y]

        xs = T.ftensor3("xs")  # batch_size x # of steps x n_in
        batch_size = T.cast(xs.shape[0], "int32")
        length = T.cast(xs.shape[1], "int32")

        [ms, cs, ys], _ = theano.scan(step,
                                      sequences=[xs.dimshuffle(1, 0, 2)],
                                      outputs_info=[T.alloc(0., batch_size, n_hidden),
                                                    T.alloc(0., batch_size, n_hidden),
                                                    None])
        ys = ys.dimshuffle(1, 0, 2)
        ms = ms.dimshuffle(1, 0, 2)
        cs = cs.dimshuffle(1, 0, 2)
        self.xs = xs    # input (batch_size x n_steps x n_in)
        self.ms = ms    # hidden unit value (batch_size x n_steps x n_hidden)
        self.cs = cs    # memory cell values (batch_size x n_steps x n_hidden)
        self.ys = ys    # output (batch_size x n_steps x n_out)

        predict = T.argmax(ys, axis=2)
        self.predict = predict  # predictions (batch_size x n_steps)

        ys_correct = T.imatrix("ys_correct")
        probs = ys[T.arange(batch_size)[:, np.newaxis], T.arange(length), ys_correct]
        cost = -T.mean(T.log(probs))
        self.ys_correct = ys_correct    # correct predictions (batch_size x n_steps)
        self.cost = cost    # cost (NLL) of all predictions

        # list of all parameters
        self.params = [w_xi, b_xi,
                       w_xf, b_xf,
                       w_xo, b_xo,
                       w_xc, b_xc,
                       w_mi, b_mi,
                       w_mf, b_mf,
                       w_mo, b_mo,
                       w_mc, b_mc,
                       w_my, b_my, ]
