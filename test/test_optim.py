import unittest

import util

from optim import *

# use 32 bit for python float
theano.config.floatX = "float32"


def early_stop_train(train_func, valid_func):
    last_valid = np.inf
    for epoch in range(50):
        train_func()
        valid = valid_func()
        print "Epoch #{}\t{}".format(epoch+1, valid)
        if valid < last_valid:
            last_valid = valid
        else:
            print "Early-stopped."
            # early-stop
            break


def mk_funcs(optimizer, params, g_params, x, cost, train_data, valid_data, batch_size):
    index = T.lscalar("index")
    train_func = theano.function(
        inputs=[index],
        outputs=cost,
        updates=optimizer(params, g_params),
        givens={x: train_data[batch_size * index:batch_size * (index + 1)]}
    )
    valid_func = theano.function(
        inputs=[],
        outputs=cost,
        givens={x: valid_data},
        )
    return train_func, valid_func


class TestOptim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n_in = 10
        n_hidden = 7

        cls.n_in = n_in
        cls.n_hidden = n_hidden

        w_e = theano.shared(value=np.zeros((n_in, n_hidden)), borrow=True)
        b_e = theano.shared(value=np.zeros(n_in), borrow=True)
        w_d = theano.shared(value=np.zeros((n_hidden, n_in)), borrow=True)
        b_d = theano.shared(value=np.zeros(n_hidden), borrow=True)

        cls.w_e = w_e
        cls.b_e = b_e
        cls.w_d = w_d
        cls.b_d = b_d

        # auto-encoder
        x = T.matrix("input")
        h = T.tanh(T.dot(x, w_e) + b_e)  # code
        x_ = T.tanh(T.dot(h, w_d) + b_d)
        cost = T.mean((x - x_) ** 2)

        ae_func = theano.function([x], x_)

        params = [w_e, b_e, w_d, b_d]
        g_params = T.grad(cost, params)  # gradients

        cls.x = x
        cls.cost = cost
        cls.ae_func = ae_func
        cls.params = params
        cls.g_params = g_params

        # sine curves
        def make_sample(rng):
            x = []
            amp, omega, theta = rng.rand(3)
            for t in np.arange(0, 1, 1./10):
                x.append((amp * np.sin((omega + 1) * t + theta)).astype("float32"))
            return x

        rng = np.random.RandomState(0)
        train_data = theano.shared(np.asarray([make_sample(rng) for _ in range(10000)]))
        valid_data = theano.shared(np.asarray([make_sample(rng) for _ in range(100)]))
        test_data = theano.shared(np.asarray([make_sample(rng) for _ in range(100)]))

        cls.train_data = train_data
        cls.valid_data = valid_data
        cls.test_data = test_data


    def setUp(self):
        rng = np.random.RandomState(0)
        init_w_e, init_b_e = util.init_weight(rng, self.n_in, self.n_hidden)
        init_w_d, init_b_d = util.init_weight(rng, self.n_hidden, self.n_in)
        self.w_e.set_value(init_w_e, borrow=True)
        self.b_e.set_value(init_b_e, borrow=True)
        self.w_d.set_value(init_w_d, borrow=True)
        self.b_d.set_value(init_b_d, borrow=True)


    def test_auto_encoder(self):
        """Test whether I correctly implemented an auto-encoder."""

        batch_size = 100
        lr = theano.shared(np.asarray(1., dtype="float32"))

        def optimizer(params, g_params):
            return [(w, w - lr * dw) for w, dw in zip(params, g_params)]
        train_func, valid_func = mk_funcs(optimizer, self.params, self.g_params,
                                          self.x, self.cost, self.train_data, self.valid_data,
                                          batch_size)
        def train_mini_batch():
            for idx in range(10000 / batch_size):
                train_func(idx)
        early_stop_train(train_mini_batch, valid_func)

        for x in self.test_data.get_value():
            self.assertTrue(np.allclose(x, self.ae_func(np.matrix(x))[0], atol=0.2))


    def test_adagrad(self):
        """ADAGRAD"""

        batch_size = 100
        lr = np.asarray(0.1, dtype="float32")    # 0.1 is the most appropriate order for this value
        epsilon = np.asarray(0.001, dtype="float32")

        train_func, valid_func = mk_funcs(lambda p, g_p: adagrad(p, g_p, lr, epsilon),
                                          self.params, self.g_params,
                                          self.x, self.cost, self.train_data, self.valid_data,
                                          batch_size)

        def train_mini_batch():
            for idx in range(10000 / batch_size):
                train_func(idx)
        early_stop_train(train_mini_batch, valid_func)

        for x in self.test_data.get_value():
            self.assertTrue(np.allclose(x, self.ae_func(np.matrix(x))[0], atol=0.2))


    def test_adadelta(self):
        """ADADELTA"""

        batch_size = 100
        # seems to be insensitive to these parameters (especially decay)
        # strange...
        # ANSWER: simple SGD with learning rate fixed to 1.0 suffices
        decay = np.asarray(0.95, dtype="float32")
        eps = np.asarray(0.001, dtype="float32")

        train_func, valid_func = mk_funcs(lambda p, g_p: adadelta(p, g_p, decay, eps),
                                          self.params, self.g_params,
                                          self.x, self.cost, self.train_data, self.valid_data,
                                          batch_size)

        def train_mini_batch():
            for idx in range(10000 / batch_size):
                train_func(idx)
        early_stop_train(train_mini_batch, valid_func)

        for x in self.test_data.get_value():
            self.assertTrue(np.allclose(x, self.ae_func(np.matrix(x))[0], atol=0.2))


if __name__ == '__main__':
    unittest.main()

