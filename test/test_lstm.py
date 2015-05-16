import unittest
from numpy.core.multiarray import dtype
import theano
import theano.printing
import optim

from lstm import *

# use 32 bit for python float
theano.config.floatX = "float32"
theano.config.exception_verbosity="high"


class TestLstm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    def test_train(self):
        """Test whether LSTM can learn to output the sign of sum of previous inputs."""

        rng = np.random.RandomState(0)
        n_in = 5
        n_out = 2
        n_hidden = 10
        lstm = Lstm(n_in, n_hidden, n_out, rng=rng, params=None)

        def make_sample(rng, n, x_dim):
            xs = []
            ys = []
            sum = 0
            for _ in range(n):
                x = rng.rand(x_dim) - 0.5
                sum += np.sum(x)
                y = 1 if sum >= 0 else 0
                xs.append(x)
                ys.append(y)
            return xs, ys

        def make_batch(rng, batch_size, n, x_dim):
            xss = []
            yss = []
            for _ in range(batch_size):
                xs, ys = make_sample(rng, n, x_dim)
                xss.append(xs)
                yss.append(ys)
            xss = np.asarray(xss, dtype="float32")
            yss = np.asarray(yss, dtype="int32")
            return xss, yss

        batch_size = 10
        batch_num = 1000

        train_x = []
        train_y = []
        for i in range(batch_num):
            x, y = make_batch(rng, batch_size, i/200 + 3, n_in)     # length is 3 to 7
            train_x.append(x)
            train_y.append(y)
        test_x_short, test_y_short = make_batch(rng, batch_size, 4, n_in)
        test_x_long, test_y_long = make_batch(rng, batch_size, 12, n_in)

        g_params = T.grad(lstm.cost, lstm.params)
        #lr = 0.1
        #updates = [(w, w - lr * dw) for w, dw in zip(lstm.params, g_params)]
        #updates = optim.adagrad(lstm.params, g_params, 0.01, 0.0001)
        updates = optim.adadelta(lstm.params, g_params, 0.99, 0.000001)
        func_train = theano.function(
            inputs=[lstm.xs, lstm.ys_correct],
            outputs=lstm.cost,
            updates=updates,
        )
        func_forward = theano.function(
            inputs=[lstm.xs],
            outputs=[lstm.predict, lstm.ys, lstm.ms, lstm.cs],
            )
        # train for 20 epochs
        for i in range(20):
            print "Epoch #{}".format(i + 1),
            cost = []
            for x, y in zip(train_x, train_y):
                cost.append(func_train(x, y))
            print "cost: ", np.mean(cost)

        # print parameters
        #for param in lstm.params:
        #    print param
        #    print param.get_value()
        #    print

        # test
        def test(test_x, test_y):
            guess, ys, ms, cs = func_forward(test_x)
            for g, ty, y, m, c, sample in zip(guess, test_y, ys, ms, cs, test_x):
                s = 0
                for g_v, a_v, x, prob, out, cec in zip(g, ty, sample, y, m, c):
                    s += x.sum()
                    print "ANS:", a_v,
                    print "GUESS:", g_v,
                    if g_v == a_v:
                        print "    ",
                    else:
                        print "[NG]",
                    print "sum:", "{:+.5f} ({:+.5f})".format(s, float(x.sum())),
                    #print "cec:", cec, "\t",
                    #print "out:", out, "\t",
                    print "prob:", prob[g_v],
                    if prob[g_v] >= 0.8:
                        print ">= 0.8",
                        self.assertTrue(np.isclose(a_v, g_v, atol=0.1))
                    print
                print

        print "TEST SHORT"
        test(test_x_short, test_y_short)
        print "TEST LONG"
        test(test_x_long, test_y_long)


if __name__ == '__main__':
    unittest.main()

