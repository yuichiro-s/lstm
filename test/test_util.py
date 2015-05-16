import unittest

from util import *


class TestUtil(unittest.TestCase):

    def setUp(self):
        pass

    def test_init_weight(self):
        rng = np.random.RandomState(0)

        w1, b1 = init_weight(rng, 2, 3)
        self.assertLess(np.amax(w1),  6. / 5)
        self.assertGreater(np.amin(w1), -6. / 5)
        self.assertLess(np.amax(b1),  6. / 5)
        self.assertGreater(np.amin(b1), -6. / 5)
        self.assertEqual((2, 3), w1.shape)
        self.assertEqual((3,), b1.shape)

        w2, b2 = init_weight(rng, 2, 3, sigmoid=True)
        self.assertLess(np.amax(w2),  24. / 5)
        self.assertGreater(np.amin(w2), -24. / 5)
        self.assertLess(np.amax(b2),  24. / 5)
        self.assertGreater(np.amin(b2), -24. / 5)
        self.assertEqual((2, 3), w2.shape)
        self.assertEqual((3,), b2.shape)


if __name__ == '__main__':
    unittest.main()

