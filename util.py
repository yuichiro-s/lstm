import theano
import numpy as np

def init_weight(rng, n_in, n_out, sigmoid=False):
    # weight
    w = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (n_in+n_out)),
            high=np.sqrt(6. / (n_in+n_out)),
            size=(n_in, n_out),
            ),
        dtype="float32"
    )

    # bias
    b = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (n_in+n_out)),
            high=np.sqrt(6. / (n_in+n_out)),
            size=n_out,
            ),
        dtype="float32"
    )

    if sigmoid:
        w *= 4
        b *= 4

    return w, b