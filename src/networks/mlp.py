import jax
import jax.numpy as np
from jax import random
def MLP(layers, L=1.0, M=1, activation=np.tanh):

    def input_encoding(t, x):
        w = 2.0 * np.pi / L
        k = np.arange(1, M + 1)
        out = np.hstack([t, 1,
                         np.cos(k * w * x), np.sin(k * w * x)])
        return out

    def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params

    def apply(params, inputs):
        t = inputs[0]
        x = inputs[1]
        H = input_encoding(t, x)
        for W, b in params[:-1]:
            outputs = np.dot(H, W) + b
            H = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(H, W) + b
        return outputs

    return init, apply