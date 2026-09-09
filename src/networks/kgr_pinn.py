import jax
import jax.numpy as np
from jax import random
def kgr_PINN(layers, L_blocks=3, L=1.0, M=1,
                         activation=np.tanh,
                         Nc=64):

    def xavier_init(key, d_in, d_out):
        std = 1. / np.sqrt((d_in + d_out) / 2.)
        W = std * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b

    # -------------------------
    # Encoding (PDE input)
    # -------------------------
    def input_encoding(t, x):
        w = 2 * np.pi / L
        k = np.arange(1, M + 1)
        return np.hstack([
            t,
            1,
            np.cos(k * w * x),
            np.sin(k * w * x)
        ])

    # -------------------------
    # RBF (scalar-based like your working model)
    # -------------------------
    def rbf(x, centers, sigma):
        x = x.reshape(-1, 1)                # (d,1)
        diff = x - centers.T               # (d, Nc)
        return np.exp(-(diff**2) / (2.0 * (sigma.T**2 + 1e-6)))

    # -------------------------
    # INIT
    # -------------------------
    def init(rng_key):

        keys = random.split(rng_key, 200)

        input_dim = 2 + 2*M
        hidden_dim = layers[1]

        # input lift
        W0, b0 = xavier_init(keys[0], input_dim, hidden_dim)
        # W0 = random.normal(keys[0], (input_dim, hidden_dim))
        W0 = jax.lax.stop_gradient(W0)



        # global gates (IMPORTANT: use lifted h)
        WU, bU = xavier_init(keys[1], hidden_dim, hidden_dim)
        WV, bV = xavier_init(keys[2], hidden_dim, hidden_dim)

        # -------------------------
        # RBF (computed ONCE like working model)
        # -------------------------
        centers = random.uniform(keys[3], (Nc, 1), minval=0.0, maxval=2.0)
        sigma = np.ones((Nc, 1)) * 0.5
        W_rbf = random.normal(keys[4], (Nc, hidden_dim)) * 0.1

        beta = np.array([0.00])   # RBF strength

        # -------------------------
        # Blocks
        # -------------------------
        blocks = []
        for i in range(L_blocks):
            W1, b1 = xavier_init(keys[5 + i], hidden_dim, hidden_dim)
            W2, b2 = xavier_init(keys[20 + i], hidden_dim, hidden_dim)

            alpha = np.array([0.00])   # residual gate

            blocks.append((W1, b1, W2, b2, alpha))

        # output
        W_out, b_out = xavier_init(keys[150], hidden_dim, layers[-1])

        return (W0, b0,
                WU, bU, WV, bV,
                centers, sigma, W_rbf, beta,
                blocks,
                W_out, b_out)

    # -------------------------
    # APPLY
    # -------------------------
    def apply(params, inputs):

        (W0, b0,
         WU, bU, WV, bV,
         centers, sigma, W_rbf, beta,
         blocks,
         W_out, b_out) = params

        t = inputs[0]
        x = inputs[1]

        # encoding
        enc = input_encoding(t, x)   # (input_dim,)

        # lift
        # h = activation(np.dot(enc, W0) + b0)
        h = activation(np.dot(enc, W0))
        # h = enc

        # global features (use h, not raw input!)
        U = activation(np.dot(h, WU) + bU)
        V = activation(np.dot(h, WV) + bV)

        # -------------------------
        # RBF computed ONCE from encoded input (stable)
        # -------------------------
        Phi = rbf(enc, centers, sigma)     # (input_dim, Nc)
        Phi = np.mean(Phi, axis=0)         # (Nc,) ← reduce dimension
        R = np.dot(Phi, W_rbf)             # (hidden_dim,)

        # -------------------------
        # Blocks
        # -------------------------
        for (W1, b1, W2, b2, alpha) in blocks:

            identity = h

            # ---- f ----
            f = activation(np.dot(h, W1) + b1)
            # f = f + beta * R
            # ---- z (Pirate mixing) ----
            z = f * U + (1 - f) * V
            # h_new = activation(z)
            # ---- Inject RBF (like your working model) ----
            # z = z + beta * R
            # h_new = activation(z)
            # ---- h_new ----
            # phi = 1.0 / (1.0 + np.exp(-beta / 1.0))   # tau = 1.0 (you can tune)
            # z = activation(phi * (beta*R) + (1.0 - phi) * z)
            h_new = activation(np.dot(z, W2) + b2)
            # h_new = activation(h_new + beta * R)
                        # ---- residual (sigmoid gate) ----
            # phi = 1.0 / (1.0 + np.exp(-beta / 1.0))   # tau = 1.0 (you can tune)
            # h_new = activation(phi * (beta * R) + (1.0 - phi) * h_new)
            # a = 1.0 / (1.0 + np.exp(-alpha))
            # h_new = a * h_new + (1 - a) * identity
            # phi = 1.0 / (1.0 + np.exp(-beta / 1.0))   # tau = 1.0 (you can tune)
            # h_new = activation(phi * (beta*R) + (1.0 - phi) * h_new)
            phi = 1.0 / (1.0 + np.exp(-beta / 1.0))   # tau = 1.0 (you can tune)
            h = activation((1-phi) * (beta*R) + (phi) * h_new)
            # h = h_new
        # output
        phi = 1.0 / (1.0 + np.exp(-beta / 1.0))   # tau = 1.0 (you can tune)
        h = activation((1-phi) * (beta*R) + (phi) * h)
        out = np.dot(h, W_out) + b_out
        # phi = 1.0 / (1.0 + np.exp(-beta / 1.0))   # tau = 1.0 (you can tune)
        # out = phi * (beta*R) + (1.0 - phi) * out

        return out

    return init, apply