import itertools
from functools import partial

import jax.numpy as np
from jax import grad, vmap, jit, lax

from jax.flatten_util import ravel_pytree


import optax
from tqdm import trange
# Define the model
class PINN:
    def __init__(
    self,
    key,
    layers,
    network,
    network_kwargs,
    M,
    nu,
    state0,
    t0,
    t1,
    n_t,
    n_x,
    tol,
    reference,
    t_star,
    x_star
):

        # collocation points
        self.t0 = t0
        self.t1 = t1
        self.t_r   = np.linspace(self.t0, self.t1, n_t)
        self.x_r = np.linspace(-1, 1, n_x)

        # For computing the temporal weights
        self.M = np.triu(np.ones((n_t, n_t)), k=1).T
        self.tol = tol

        self.reference = reference
        self.t_star = t_star
        self.x_star = x_star

        # IC
        t_ic = np.zeros((x_star.shape[0], 1))
        x_ic = x_star.reshape(-1, 1)
        self.X_ic = np.hstack([t_ic, x_ic])
        self.Y_ic = state0
        self.nu = nu

        
        self.init, self.apply = network(
    layers,
    **network_kwargs
)

        # Initialize the parameters
        params = self.init(rng_key = key)
        _, self.unravel = ravel_pytree(params)
        self.params = params


        # Use optimizers to set optimizer initialization and update functions
        # lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
        # self.opt_init, \
        # self.opt_update, \
        # self.get_params = optimizers.adam(lr)
        # self.opt_state = self.opt_init(params)


        # === Hyperparameters (taken from your config) ===
        learning_rate = 1e-3
        decay_rate = 0.9
        decay_steps = 2000

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        # === Learning rate schedule (exponential decay) ===
        lr_schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=decay_steps,
            decay_rate=decay_rate,
        )

        # === Optimizer (AGC + Adam) ===
        tx = optax.chain(
            optax.adaptive_grad_clip(1e-2),  # same as your example
            optax.adam(
                learning_rate=lr_schedule,
                b1=beta1,
                b2=beta2,
                eps=eps,
            ),
        )

        # === Initialize ===
        self.tx = tx
        self.opt_state = self.tx.init(self.params)

        # Evaluate the network and the residual over the grid
        self.u_pred_fn = vmap(vmap(self.neural_net, (None, 0, None)), (None, None, 0))  # consistent with the dataset
        self.r_pred_fn = vmap(vmap(self.residual_net, (None, None, 0)), (None, 0, None))

        # Logger
        self.itercount = itertools.count()

        self.loss_log = []
        self.loss_ics_log = []
        self.loss_res_log = []
        self.W_log = []
        self.L_t_log = []
        self.error_data = []

    def neural_net(self, params, t, x):
        z = np.stack([t, x])
        outputs = self.apply(params, z)
        return outputs[0]

    def residual_net(self, params, t, x):
        u = self.neural_net(params, t, x)
        u_t = grad(self.neural_net, argnums=1)(params, t, x)
        u_x = grad(self.neural_net, argnums=2)(params, t, x)
        u_xx = grad(grad(self.neural_net, argnums = 2), argnums=2)(params, t, x)
        return u_t + 5 * u**3 - 5 * u - self.nu * u_xx

    # def residual_net(self, params, t, x):
    #     u = self.neural_net(params, t, x)
    #     u_t = grad(self.neural_net, argnums=1)(params, t, x)
    #     u_fn = lambda x: self.neural_net(params, t, x) # For using Taylor-mode AD
    #     _, (u_x, u_xx) = jet(u_fn, (x, ), [[1.0, 0.0]]) #  Taylor-mode AD
    #     return u_t + 5 * u**3 - 5 * u - self.nu * u_xx

    @partial(jit, static_argnums=(0,))
    def residuals_and_weights(self, params, tol):
        r_pred = self.r_pred_fn(params, self.t_r, self.x_r)
        L_t = np.mean(r_pred**2, axis=1)
        W = lax.stop_gradient(np.exp(- tol * (self.M @ L_t)))
        return L_t, W


    @partial(jit, static_argnums=(0,))
    def loss_ics(self, params):
        # Evaluate the network over IC
        u_pred = vmap(self.neural_net, (None, 0, 0))(params, self.X_ic[:,0], self.X_ic[:,1])
        # Compute the initial loss
        loss_ics = np.mean((self.Y_ic.flatten() - u_pred.flatten())**2)
        return loss_ics

    @partial(jit, static_argnums=(0,))
    def loss_res(self, params):
        r_pred = self.r_pred_fn(params, self.t_r, self.x_r)
        # Compute loss
        loss_r = np.mean(r_pred**2)
        return loss_r

    @partial(jit, static_argnums=(0,))
    def loss(self, params):
        L0 = 100 * self.loss_ics(params)
        L_t, W = self.residuals_and_weights(params, self.tol)
        # Compute loss
        loss = np.mean(W * L_t) + L0
        return loss

    # Define a compiled update step
    # @partial(jit, static_argnums=(0,))
    # def step(self, i, opt_state):
    #     params = self.get_params(opt_state)
    #     g = grad(self.loss)(params)

    #     return self.opt_update(i, g, opt_state)
    @partial(jit, static_argnums=(0,))
    def step(self, i, params, opt_state):
        grads = grad(self.loss)(params)

        updates, opt_state = self.tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state

    # Optimize parameters in a loop
    def train(self, nIter = 10000):
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            self.current_count = next(self.itercount)
            # self.opt_state = self.step(self.current_count, self.opt_state)
            self.params, self.opt_state = self.step(
                                                  self.current_count,
                                                  self.params,
                                                  self.opt_state
                                              )

            if it % 1000 == 0:
                # params = self.get_params(self.opt_state)
                params = self.params

                loss_value = self.loss(params)
                loss_ics_value = self.loss_ics(params)
                loss_res_value = self.loss_res(params)
                L_t_value, W_value = self.residuals_and_weights(params, self.tol)


                u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
                error = np.linalg.norm(u_pred - self.reference) / np.linalg.norm(self.reference)


                self.loss_log.append(loss_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_res_log.append(loss_res_value)
                self.W_log.append(W_value)
                self.L_t_log.append(L_t_value)
                self.error_data.append(error)

                pbar.set_postfix({'Loss': loss_value,
                                  'loss_ics' : loss_ics_value,
                                  'loss_res':  loss_res_value})
