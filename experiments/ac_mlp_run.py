import jax.numpy as np
from jax import random

from src.networks.mlp import MLP
from src.pinn.pinn import PINN
from src.utils.data import load_ac_data


def main():

    # -------------------------
    # Load data
    # -------------------------
    usol, t_star, x_star = load_ac_data("data/AC.mat")

    state0 = usol[:, 0:1]
    reference = usol

    # -------------------------
    # Random key
    # -------------------------
    key = random.PRNGKey(1234)

    # -------------------------
    # Network architecture
    # -------------------------
    M = 10
    d0 = 2 * M + 2

    layers = [
        d0,
        128,
        128,
        128,
        128,
        1
    ]

    # -------------------------
    # PDE parameters
    # -------------------------
    nu = 0.0001

    # -------------------------
    # PINN parameters
    # -------------------------
    t0 = 0.0
    t1 = 1.0

    n_t = 100
    n_x = 256

    tol = 100.0

    # -------------------------
    # Create model
    # -------------------------
    model = PINN(
        key=key,
        layers=layers,

        # Select network
        network=MLP,

        # Parameters specific to MLP
        network_kwargs={
            "L": 2.0,
            "M": M,
            "activation": np.tanh,
        },

        M=M,
        nu=nu,
        state0=state0,
        t0=t0,
        t1=t1,
        n_t=n_t,
        n_x=n_x,
        tol=tol,
        reference=reference,
        t_star=t_star,
        x_star=x_star,
    )

    # -------------------------
    # Train
    # -------------------------
    model.train(nIter=51)
    import jax.numpy as np
from jax import random

from src.networks.modified_mlp import modified_MLP
from src.pinn.pinn import PINN
from src.utils.data import load_ac_data


def main():

    # -------------------------
    # Load data
    # -------------------------
    usol, t_star, x_star = load_ac_data("data/AC.mat")

    state0 = usol[:, 0:1]
    reference = usol

    # -------------------------
    # Random key
    # -------------------------
    key = random.PRNGKey(1234)

    # -------------------------
    # Network architecture
    # -------------------------
    M = 10
    d0 = 2 * M + 2

    layers = [
        d0,
        128,
        128,
        128,
        128,
        1
    ]

    # -------------------------
    # PDE parameters
    # -------------------------
    nu = 0.0001

    # -------------------------
    # PINN parameters
    # -------------------------
    t0 = 0.0
    t1 = 1.0

    n_t = 100
    n_x = 256

    tol = 100.0

    # -------------------------
    # Create model
    # -------------------------
    model = PINN(
        key=key,
        layers=layers,

        # Select Modified MLP
        network=modified_MLP,

        # Parameters specific to Modified MLP
        network_kwargs={
            "L": 2.0,
            "M": M,
            "activation": np.tanh,
        },

        M=M,
        nu=nu,
        state0=state0,
        t0=t0,
        t1=t1,
        n_t=n_t,
        n_x=n_x,
        tol=tol,
        reference=reference,
        t_star=t_star,
        x_star=x_star,
    )

    # -------------------------
    # Train
    # -------------------------
    model.train(nIter=51)
    # Compute prediction
    params = model.params
    u_pred = model.u_pred_fn(params, t_star, x_star)

    # Save results
    np.save("results/loss_log.npy", model.loss_log)
    np.save("results/loss_ics_log.npy", model.loss_ics_log)
    np.save("results/loss_res_log.npy", model.loss_res_log)
    np.save("results/error_data.npy", model.error_data)
    np.save("results/u_pred.npy", u_pred)
    # np.save("results/beta_list.npy", model.beta_log)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()