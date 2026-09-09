import numpy as np
import scipy.io


# Load reference solution
data = scipy.io.loadmat("data/AC.mat")

usol = data["uu"]

# Load predicted solution
u_pred = np.load("results/u_pred.npy")


# Relative L2 error
error = np.linalg.norm(u_pred - usol) / np.linalg.norm(usol)

print("Relative L2 error: {:.3e}".format(error))