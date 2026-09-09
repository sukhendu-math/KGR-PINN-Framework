import numpy as np
import scipy.io
import matplotlib.pyplot as plt


# Load reference solution
data = scipy.io.loadmat("data/AC.mat")

usol = data["uu"]
t_star = data["tt"][0]
x_star = data["x"][0]

# Load predicted solution
u_pred = np.load("results/u_pred.npy")


# Create mesh
TT, XX = np.meshgrid(t_star, x_star)


# Plot results
fig = plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.pcolor(TT, XX, usol, cmap="jet")
plt.colorbar()
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.title(r"Exact $u(x,t)$")

plt.subplot(1, 3, 2)
plt.pcolor(TT, XX, u_pred, cmap="jet")
plt.colorbar()
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.title(r"Predicted $u(x,t)$")

plt.subplot(1, 3, 3)
plt.pcolor(TT, XX, np.abs(usol - u_pred), cmap="jet")
plt.colorbar()
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.title("Absolute error")

plt.tight_layout()
plt.show()