import scipy.io


def load_ac_data(path):
    data = scipy.io.loadmat(path)

    usol = data["uu"]
    t_star = data["tt"][0]
    x_star = data["x"][0]

    return usol, t_star, x_star