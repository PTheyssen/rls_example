import numpy as np


def batch_ls(s, dim, data_x, data_y, ridge_factor=None):
    """
    Batch Least Squared Regression

    s : number of samples
    dim : dimension of the model
    data_x : points at which measurement was observed
    data_y : observed measurements
    ridge_factor : possible ridge factor for the model
    """

    # create H matrix
    h = np.vstack([np.ones(s), data_x]).T

    if ridge_factor is not None:
        m = np.eye(dim)
        m[0, 0] = 0
        h_t_h = h.T @ h + ridge_factor * m
    else:
        h_t_h = h.T @ h

    return np.linalg.solve(h_t_h, h.T @ data_y)


def rls(s, dim, data_x, data_y, f, sigma):
    """
    Recursive Least Squared Regression

    s : number of samples
    dim : dimension of the model
    data_x : points at which measurement was observed
    data_y : observed measurements
    f : forgetting factor usually between 0.95 and 1
    sigma : used to initialize first p matrix
    """
    parameters = []

    # initialize values
    par_prev = np.zeros(dim).reshape(dim,1)  # or random values
    p_prev = np.eye(dim) * sigma

    # create H matrix
    h = np.vstack([np.ones(s), data_x]).T

    # prediction loop
    for i in range(s):
        h_i = h[i, :].reshape(1, dim)
        y_i = data_y[i]

        s = f + h_i @ p_prev @ h_i.T
        k = p_prev @ h_i.T * (s**-1)
        par = par_prev + k * (y_i - h_i @ par_prev)

        par_prev = par
        p_prev = (1/f) * (p_prev - s * k @ k.T)
        parameters.append(par)

    return parameters
