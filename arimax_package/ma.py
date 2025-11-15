import numpy as np

def compute_ma_component(errors, q, thetas):
    """Compute MA contribution for each time step given past errors.

    errors: 1-D array of residuals (assumed aligned so errors[t] is residual at time t)
    q: order of MA
    thetas: array-like of length q
    Returns array of same length as errors where ma[t] = sum_{j=1..q} theta_j * errors[t-j]
    (for indices < 0, errors treated as 0)
    """
    e = np.asarray(errors)
    n = len(e)
    thetas = np.asarray(thetas)
    ma = np.zeros(n)
    for t in range(n):
        acc = 0.0
        for j in range(1, q + 1):
            if t - j >= 0:
                acc += thetas[j - 1] * e[t - j]
        ma[t] = acc
    return ma
