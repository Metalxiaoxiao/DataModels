import numpy as np

def make_lag_matrix(series, p):
    """Create lagged matrix for AR(p). Returns array shape (n-p, p).

    Each row t corresponds to lags [y_{t-1}, y_{t-2}, ..., y_{t-p}].
    """
    y = np.asarray(series)
    n = len(y)
    if p <= 0:
        return np.zeros((n, 0))
    if n <= p:
        return np.empty((0, p))
    mat = np.zeros((n - p, p))
    for i in range(p, n):
        mat[i - p, :] = y[i - 1 : i - p - 1 : -1]
    return mat

def ar_predict(ar_params, lag_matrix, intercept=0.0):
    """Predict AR component given lag matrix and parameters."""
    if lag_matrix.size == 0:
        return np.full((lag_matrix.shape[0],), intercept)
    return intercept + lag_matrix.dot(np.asarray(ar_params))
