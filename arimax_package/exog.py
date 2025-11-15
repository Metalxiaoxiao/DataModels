import numpy as np

def prepare_exog(exog):
    """Prepare exogenous matrix. Accepts 1-D or 2-D array-like.

    Returns 2-D numpy array with shape (n_samples, n_exog).
    """
    if exog is None:
        return None
    arr = np.asarray(exog)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr
