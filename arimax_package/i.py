import numpy as np

def difference(series, d=1):
    """Apply differencing of order d to a 1-D array-like series."""
    s = np.asarray(series)
    if d <= 0:
        return s.copy()
    for _ in range(int(d)):
        s = np.diff(s, n=1)
    return s

def inverse_difference(original_last, diffed_series, d=1):
    """Invert differencing given the last original observation.

    original_last: scalar or array-like last observed value before differencing
    diffed_series: 1-D array which is the result of differencing
    d: order used when differencing
    """
    res = np.asarray(diffed_series).copy()
    for _ in range(int(d)):
        res = np.r_[original_last, res].cumsum()
        original_last = res[-1]
    return res
