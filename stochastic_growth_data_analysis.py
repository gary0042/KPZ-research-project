from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time as _time

def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Fit slope in log-log space for positive x, y."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return np.nan
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, _ = np.polyfit(lx, ly, deg=1)
    return float(slope)

def local_slopes(x, y, s=1):
    """
    Calculates the local slopes using a windowed linear regression.
    """
    y_dim = len(y)
    ms = np.zeros(y_dim - s)
    xs = np.zeros(y_dim - s)

    for i in range(y_dim - s):
        #ms[i] = (y[i+s] - y[i]) / (x[i+s] - x[i])
        ms[i], _ = np.polyfit(x[i:i+s+1], y[i:i+s+1], deg=1)
        xs[i] = np.mean(x[i:i+s+1]) # this better reflects the centroid of window if
        # points are not evenly spaced
        # xs[i] = (x[i+s+1] + x[i])/2

    return xs, ms