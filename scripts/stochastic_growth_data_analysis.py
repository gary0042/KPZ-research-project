from __future__ import annotations
from joblib import Parallel, delayed
from scipy.optimize import curve_fit, minimize
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
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

def w_sat(w_l_0, last=20):
    """Returns the approximate saturated value of a curve 
    by averaging over the last s points."""
    ws = w_l_0[-last:]
    l_sat = np.mean(ws)
    l_sat_std = np.std(ws)
    return l_sat, l_sat_std

def powerlaw_model(x, A, B, C):
    return A * (x**B) + C

def linear_model(x, A, B):
    return A*x + B

def BIC_logspaced_1d(x_max, X=None, Y=None):
    x_min = X.min()
    if x_max <= x_min:
        return np.inf
    mask = (X >= x_min) & (X <= x_max)
    x_subset, y_subset = X[mask], Y[mask]
    n = len(x_subset)

    if n <= 3:
        return np.inf

    try:
        params, _ = curve_fit(linear_model, x_subset, y_subset, maxfev=2_000)
    except Exception:
        return np.inf

    rss = np.sum((y_subset - linear_model(x_subset, *params))**2)
    k = 2
    return k * np.log(n) + n * (np.log(2 * np.pi) + 1 + np.log(rss / n))

def grid_search_BIC_logspaced_1d_parallel(X_in, Y_in, x_range=None, stride=1, n_jobs=64):
    X = X_in.copy()
    Y = Y_in.copy()
    if x_range:
        mask = (X >= x_range[0]) & (X <= x_range[1])
        X, Y = X[mask], Y[mask]

    X_ind = X[::stride]
    N = len(X_ind)

    # Only x_max varies; x_min is fixed at X.min()
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(BIC_logspaced_1d)(X_ind[j], X=X, Y=Y)
        for j in range(N)
    )

    BIC_curve = np.array(results)
    j_best = int(np.argmin(BIC_curve))
    best = (X.min(), X_ind[j_best])
    return BIC_curve, X_ind, best

def best_slope_1d(best, X=None, Y=None):
    x_min, x_max = best
    if x_min >= x_max:
        return np.inf
    mask = (X >= x_min) & (X <= x_max)
    x_subset, y_subset = X[mask], Y[mask]
    if len(x_subset) <= 3:
        return np.inf
    try:
        params, _ = curve_fit(linear_model, x_subset, y_subset, maxfev=2_000)
    except Exception:
        return np.inf
    return params[0]


def BIC(x_min, x_max, X=None, Y=None):
    
    if x_min >= x_max: # exclude invalid subsets 
        return np.inf
    mask = (X >= x_min) & (X <= x_max)
    x_subset, y_subset = X[mask], Y[mask]
    n = len(x_subset)

    if n <= 3: # exclude subsets with less than 3 elements
        return np.inf 

    try:
        params, _ = curve_fit(powerlaw_model, x_subset, y_subset, maxfev = 2_000)
    except Exception:
        return np.inf
        
    rss = np.sum((y_subset - powerlaw_model(x_subset, *params))**2)
    k = 3
    return k * np.log(n) + n * (np.log(2 * np.pi) + 1 + np.log(rss / n))

def grid_search_BIC(X_in, Y_in, x_range: Optional[tuple] = None, stride=1):
    X = X_in.copy()
    Y = Y_in.copy()

    if x_range:
        mask = (X >= x_range[0]) & (X <= x_range[1])
        X, Y = X[mask], Y[mask]

    X_ind = X[::stride]
    BIC_grid = np.full((len(X_ind), len(X_ind)), np.inf)

    # double for loop, can be parallelized later
    for i, x_min in enumerate(X_ind):
        for j, x_max in enumerate(X_ind):
            if x_min >= x_max:
                continue
            BIC_grid[i, j] = BIC(x_min, x_max, X=X, Y=Y)

    best_ind = np.unravel_index(np.argmin(BIC_grid), BIC_grid.shape)
    best = (X_ind[best_ind[0]], X_ind[best_ind[1]])
    
    return BIC_grid, X_ind, best

# claude speedup
def grid_search_BIC_parallel(X_in, Y_in, x_range=None, stride=1, n_jobs=64):
    X = X_in.copy()
    Y = Y_in.copy()
    if x_range:
        mask = (X >= x_range[0]) & (X <= x_range[1])
        X, Y = X[mask], Y[mask]

    X_ind = X[::stride]
    N = len(X_ind)

    # Flatten to valid pairs only — avoids ~half the overhead
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(BIC)(X_ind[i], X_ind[j], X=X, Y=Y)
        for i, j in pairs
    )

    BIC_grid = np.full((N, N), np.inf)
    for (i, j), val in zip(pairs, results):
        BIC_grid[i, j] = val

    best_ind = np.unravel_index(np.argmin(BIC_grid), BIC_grid.shape)
    best = (X_ind[best_ind[0]], X_ind[best_ind[1]])
    return BIC_grid, X_ind, best

def best_exponent(best, X=None, Y=None):
    x_min, x_max = best
    if x_min >= x_max: # exclude invalid subsets 
        return np.inf
    mask = (X >= x_min) & (X <= x_max)
    x_subset, y_subset = X[mask], Y[mask]
    n = len(x_subset)

    if n <= 3: # exclude subsets with less than 3 elements
        return np.inf 

    try:
        params, _ = curve_fit(powerlaw_model, x_subset, y_subset, maxfev = 2_000)
    except:
        return np.inf

    return params[1]

def w_sat(w_l_0, last=20):
    """Returns the approximate saturated value of a curve 
    by averaging over the last s points."""
    ws = w_l_0[-last:]
    l_sat = np.mean(ws)
    l_sat_std = np.std(ws)
    return l_sat, l_sat_std