"""
validate_simulation.py

Script for validating simulations — i.e. checking that mean heights grow
in true time. For each ensemble we:

  1. Plot mean height vs time for one replica (diagnostic, saved to
     ``save_dir`` as a PNG).
  2. Compute the log-log slope of mean height vs time on the last 10% of
     each replica's time series (avoiding transients), then average across
     replicas to get a per-mu estimate with standard deviation.
  3. Save the resulting per-mu table (sorted by mu) as ``llslopes.csv`` in
     ``save_dir``.

For mu > 3.0, we expect linear growth. 
"""

from pathlib import Path

from stochastic_growth_true_time_simulation import *
from stochastic_growth_data_analysis import *
from batch_script import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ====================================================

ensemble_names = [
    "ensemble_L8192_mu3p10_t124000M_dtau3_N80",
    "ensemble_L8192_mu3p20_t124000M_dtau3_N80",
    "ensemble_L8192_mu3p30_t124000M_dtau3_N80",
    "ensemble_L16384_mu4p00_t9000M_dtau0.05_N100",
    "ensemble_L8192_mu4p30_t124000M_dtau3_N80",
    "ensemble_L8192_mu5p60_t124000M_dtau3_N80",
    "ensemble_L16384_mu3p50_t9000M_dtau0.05_N100",
    "ensemble_L16384_mu4p50_t9000M_dtau0.05_N100",
    "ensemble_L16384_mu5p00_t9000M_dtau0.05_N100",
    "ensemble_L16384_mu5p50_t9000M_dtau0.05_N100",
    "ensemble_L16384_mu6p00_t9000M_dtau0.05_N100",
]

save_dir = "validation_plots"

# ====================================================


def get_num_range(times, time_range=(0.5,1.0)):
    """Given a sorted array of times, return the indices of the times closest
    to the lower and upper bounds of `time_range`, where each bound is a
    fraction (0.0–1.0) of the interval between the first and last time.

    For example, time_range=(0.5, 1.0) returns the indices of the times
    nearest the midpoint and the end of the time span.
    """

    times = np.array(times)

    min_time = times[0]
    max_time = times[-1]

    lower_time = (max_time - min_time) * time_range[0] + min_time
    upper_time = (max_time - min_time) * time_range[1] + min_time

    # get the closest times to lower_time and upper_time
    lower_close = np.argmin(np.abs(times - lower_time))
    upper_close = np.argmin(np.abs(times - upper_time))
    return lower_close, upper_close


def compute_height_slope(replica):
    """Fit the log-log slope of mean height vs time on the last 10% of the
    time range (to avoid transients). Returns the slope."""
    obs = replica.get_obs()
    mean_height = obs['mean_height_history']
    t = obs['t']
    lower_idx, upper_idx = get_num_range(t, time_range=(0.9, 1.0))
    return fit_loglog_slope(t[lower_idx:upper_idx], mean_height[lower_idx:upper_idx])


def _format_mu(mu):
    return str(mu).replace('.', 'p')


def plot_height(replica, save_dir=None):
    """Plot mean height vs time with a linear fit overlay. Returns the
    last-10% log-log slope."""
    obs = replica.get_obs()
    mean_height = obs['mean_height_history']
    t = obs['t']
    mu = replica.mu

    def model(x, m):
        return m * x + 1  # at t=0 mean height is 1

    llslope = compute_height_slope(replica)
    popt, _ = curve_fit(model, t, mean_height)
    pred = popt[0] * t + 1

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, mean_height, lw=0.0, marker='o', markersize=0.5,
            label=f'data, est log-log slope {llslope}')
    ax.plot(t, pred, lw=1.0, linestyle='--', label="linear fit")
    ax.set_xlabel("true time")
    ax.set_ylabel("mean height")
    ax.set_title(f"height vs time | mu = {mu}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        total_path = (save_dir /
                      f"diagnostic_plot_L{replica.L}_mu{_format_mu(replica.mu)}"
                      f"_t{replica.attempts // 1_000_000}M.png")
        fig.savefig(total_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return llslope


def main():
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load ensembles.
    ensemble_collection = []
    for ensemble_name in ensemble_names:
        replicas, ensemble_meta = batch_load(ensemble_name)
        obs_per_seed = {seed: sim.get_obs() for seed, sim in replicas.items()}
        ensemble_collection.append({
            "name":          ensemble_name,
            "ensemble_meta": ensemble_meta,
            "obs_per_seed":  obs_per_seed,
            "replicas":      replicas,
        })

    # Per-ensemble: save a diagnostic plot from one replica, and average the
    # log-log slope across all replicas (slope is taken from the last 10% of
    # the time range to avoid transients).
    avg_llslopes = {'mu': [], 'avg ll slope': [], 'std ll slope': []}

    for ensemble_dict in ensemble_collection:
        replicas = list(ensemble_dict['replicas'].values())
        plot_height(replicas[0], save_dir=save_dir)

        slopes = [compute_height_slope(r) for r in replicas]
        avg_llslopes['mu'].append(replicas[0].mu)
        avg_llslopes['avg ll slope'].append(np.mean(slopes))
        avg_llslopes['std ll slope'].append(np.std(slopes))

    llslope_df = pd.DataFrame(avg_llslopes).sort_values(by='mu')
    llslope_df.to_csv(save_path / "llslopes.csv", index=False)
    print(llslope_df)


if __name__ == "__main__":
    main()