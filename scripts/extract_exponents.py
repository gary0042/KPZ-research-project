"""
Batch ensemble analysis: load ensembles specified by (L, mu, t, dtau, N),
plot ensemble means, optionally extract beta and/or alpha exponents via BIC,
and save all results to a single JSON.

Edit the ENSEMBLES list below to specify which ensembles to process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

from batch_script import batch_load
from stochastic_growth_data_analysis import (
    best_exponent,
    best_slope_1d,
    grid_search_BIC_logspaced_1d_parallel,
    grid_search_BIC_parallel,
    linear_model,
    local_slopes,
    powerlaw_model,
    w_sat,
)

# ---------------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------------

BASE_DIR = Path("/pscratch/sd/g/ghan36")
OUTPUT_DIR = BASE_DIR / "exponent_results3"
N_JOBS = -10  # joblib n_jobs for BIC searches

# Each entry fully specifies an ensemble plus what to extract.
# t is in millions of attempts (the "M" tag from the save name).
# beta_x_range / beta_stride / alpha_stride are optional per-ensemble overrides.

# code to construct ENSEMBLE
base_dict = {
        "L": 16384,
        "N_ensemble": 100,
        "extract_beta": False,
        "extract_alpha": True,
        "beta_x_range": (0.0, 100),
        "beta_stride": 10,
        "beta_ls": [3687, # fit to last 20 l's 
    3845,
    4010,
    4182,
    4362,
    4549,
    4744,
    4948,
    5160,
    5382,
    5612,
    5853,
    6104,
    6366,
    6640,
    6924,
    7221,
    7531,
    7854,
    8192],           # subset of l-values to fit beta on; None → all
        "alpha_stride": 1,
        "w_sat_last": 200,
    }

mus = [3.50, 4.00, 4.50, 5.00, 5.50, 6.0]
t = 3_000_000
dtau = 5

ENSEMBLES = []

for mu in mus:
    temp_dict = base_dict.copy()
    temp_dict["t"] = t
    temp_dict["record_interval_true"] = dtau
    temp_dict["mu"] = mu
    ENSEMBLES.append(temp_dict)

base_dict = {
        "L": 16384,
        "N_ensemble": 100,
        "extract_beta": True,
        "extract_alpha": False,
        "beta_x_range": (0.0, 100),
        "beta_stride": 10,
        "beta_ls": [3687,
    3845,
    4010,
    4182,
    4362,
    4549,
    4744,
    4948,
    5160,
    5382,
    5612,
    5853,
    6104,
    6366,
    6640,
    6924,
    7221,
    7531,
    7854,
    8192],           # subset of l-values to fit beta on; None → all
        "alpha_stride": 1,
        "w_sat_last": 200,
    }

t = 9_000
dtau = 0.05

for mu in mus:
    temp_dict = base_dict.copy()
    temp_dict["t"] = t
    temp_dict["record_interval_true"] = dtau
    temp_dict["mu"] = mu
    ENSEMBLES.append(temp_dict)

# ENSEMBLES = [
#     {
#         "L": 16384,
#         "mu": 3.50,
#         "t": 3_000_000,           # in millions
#         "record_interval_true": 5,
#         "N_ensemble": 100,
#         "extract_beta": False,
#         "extract_alpha": True,
#         "beta_x_range": (0.0, 100),     # (t_lo, t_hi) in true time, or None for all
#         "beta_stride": 10,
#         "beta_l_every": 5,        # run beta on every Nth l-value
#         "alpha_stride": 1,
#         "w_sat_last": 200,        # # of trailing samples used to estimate w_sat
#     },
#     # more ensembles here
# ]

# ---------------------------------------------------------------------------
# Path construction
# ---------------------------------------------------------------------------


def _format_mu(mu: float) -> str:
    return f"{mu:.2f}".replace(".", "p")


def _format_dtau(dtau) -> str:
    # match notebook convention: "5", "0.05", "0.2", "1", etc.
    if isinstance(dtau, int) or float(dtau).is_integer():
        return str(int(dtau))
    return str(dtau)


def ensemble_dir_name(L, mu, t, record_interval_true, N_ensemble) -> str:
    return (
        f"ensemble_L{L}"
        f"_mu{_format_mu(mu)}"
        f"_t{t}M"
        f"_dtau{_format_dtau(record_interval_true)}"
        f"_N{N_ensemble}"
    )


def ensemble_tag(spec) -> str:
    """Short tag used in output filenames."""
    return (
        f"L{spec['L']}_mu{_format_mu(spec['mu'])}"
        f"_t{spec['t']}M_dtau{_format_dtau(spec['record_interval_true'])}"
        f"_N{spec['N_ensemble']}"
    )


# ---------------------------------------------------------------------------
# Ensemble averaging (interpolation onto a common grid — robust to ragged t)
# ---------------------------------------------------------------------------


def average_ensemble(replicas):
    """Returns t_grid, sw_mean (shape (n_grid, n_l))."""
    obs_per_seed = {seed: sim.get_obs() for seed, sim in replicas.items()}

    t_lo = max(obs["t"][0] for obs in obs_per_seed.values())
    t_hi = float(np.median([obs["t"][-1] for obs in obs_per_seed.values()]))
    num = max(len(obs["t"]) for obs in obs_per_seed.values())
    t_grid = np.linspace(t_lo, t_hi - 100, num=num)

    sw_interp = []
    for obs in obs_per_seed.values():
        t_r = obs["t"]
        sw_r = obs["surface_width_history"]  # (n_t_r, n_l)
        cols = [
            np.interp(t_grid, t_r, sw_r[:, i], left=np.nan, right=np.nan)
            for i in range(sw_r.shape[1])
        ]
        sw_interp.append(np.column_stack(cols))

    sw_stack = np.stack(sw_interp)  # (N_rep, n_grid, n_l)
    sw_mean = np.nanmean(sw_stack, axis=0)
    return t_grid, sw_mean, len(obs_per_seed)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_ensemble_mean(t_mean, sw_mean, ls, L, mu, N_ensemble, out_path):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(ls)))
    for i, l_val in enumerate(ls):
        ax.plot(
            t_mean, sw_mean[:, i],
            color=cmap[i], lw=0.5, marker="o", markersize=0.5,
            label=f"$l$={l_val}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"true time $\tau$")
    ax.set_ylabel(r"$\langle w(l,\tau) \rangle$")
    ax.set_title(
        rf"Ensemble mean $w(l,\tau)$  |  $L={L}$, $\mu={mu}$, $N={N_ensemble}$ replicas"
    )
    ax.legend(fontsize=6, ncol=4, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.minorticks_on()
    ax.grid(True)
    ax.grid(which="minor", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bic_beta(BIC_grid, X, best, beta, l_val, L, mu, N_ensemble, out_path):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
    pcm = ax.pcolormesh(X, X, BIC_grid.T)
    fig.colorbar(pcm, ax=ax, label="BIC")
    ax.set_xlabel(r"$\tau_\mathrm{min}$")
    ax.set_ylabel(r"$\tau_\mathrm{max}$")
    ax.set_title(
        rf"BIC landscape ($\beta$) | $L={L}$, $\mu={mu}$, $N={N_ensemble}$" + "\n"
        + rf"$l$={l_val}, growth=({best[0]:.2f}, {best[1]:.2f}), $\beta$={beta:.3f}"
    )
    ax.minorticks_on()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bic_alpha(BIC_curve, X, best, alpha, L, mu, N_ensemble, out_path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.plot(X, BIC_curve, marker="o", markersize=3, lw=0.8)
    ax.axvline(best[1], color="r", ls="--", lw=0.8,
               label=fr"$\log l_\mathrm{{max}}^*$ = {best[1]:.2f}")
    ax.set_xlabel(r"$\log l_\mathrm{max}$")
    ax.set_ylabel("BIC")
    ax.set_title(
        rf"BIC curve ($\alpha$) | $L={L}$, $\mu={mu}$, $N={N_ensemble}$" + "\n"
        + rf"$l$ range=({np.exp(best[0]):.2f}, {np.exp(best[1]):.2f}), $\alpha$={alpha:.3f}"
    )
    ax.legend()
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def refit_powerlaw(best, X, Y):
    """Refit y = A*x^B + C over [x_min, x_max] and return (A, B, C) or None."""
    x_min, x_max = best
    mask = (X >= x_min) & (X <= x_max)
    if mask.sum() <= 3:
        return None
    try:
        params, _ = curve_fit(powerlaw_model, X[mask], Y[mask], maxfev=2_000)
    except Exception:
        return None
    return tuple(float(p) for p in params)


def refit_linear(best, X, Y):
    """Refit y = A*x + B over [x_min, x_max] (log-log).
    Returns (A, B, sigma_A, sigma_B) or None."""
    x_min, x_max = best
    mask = (X >= x_min) & (X <= x_max)
    if mask.sum() <= 3:
        return None
    try:
        params, pcov = curve_fit(linear_model, X[mask], Y[mask], maxfev=2_000)
    except Exception:
        return None
    perr = np.sqrt(np.diag(pcov))
    return (float(params[0]), float(params[1]),
            float(perr[0]), float(perr[1]))


def plot_beta_overlay(t_mean, sw_mean, beta_fits, L, mu, N_ensemble, out_path):
    """
    beta_fits: list of dicts with keys {l_idx, l_val, A, B, C, t_min, t_max}
    Plots w(l,t) for each fit l together with dashed power-law overlay.
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    cmap = plt.cm.viridis(np.linspace(0, 1, max(len(beta_fits), 1)))
    # full t range across the ensemble mean (positive only, for log scale)
    t_pos = t_mean[t_mean > 0]
    t_lo, t_hi = float(t_pos.min()), float(t_pos.max())
    t_fit = np.logspace(np.log10(t_lo), np.log10(t_hi), 400)
    for k, fit in enumerate(beta_fits):
        color = cmap[k]
        i = fit["l_idx"]
        ax.plot(
            t_mean, sw_mean[:, i],
            color=color, lw=0.0, marker="o", markersize=1.0,
            label=fr"$l$={fit['l_val']}, $\beta$={fit['B']:.3f}",
        )
        # dashed fit extrapolated over the full data range
        y_fit = powerlaw_model(t_fit, fit["A"], fit["B"], fit["C"])
        ax.plot(t_fit, y_fit, color=color, ls="--", lw=1.0, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"true time $\tau$")
    ax.set_ylabel(r"$\langle w(l,\tau)\rangle$")
    ax.set_title(
        rf"$\beta$ fits | $L={L}$, $\mu={mu}$, $N={N_ensemble}$"
        + "\n(dashed: $A\\tau^\\beta + C$ over BIC window)"
    )
    ax.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.minorticks_on()
    ax.grid(True)
    ax.grid(which="minor", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_alpha_overlay(ls, L_saturate, alpha_fit, L, mu, N_ensemble, out_path):
    """alpha_fit: dict with keys {alpha, log_intercept, l_min, l_max}."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.plot(ls, L_saturate, lw=0.5, marker="o", markersize=3,
            label=r"$w_\mathrm{sat}(l)$")
    if alpha_fit is not None:
        ls_arr = np.asarray(ls, dtype=float)
        l_lo, l_hi = float(ls_arr.min()), float(ls_arr.max())
        l_fit = np.logspace(np.log10(l_lo), np.log10(l_hi), 400)
        y_fit = np.exp(alpha_fit["log_intercept"]) * l_fit ** alpha_fit["alpha"]
        ax.plot(l_fit, y_fit, color="red", ls="--", lw=1.5,
                label=fr"fit: $\alpha$={alpha_fit['alpha']:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$w_\mathrm{sat}(l)$")
    ax.set_title(rf"$\alpha$ fit | $L={L}$, $\mu={mu}$, $N={N_ensemble}$")
    ax.legend()
    ax.minorticks_on()
    ax.grid(True)
    ax.grid(which="minor", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_alpha_diagnostic(ls, L_saturate, L, mu, N_ensemble, out_path, s=10):
    xs, ms = local_slopes(np.log(ls), np.log(L_saturate), s=s)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), dpi=200, sharex=True)
    ax[0].plot(ls, L_saturate, lw=0.5, marker="o", markersize=2)
    ax[0].set_ylabel(r"$w_\mathrm{sat}(l)$")
    ax[0].set_title(rf"$w_\mathrm{{sat}}$ vs $l$ | $L={L}$, $\mu={mu}$, $N={N_ensemble}$")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].grid(True)
    ax[0].grid(which="minor", alpha=0.2)
    ax[0].minorticks_on()

    ax[1].plot(np.exp(xs), ms, lw=0.5, marker="o", markersize=2)
    ax[1].set_xlabel(r"$l$")
    ax[1].set_ylabel("local log-log slope")
    ax[1].set_xscale("log")
    ax[1].grid(True)
    ax[1].grid(which="minor", alpha=0.2)
    ax[1].minorticks_on()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-ensemble pipeline
# ---------------------------------------------------------------------------


def process_ensemble(spec, output_dir: Path):
    L = spec["L"]
    mu = spec["mu"]
    tag = ensemble_tag(spec)
    ens_dir = BASE_DIR / ensemble_dir_name(
        L, mu, spec["t"], spec["record_interval_true"], spec["N_ensemble"]
    )

    print(f"\n[{tag}] loading {ens_dir}")
    if not ens_dir.is_dir():
        print(f"  ! ensemble dir not found, skipping")
        return {"tag": tag, "error": "ensemble dir not found", "path": str(ens_dir)}

    replicas, ensemble_meta = batch_load(str(ens_dir))
    ls = list(ensemble_meta["ls"])
    N_ensemble = len(replicas)
    print(f"  loaded {N_ensemble} replicas; n_l={len(ls)}")

    t_mean, sw_mean, _ = average_ensemble(replicas)

    # 1. ensemble mean plot
    mean_plot = output_dir / f"mean_{tag}.png"
    plot_ensemble_mean(t_mean, sw_mean, ls, L, mu, N_ensemble, mean_plot)
    print(f"  saved {mean_plot.name}")

    result = {
        "tag": tag,
        "path": str(ens_dir),
        "L": L,
        "mu": mu,
        "t": spec["t"],
        "record_interval_true": spec["record_interval_true"],
        "N_ensemble": N_ensemble,
        "ls": ls,
        "beta": None,
        "alpha": None,
    }

    # 2. beta extraction (on user-specified subset of l)
    if spec.get("extract_beta", False):
        beta_x_range = spec.get("beta_x_range", None)
        beta_stride = spec.get("beta_stride", 10)
        beta_ls = spec.get("beta_ls", None)
        result["beta"] = {}

        if beta_ls is None:
            beta_indices = list(range(len(ls)))
        else:
            # Map each requested l to the nearest available l in the ensemble.
            ls_arr = np.asarray(ls)
            beta_indices = []
            for l_req in beta_ls:
                idx = int(np.argmin(np.abs(ls_arr - l_req)))
                if idx not in beta_indices:
                    beta_indices.append(idx)
            missing = [l for l in beta_ls if l not in [ls[i] for i in beta_indices]]
            if missing:
                print(f"  [beta] note: requested l={missing} snapped to nearest "
                      f"available values")

        print(f"  [beta] running on {len(beta_indices)}/{len(ls)} l-values: "
              f"{[ls[i] for i in beta_indices]}")

        beta_fits = []  # for overlay plot
        for l_idx in beta_indices:
            l_val = ls[l_idx]
            X_in = t_mean.copy()
            Y_in = sw_mean[:, l_idx].copy()
            # drop NaN tail from interpolation
            good = np.isfinite(X_in) & np.isfinite(Y_in)
            X_in, Y_in = X_in[good], Y_in[good]

            print(f"    l={l_val} (idx {l_idx}/{len(ls)})")
            try:
                BIC_grid, X, best = grid_search_BIC_parallel(
                    X_in, Y_in,
                    x_range=beta_x_range, stride=beta_stride, n_jobs=N_JOBS,
                )
                beta_val = float(best_exponent(best, X_in, Y_in))
            except Exception as e:
                print(f"    ! beta fit failed for l={l_val}: {e}")
                result["beta"][str(l_val)] = {"error": str(e)}
                continue

            plot_path = output_dir / f"bic_beta_{tag}_l{l_val}.png"
            plot_bic_beta(BIC_grid, X, best, beta_val, l_val,
                          L, mu, N_ensemble, plot_path)

            # refit to get full (A, B, C) for the overlay
            params = refit_powerlaw(best, X_in, Y_in)
            entry = {
                "beta": beta_val,
                "t_min": float(best[0]),
                "t_max": float(best[1]),
            }
            if params is not None:
                A_fit, B_fit, C_fit = params
                entry.update({"A": A_fit, "B_refit": B_fit, "C": C_fit})
                beta_fits.append({
                    "l_idx": l_idx, "l_val": l_val,
                    "A": A_fit, "B": B_fit, "C": C_fit,
                    "t_min": float(best[0]), "t_max": float(best[1]),
                })
            result["beta"][str(l_val)] = entry
            print(f"    beta={beta_val:.4f}, range=({best[0]:.2f},{best[1]:.2f})")

        if beta_fits:
            overlay_path = output_dir / f"beta_overlay_{tag}.png"
            plot_beta_overlay(t_mean, sw_mean, beta_fits,
                              L, mu, N_ensemble, overlay_path)
            print(f"  saved {overlay_path.name}")

    # 3. alpha extraction
    if spec.get("extract_alpha", False):
        last = spec.get("w_sat_last", 200)
        L_saturate = np.array([w_sat(sw_mean[:, i], last)[0] for i in range(len(ls))])

        diag_path = output_dir / f"alpha_diagnostic_{tag}.png"
        plot_alpha_diagnostic(np.array(ls), L_saturate, L, mu, N_ensemble, diag_path)
        print(f"  saved {diag_path.name}")

        X_in = np.log(np.array(ls, dtype=float))
        Y_in = np.log(L_saturate.astype(float))
        try:
            BIC_curve, X, best = grid_search_BIC_logspaced_1d_parallel(
                X_in, Y_in,
                x_range=None, stride=spec.get("alpha_stride", 1), n_jobs=N_JOBS,
            )
            alpha_val = float(best_slope_1d(best, X_in, Y_in))
            plot_path = output_dir / f"bic_alpha_{tag}.png"
            plot_bic_alpha(BIC_curve, X, best, alpha_val,
                           L, mu, N_ensemble, plot_path)

            # refit linear in log-log to get the intercept (for overlay)
            # and uncertainty in alpha (= slope) from the covariance matrix.
            lin_params = refit_linear(best, X_in, Y_in)
            if lin_params is not None:
                _, log_intercept, sigma_alpha, sigma_intercept = lin_params
            else:
                log_intercept = None
                sigma_alpha = None
                sigma_intercept = None

            alpha_entry = {
                "alpha": alpha_val,
                "alpha_uncertainty": sigma_alpha,
                "log_l_min": float(best[0]),
                "log_l_max": float(best[1]),
                "l_min": float(np.exp(best[0])),
                "l_max": float(np.exp(best[1])),
                "log_intercept": log_intercept,
                "log_intercept_uncertainty": sigma_intercept,
            }
            result["alpha"] = alpha_entry

            if log_intercept is not None:
                alpha_fit = {
                    "alpha": alpha_val,
                    "log_intercept": log_intercept,
                    "l_min": float(np.exp(best[0])),
                    "l_max": float(np.exp(best[1])),
                }
                overlay_path = output_dir / f"alpha_overlay_{tag}.png"
                plot_alpha_overlay(np.array(ls), L_saturate, alpha_fit,
                                   L, mu, N_ensemble, overlay_path)
                print(f"  saved {overlay_path.name}")

            sig_str = f" ± {sigma_alpha:.4f}" if sigma_alpha is not None else ""
            print(f"  alpha={alpha_val:.4f}{sig_str}, "
                  f"l-range=({np.exp(best[0]):.2f},{np.exp(best[1]):.2f})")
        except Exception as e:
            print(f"  ! alpha fit failed: {e}")
            result["alpha"] = {"error": str(e)}

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for spec in ENSEMBLES:
        try:
            res = process_ensemble(spec, OUTPUT_DIR)
        except Exception as e:
            res = {"tag": ensemble_tag(spec), "error": str(e)}
            print(f"  ! ensemble failed: {e}")
        all_results.append(res)

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
