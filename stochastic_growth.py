from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def sample_power_law_jump(mu: float, rng: np.random.Generator, min_jump: float = 1.0) -> float:
    """
    Sample jump distance xi from:
        J(xi, mu) = mu * xi^(-(mu + 1)),  xi >= min_jump
    via inverse transform sampling.
    """
    if mu <= 0:
        raise ValueError("mu must be > 0 for a normalizable power-law kernel.")
    u = rng.random()
    return min_jump * (1.0 - u) ** (-1.0 / mu)


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Fit slope in log-log space for positive x, y."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return np.nan
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, _ = np.polyfit(lx, ly, deg=1)
    return float(slope)


@dataclass
class ExponentResults:
    beta: float
    alpha: float
    z: float
    z_from_alpha_beta: float


class StochasticGrowth2Pop:
    """
    Long-range stochastic growth on an L x L lattice with two populations:
      - A_1 encoded as +1
      - A_2 encoded as +2
      - empty encoded as 0

    Initialization:
      Bottom row (y=0) filled with sources.
      x in [0, L/2) -> A_1
      x in [L/2, L) -> A_2

    Update rule (one attempted event per step):
      1) Sample occupied source site uniformly at random.
      2) Sample jump distance from power-law kernel and angle uniformly.
      3) Jump to nearest lattice point; if inside bounds and empty, fill with source type.
    """

    def __init__(self, L: int, mu: float, seed: Optional[int] = None):
        if L < 2:
            raise ValueError("L must be >= 2.")
        self.L = int(L)
        self.mu = float(mu)
        self.rng = np.random.default_rng(seed)

        self.grid = np.zeros((self.L, self.L), dtype=np.int8)
        self._init_sources()

        self.occupied_sites: List[Tuple[int, int]] = []
        ys, xs = np.where(self.grid > 0)
        for y, x in zip(ys, xs):
            self.occupied_sites.append((int(x), int(y)))

        self.time = 0
        self.attempts = 0
        self.accepted = 0
        self.history_t: List[int] = []
        self.history_w: List[float] = []
        self.history_fill_fraction: List[float] = []

    def _init_sources(self) -> None:
        half = self.L // 2
        self.grid[0, :half] = 1
        self.grid[0, half:] = 2

    def step(self) -> bool:
        """Run one attempted growth step. Returns True iff a new site was filled."""
        self.attempts += 1
        idx = self.rng.integers(0, len(self.occupied_sites))
        sx, sy = self.occupied_sites[int(idx)]
        pop = int(self.grid[sy, sx])

        jump = sample_power_law_jump(self.mu, self.rng, min_jump=1.0)
        theta = 2.0 * np.pi * self.rng.random()
        tx = int(round(sx + jump * np.cos(theta))) 
        ty = int(round(sy + jump * np.sin(theta)))

        self.time += 1

        if tx < 0 or tx >= self.L or ty < 0 or ty >= self.L:
            return False
        if self.grid[ty, tx] != 0:
            return False

        self.grid[ty, tx] = pop
        self.occupied_sites.append((tx, ty))
        self.accepted += 1
        return True

    def interface_height(self) -> np.ndarray:
        """
        Interface profile h(x): highest occupied y in each column.
        If column empty (should not occur initially), use 0.
        """
        h = np.zeros(self.L, dtype=np.float64)
        for x in range(self.L):
            ys = np.where(self.grid[:, x] > 0)[0]
            h[x] = ys.max() if ys.size else 0.0
        return h

    def width(self) -> float:
        """Interface width w(L,t) = std_x(h(x,t))."""
        h = self.interface_height()
        return float(np.std(h))

    def fill_fraction(self) -> float:
        return float(np.count_nonzero(self.grid) / (self.L * self.L))

    def _save_snapshot(self, out_dir: Path, t: int, save_png: bool = False) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / f"snapshot_t{t:08d}.npy"
        np.save(npy_path, self.grid)

        if save_png:
            try:
                import matplotlib.pyplot as plt  # local import; optional dependency
            except Exception:
                return
            fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
            cmap = plt.matplotlib.colors.ListedColormap(["white", "#1f77b4", "#d62728"])
            ax.imshow(self.grid, origin="lower", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
            ax.set_title(f"L={self.L}, mu={self.mu}, t={t}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.tight_layout()
            fig.savefig(out_dir / f"snapshot_t{t:08d}.png")
            plt.close(fig)

    def run(
        self,
        n_steps: int,
        record_every: int = 1,
        snapshot_steps: Optional[Iterable[int]] = None,
        snapshot_dir: Optional[str] = None,
        save_png_snapshots: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Run n_steps attempted updates.
        - record_every: sample observables every this many steps
        - snapshot_steps: timesteps at which to save lattice state
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if record_every <= 0:
            raise ValueError("record_every must be positive.")

        snapshot_set = set(snapshot_steps or [])
        out_dir = Path(snapshot_dir) if snapshot_dir else None

        for _ in range(n_steps):
            self.step()
            if self.time % record_every == 0:
                self.history_t.append(self.time)
                self.history_w.append(self.width())
                self.history_fill_fraction.append(self.fill_fraction())
            if self.time in snapshot_set and out_dir is not None:
                self._save_snapshot(out_dir, self.time, save_png=save_png_snapshots)

        return {
            "t": np.asarray(self.history_t, dtype=np.float64),
            "w": np.asarray(self.history_w, dtype=np.float64),
            "fill_fraction": np.asarray(self.history_fill_fraction, dtype=np.float64),
        }


def estimate_beta(t: np.ndarray, w: np.ndarray, early_fraction: float = 0.3) -> float:
    """Estimate growth exponent beta from early-time w ~ t^beta."""
    if not (0.0 < early_fraction <= 1.0):
        raise ValueError("early_fraction must be in (0, 1].")
    n = len(t)
    m = max(2, int(np.ceil(n * early_fraction)))
    return fit_loglog_slope(t[:m], w[:m])


def estimate_saturation_time(t: np.ndarray, w: np.ndarray, frac_of_plateau: float = 0.95) -> float:
    """
    Estimate saturation time t* as first time w(t) reaches given fraction of
    late-time average plateau.
    """
    if len(t) == 0:
        return np.nan
    tail_n = max(3, len(w) // 5)
    plateau = float(np.mean(w[-tail_n:]))
    target = frac_of_plateau * plateau
    idx = np.where(w >= target)[0]
    if idx.size == 0:
        return np.nan
    return float(t[int(idx[0])])


def simulate_sizes_for_exponents(
    sizes: Sequence[int],
    mu: float,
    n_steps: int,
    runs_per_size: int = 8,
    seed: int = 0,
    record_every: int = 10,
) -> Tuple[ExponentResults, Dict[int, Dict[str, float]]]:
    """
    Finite-size scaling workflow:
      - beta from early-time growth on largest L
      - alpha from w*(L) ~ L^alpha
      - z from t*(L) ~ L^z
    """
    sizes = sorted(int(s) for s in sizes)
    if any(s < 2 for s in sizes):
        raise ValueError("All sizes must be >= 2.")

    rng = np.random.default_rng(seed)
    per_size: Dict[int, Dict[str, float]] = {}
    beta_samples: List[float] = []

    for L in sizes:
        w_plateaus = []
        t_stars = []
        for _ in range(runs_per_size):
            sim_seed = int(rng.integers(0, 2**32 - 1))
            sim = StochasticGrowth2Pop(L=L, mu=mu, seed=sim_seed)
            obs = sim.run(n_steps=n_steps, record_every=record_every)
            t = obs["t"]
            w = obs["w"]
            if L == sizes[-1]:
                beta_samples.append(estimate_beta(t, w, early_fraction=0.3))
            tail_n = max(3, len(w) // 5)
            w_plateaus.append(float(np.mean(w[-tail_n:])))
            t_stars.append(estimate_saturation_time(t, w, frac_of_plateau=0.95))

        per_size[L] = {
            "w_star_mean": float(np.nanmean(w_plateaus)),
            "w_star_std": float(np.nanstd(w_plateaus)),
            "t_star_mean": float(np.nanmean(t_stars)),
            "t_star_std": float(np.nanstd(t_stars)),
        }

    Ls = np.asarray(list(per_size.keys()), dtype=np.float64)
    w_stars = np.asarray([per_size[L]["w_star_mean"] for L in per_size], dtype=np.float64)
    t_stars = np.asarray([per_size[L]["t_star_mean"] for L in per_size], dtype=np.float64)

    alpha = fit_loglog_slope(Ls, w_stars)
    z = fit_loglog_slope(Ls, t_stars)
    beta = float(np.nanmean(beta_samples))
    z_from_alpha_beta = alpha / beta if beta > 0 else np.nan

    return (
        ExponentResults(beta=beta, alpha=alpha, z=z, z_from_alpha_beta=z_from_alpha_beta),
        per_size,
    )


if __name__ == "__main__":
    # Example usage.
    sim = StochasticGrowth2Pop(L=128, mu=1.5, seed=42)
    obs = sim.run(
        n_steps=50_000,
        record_every=50,
        snapshot_steps=[1_000, 5_000, 10_000, 25_000, 50_000],
        snapshot_dir="snapshots_L128_mu1p5",
        save_png_snapshots=True,
    )
    print(f"Recorded {len(obs['t'])} points. Final width={obs['w'][-1]:.3f}")

    exponents, stats = simulate_sizes_for_exponents(
        sizes=[32, 64, 96, 128],
        mu=1.5,
        n_steps=50_000,
        runs_per_size=4,
        seed=123,
        record_every=50,
    )
    print("Exponent estimates:", exponents)
    print("Per-size stats:", stats)
