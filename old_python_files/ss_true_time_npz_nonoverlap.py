from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import matplotlib.pyplot as plt
import numpy as np
import numba
from scipy.ndimage import binary_erosion
from scipy.optimize import curve_fit
import time as _time


def sample_power_law_jump(mu: float, rng: np.random.Generator, min_jump: float = np.sqrt(2)) -> float:
    """
    Sample jump distance xi from:
        J(xi, mu) = mu * xi^(-(mu + 1)),  xi >= min_jump
    via inverse transform sampling.
    """
    if mu <= 0:
        raise ValueError("mu must be > 0 for a normalizable jump distribution.")
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

def local_slopes(x, y, s=1):
    """
    Calculates the local slopes using 
        $m(x^*) = (y_{i+s} - y_i)/ (x_{i+s} - x_i)$ 
    where $x^* = (x_{i+s}+x_i)/2$ and $s$ is a skipping parameter. 
    """
    y_dim = len(y)
    ms = np.zeros(y_dim - s)
    xs = np.zeros(y_dim - s)

    for i in range(y_dim - s):
        ms[i] = (y[i+s] - y[i]) / (x[i+s] - x[i])
        xs[i] = (x[i+s] + x[i])/2

    return xs, ms


# ── Numba helpers ────────────────────────────────────────────────────────────

@numba.njit(cache=True)
def _seed_numba_rng(seed: int) -> None:
    np.random.seed(seed)


@numba.njit(cache=True)
def _batch_step_kernel(
    grid: np.ndarray,
    occupied_xy: np.ndarray,
    n_occupied: int,
    mu: float,
    L: int,
    grid_rows: int,
    time_acc: float,
    accepted: int,
    n_steps: int,
) -> tuple:
    two_pi = 2.0 * np.pi
    for _ in range(n_steps):
        idx = np.random.randint(0, n_occupied)
        sy = occupied_xy[idx, 0]
        sx = occupied_xy[idx, 1]
        pop_id = grid[sy, sx]

        u = np.random.random()
        jump = (1.0 - u) ** (-1.0 / mu)
        theta = two_pi * np.random.random()
        tx = int(round(sx + jump * np.cos(theta))) % L
        ty = int(round(sy + jump * np.sin(theta)))
        time_acc += 1.0 / accepted

        if ty < 0 or ty >= grid_rows:
            continue
        if grid[ty, tx] != 0:
            continue

        grid[ty, tx] = pop_id
        occupied_xy[n_occupied, 0] = ty
        occupied_xy[n_occupied, 1] = tx
        n_occupied += 1
        accepted += 1

    return n_occupied, time_acc, accepted


@numba.njit(cache=True)
def _sd_width_kernel(
    n_col: np.ndarray,
    sum_y_col: np.ndarray,
    sum_y2_col: np.ndarray,
    ls: np.ndarray,
) -> np.ndarray:
    """
    Compute surface width (mean SD over non-overlapping windows) for each l in ls.

    Uses non-overlapping windows of width l starting at 0, l, 2l, ...
    giving floor(L/l) windows per l-value.  
    """
    L = n_col.shape[0]

    # Build prefix sums once — reused for every l.
    prefix_n   = np.zeros(L + 1)
    prefix_sy  = np.zeros(L + 1)
    prefix_sy2 = np.zeros(L + 1)
    for i in range(L):
        prefix_n[i + 1]   = prefix_n[i]   + n_col[i]
        prefix_sy[i + 1]  = prefix_sy[i]  + sum_y_col[i]
        prefix_sy2[i + 1] = prefix_sy2[i] + sum_y2_col[i]

    widths = np.empty(ls.shape[0])
    for li in range(ls.shape[0]):
        l = int(ls[li])
        n_windows = L // l  # floor(L / l) non-overlapping windows

        total_sd = 0.0
        valid = 0
        for k in range(n_windows):
            start = k * l
            end   = start + l  # exclusive

            n_w  = prefix_n[end]   - prefix_n[start]
            sy   = prefix_sy[end]  - prefix_sy[start]
            sy2  = prefix_sy2[end] - prefix_sy2[start]

            if n_w > 0.0:
                mean_y = sy / n_w
                var = sy2 / n_w - mean_y * mean_y
                if var < 0.0: # make sure that variance is never negative
                    var = 0.0
                total_sd += var ** 0.5
                valid += 1

        widths[li] = total_sd / valid if valid > 0 else np.nan

    return widths


# ── JSON helper ──────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalars and arrays so they survive JSON round-trips."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Simulation class ─────────────────────────────────────────────────────────

class StochasticGrowthStripGeometry:
    """Stochastic growth on L×L square lattice with 2 populations.

    Population A_1 → 1, A_2 → 2, unoccupied → 0.
    Periodic boundary conditions in x (cylinder geometry).
    """

    def __init__(self, L: int, mu: float, seed: int = 42, initial_height: int = 600):
        if L < 2:
            raise ValueError("L must be at least 2")
        self.L       = int(L)
        self.mu      = float(mu)
        self.rng     = np.random.default_rng(seed)
        self.grid    = np.zeros((initial_height, self.L), dtype=np.int8)

        self._initialize_strip()

        capacity = self.grid.shape[0] * self.L
        self.occupied_xy = np.zeros((capacity, 2), dtype=np.int32)
        ys, xs = np.where(self.grid > 0)
        self.n_occupied = int(len(ys))
        for i, (y, x) in enumerate(zip(ys, xs)):
            self.occupied_xy[i, 0] = int(y)
            self.occupied_xy[i, 1] = int(x)

        self.time     = 0.0
        self.attempts = 0
        self.accepted = self.L

        self.history_accepted:        List[int]        = []
        self.history_attempts:        List[int]        = []
        self.history_record_interval: List[int]        = []
        self.history_t:               List[float]      = []
        self.history_max_height:      List[float]      = []
        self.history_mean_height:     List[float]      = []
        self.history_surface_width:   List[np.ndarray] = []
        self.ls: List[int] = []

        _seed_numba_rng(seed)

    def _initialize_strip(self) -> None:
        half = self.L // 2
        self.grid[0, :half] = 1
        self.grid[0, half:] = 2

    # ── Save / load (npz + json) ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save simulation state to <base>.npz (arrays) and <base>.json (metadata).

        The numpy RNG state is serialised to JSON so it is restored on load.
        Numba's internal RNG state is not preserved; call _seed_numba_rng()
        after loading if exact step reproducibility is needed.
        """
        base = str(Path(path).with_suffix(''))

        sw_arr = (
            np.stack(self.history_surface_width, axis=0)
            if self.history_surface_width
            else np.empty((0, max(len(self.ls), 1)), dtype=np.float64)
        )

        np.savez_compressed(
            base + '.npz',
            grid=self.grid,
            occupied_xy=self.occupied_xy[:self.n_occupied],
            history_accepted=np.asarray(self.history_accepted,        dtype=np.int64),
            history_attempts=np.asarray(self.history_attempts,        dtype=np.int64),
            history_record_interval=np.asarray(self.history_record_interval, dtype=np.int64),
            history_t=np.asarray(self.history_t,                      dtype=np.float64),
            history_max_height=np.asarray(self.history_max_height,    dtype=np.float64),
            history_mean_height=np.asarray(self.history_mean_height,  dtype=np.float64),
            history_surface_width=sw_arr,
            ls=np.asarray(self.ls, dtype=np.int64),
        )

        meta = {
            'L':          self.L,
            'mu':         self.mu,
            'time':       self.time,
            'attempts':   self.attempts,
            'accepted':   self.accepted,
            'n_occupied': self.n_occupied,
            'rng_state':  self.rng.bit_generator.state,
        }
        with open(base + '.json', 'w') as f:
            json.dump(meta, f, cls=_NumpyEncoder)

        print(f"Simulation saved to {base}.npz/.json  (t={self.time:.4f}, accepted={self.accepted})")

    @classmethod
    def load(cls, path: str) -> "StochasticGrowthStripGeometry":
        """Load simulation state from <base>.npz + <base>.json.

        Accepts paths with or without extension (strips it automatically).
        """
        base = str(Path(path).with_suffix(''))

        data = np.load(base + '.npz')
        with open(base + '.json', 'r') as f:
            meta = json.load(f)

        obj = cls.__new__(cls)
        obj.L          = int(meta['L'])
        obj.mu         = float(meta['mu'])
        obj.time       = float(meta['time'])
        obj.attempts   = int(meta['attempts'])
        obj.accepted   = int(meta['accepted'])
        obj.n_occupied = int(meta['n_occupied'])

        obj.grid = data['grid']

        obj.rng = np.random.default_rng()
        obj.rng.bit_generator.state = meta['rng_state']

        capacity = obj.grid.shape[0] * obj.L
        obj.occupied_xy = np.zeros((capacity, 2), dtype=np.int32)
        obj.occupied_xy[:obj.n_occupied] = data['occupied_xy']

        obj.ls                    = data['ls'].tolist()
        obj.history_accepted      = data['history_accepted'].tolist()
        obj.history_attempts      = data['history_attempts'].tolist()
        obj.history_record_interval = data['history_record_interval'].tolist()
        obj.history_t             = data['history_t'].tolist()
        obj.history_max_height    = data['history_max_height'].tolist()
        obj.history_mean_height   = data['history_mean_height'].tolist()

        sw_arr = data['history_surface_width']
        obj.history_surface_width = [sw_arr[i] for i in range(sw_arr.shape[0])]

        print(f"Simulation loaded from {base}.npz/.json  (t={obj.time:.4f}, accepted={obj.accepted})")
        return obj

    # ── Simulation methods ───────────────────────────────────────────────────

    def step(self) -> bool:
        self.attempts += 1
        idx    = int(self.rng.integers(0, self.n_occupied))
        sy     = int(self.occupied_xy[idx, 0])
        sx     = int(self.occupied_xy[idx, 1])
        pop_id = int(self.grid[sy, sx])

        jump  = sample_power_law_jump(self.mu, self.rng, min_jump=1.0)
        theta = 2.0 * np.pi * self.rng.random()
        tx    = int(round(sx + jump * np.cos(theta))) % self.L
        ty    = int(round(sy + jump * np.sin(theta)))
        self.time += 1.0 / self.accepted

        if ty < 0 or ty >= self.grid.shape[0]:
            return False
        if self.grid[ty, tx] != 0:
            return False

        self.grid[ty, tx] = pop_id
        self.occupied_xy[self.n_occupied, 0] = ty
        self.occupied_xy[self.n_occupied, 1] = tx
        self.n_occupied += 1
        self.accepted   += 1
        return True

    def _expand_grid(self) -> None:
        extra_rows = self.grid.shape[0]
        extra = np.zeros((extra_rows, self.L), dtype=np.int8)
        self.grid = np.vstack([self.grid, extra])
        new_cap = self.grid.shape[0] * self.L
        # expand the occupied sites array
        new_occ = np.zeros((new_cap, 2), dtype=np.int32)
        new_occ[:self.n_occupied] = self.occupied_xy[:self.n_occupied]
        self.occupied_xy = new_occ

    def fill_fraction(self) -> float:
        return float(np.count_nonzero(self.grid) / (self.grid.shape[0] * self.L))

    def max_interface_height(self) -> float:
        if self.n_occupied == 0:
            return 0.0
        return float(self.occupied_xy[:self.n_occupied, 0].max())

    def mean_interface_height(self, surface: np.ndarray) -> float:
        ys_idx, _ = np.where(surface > 0)
        return np.mean(ys_idx)

    def extract_surface_width_sd_fast(self, surface: np.ndarray, ls) -> np.ndarray:
        L = self.L
        ys_idx, xs_idx = np.where(surface > 0)
        ys_f       = ys_idx.astype(np.float64)
        n_col      = np.bincount(xs_idx, minlength=L).astype(np.float64)
        sum_y_col  = np.bincount(xs_idx, weights=ys_f,      minlength=L)
        sum_y2_col = np.bincount(xs_idx, weights=ys_f ** 2, minlength=L)
        return _sd_width_kernel(n_col, sum_y_col, sum_y2_col, np.asarray(ls, dtype=np.int64))

    def save_snapshot(self, out_dir: Path, t: int, save_png: bool = False) -> None:
        max_height = self.max_interface_height()
        y_bound    = int(3 * max_height)
        out_dir.mkdir(parents=True, exist_ok=True)
        if save_png:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
            cmap = plt.matplotlib.colors.ListedColormap(["white", "#1f77b4", "#d62728"])
            ax.imshow(self.grid[:y_bound, :], origin="lower", cmap=cmap,
                      vmin=0, vmax=2, interpolation="nearest")
            ax.set_title(f"L={self.L}, mu={self.mu}, t={t}")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            fig.tight_layout()
            fig.savefig(out_dir / f"snapshot_t{t:012d}.png")
            plt.close(fig)

    def show_sim(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        cmap = plt.matplotlib.colors.ListedColormap(["white", "#1f77b4", "#d62728"])
        ax.imshow(self.grid[:, :], origin="lower", cmap=cmap,
                  vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"L={self.L}, mu={self.mu}, t={self.attempts}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.tight_layout()

    def run(self, n_steps: int, record_interval_true: int = 10,
            ls: list = [],
            snapshot_steps: Optional[Iterable[int]] = None,
            snapshot_dir: Optional[str] = None,
            save_snapshots: bool = False):
        if not self.ls:
            if not ls:
                raise ValueError("ls has not been specified. Pass ls= to run() or set sim.ls.")
            self.ls = ls

        snapshot_set = set(snapshot_steps or [])
        out_dir      = Path(snapshot_dir) if snapshot_dir else None
        grid_rows    = self.grid.shape[0]
        steps_done   = 0
        record_interval = int(np.ceil(
                np.sqrt(2 * self.attempts * self.L) * record_interval_true
                + record_interval_true**2 * self.L / 2
            ))

        while steps_done < n_steps:
            batch = min(record_interval, n_steps - steps_done)
            print(f"Current progress: {steps_done} of {n_steps} --- {steps_done*100/n_steps:.2f} %", end="\r")

            self.n_occupied, self.time, self.accepted = _batch_step_kernel(
                self.grid, self.occupied_xy, self.n_occupied,
                self.mu, self.L, grid_rows,
                self.time, self.accepted, batch,
            )
            self.attempts += batch
            steps_done    += batch

            self.history_accepted.append(self.accepted)
            self.history_attempts.append(self.attempts)
            self.history_t.append(self.time)
            max_h = self.max_interface_height()
            self.history_max_height.append(max_h)
            if max_h > 0.67 * self.grid.shape[0]:
                print(f"Lattice height doubled. Previous {self.grid.shape[0]}. Max interface height: {max_h}.")
                self._expand_grid()
                grid_rows = self.grid.shape[0]

            # surface width extraction
            y_top = int(max_h) + 5 # 5 is arbitrary, it just has to be >= 2
            reduced_grid = self.grid[:y_top, :] # work with reduced grid since most of array is 0
            binary = (reduced_grid > 0)
            eroded = binary_erosion(binary, border_value=1)
            surface = (binary ^ eroded).astype(np.int8)

            self.history_surface_width.append(
                self.extract_surface_width_sd_fast(surface, self.ls)
            )

            self.history_mean_height.append(self.mean_interface_height(surface))

            record_interval = int(np.ceil(
                np.sqrt(2 * self.attempts * self.L) * record_interval_true
                + record_interval_true**2 * self.L / 2
            ))
            self.history_record_interval.append(record_interval)

            if out_dir is not None and save_snapshots:
                batch_start = self.attempts - batch
                if snapshot_set & set(range(batch_start, self.attempts)):
                    self.save_snapshot(out_dir, self.attempts, save_png=True)

        return self.get_obs()

    def get_obs(self):
        return {
            "t":                     np.asarray(self.history_t,              dtype=np.float64),
            "t_compute":             np.asarray(self.history_attempts),
            "record_interval":       np.asarray(self.history_record_interval),
            "max_height_history":    np.asarray(self.history_max_height,     dtype=np.float64),
            "mean_height_history":   np.asarray(self.history_mean_height,    dtype=np.float64),
            "surface_width_history": np.asarray(self.history_surface_width,  dtype=np.float64),
            "population_size":       np.asarray(self.history_accepted),
            "accepted": self.accepted,
        }


# ── Speedup analysis ─────────────────────────────────────────────────────────

def speedup_analysis(L: int, ls: list) -> None:
    """
    Print a theoretical speedup analysis and run a timing benchmark.

    Sliding-window (old):   L variance computations per l-value
    Non-overlapping (new):  floor(L/l) variance computations per l-value,
                            prefix sums built once and shared across all l-values.

    Total variance-computation count
        old:  n_l * L
        new:  sum_l floor(L/l)          (+ O(L) prefix setup, paid once)
    """
    ls_arr = np.asarray(ls, dtype=np.int64)
    n_l = len(ls)

    old_total = n_l * L
    new_total = sum(L // l for l in ls)
    overall_speedup = old_total / new_total if new_total else float('inf')

    print("=" * 60)
    print(f"Theoretical speedup analysis  (L={L}, {n_l} window sizes)")
    print("=" * 60)
    print(f"{'l':>6}  {'old (L)':>10}  {'new (L//l)':>12}  {'per-l speedup':>14}")
    print("-" * 60)
    for l in ls:
        new_windows = L // l
        per_l = L / new_windows if new_windows else float('inf')
        print(f"{l:>6}  {L:>10}  {new_windows:>12}  {per_l:>14.2f}x")
    print("-" * 60)
    print(f"{'TOTAL':>6}  {old_total:>10}  {new_total:>12}  {overall_speedup:>14.2f}x")
    print()
    print("Note: the prefix-sum build (3*L additions) is a one-time")
    print("cost shared across all l-values — counted as ~0 above.")
    print("=" * 60)

    # ── Timing benchmark ─────────────────────────────────────────────────────
    import ss_true_time_npz as old_mod

    rng = np.random.default_rng(0)
    n_col      = rng.integers(0, 5, size=L).astype(np.float64)
    sum_y_col  = rng.random(L) * 50
    sum_y2_col = sum_y_col ** 2 + rng.random(L) * 10
    ls_nb = np.asarray(ls, dtype=np.int64)

    # Warm up JIT for both kernels
    old_mod._sd_width_kernel(n_col, sum_y_col, sum_y2_col, ls_nb)
    _sd_width_kernel(n_col, sum_y_col, sum_y2_col, ls_nb)

    N_REPS = 500
    t0 = _time.perf_counter()
    for _ in range(N_REPS):
        old_mod._sd_width_kernel(n_col, sum_y_col, sum_y2_col, ls_nb)
    t_old = (_time.perf_counter() - t0) / N_REPS

    t0 = _time.perf_counter()
    for _ in range(N_REPS):
        _sd_width_kernel(n_col, sum_y_col, sum_y2_col, ls_nb)
    t_new = (_time.perf_counter() - t0) / N_REPS

    measured_speedup = t_old / t_new if t_new else float('inf')
    print()
    print(f"Measured wall-clock time ({N_REPS} reps each, post-JIT warm-up):")
    print(f"  old (sliding):        {t_old*1e6:8.2f} µs/call")
    print(f"  new (non-overlapping):{t_new*1e6:8.2f} µs/call")
    print(f"  measured speedup:     {measured_speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    L  = 512
    ls = [2, 4, 8, 16, 32, 64, 128, 256]
    speedup_analysis(L, ls)
