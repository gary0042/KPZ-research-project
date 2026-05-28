from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import numba
from scipy.ndimage import binary_erosion

def sample_power_law_jump(mu: float, rng: np.random.Generator, min_jump: float = np.sqrt(2)) -> float:
    """
    Sample jump distance xi from:
        J(xi, mu) = mu * xi^(-(mu + 1)),  xi >= min_jump
    via inverse transform sampling.

    Returns jump distance r in interval [min_jump, infty]
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


# ── Numba helpers ────────────────────────────────────────────────────────────

@numba.njit(cache=True)
def _seed_numba_rng(seed: int) -> None:
    """Seed Numba's internal RNG (separate from numpy's Generator)."""
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
    """Run n_steps growth attempts entirely in compiled Numba code.

    By compiling the inner step loop, all Python interpreter overhead — function
    call dispatch, attribute lookups, int/float boxing — is eliminated. Combined
    with the numpy occupied_xy array (bottleneck #3 fix), every hot-path
    operation is a plain C memory access or arithmetic instruction.

    Uses Numba's internal RNG, seeded separately via _seed_numba_rng.
    Updates grid and occupied_xy in place; returns updated scalar state.

    Args:
        grid:        (grid_rows, L) int8 grid, modified in place.
        occupied_xy: Pre-allocated (capacity, 2) int32 array of (y, x) coords.
        n_occupied:  Current number of occupied sites (active length of occupied_xy).
        mu:          Power-law jump exponent.
        L:           Grid width (columns).
        grid_rows:   Grid height (rows).
        time_acc:    Accumulated physical time so far.
        accepted:    Total occupied sites so far.
        n_steps:     Number of step attempts to run.

    Returns:
        (n_occupied, time_acc, accepted) after n_steps attempts.
    """
    two_pi = 2.0 * np.pi
    for _ in range(n_steps):
        # Sample a random source site from the flat numpy array (O(1), cache-friendly)
        idx = np.random.randint(0, n_occupied)
        sy = occupied_xy[idx, 0]
        sx = occupied_xy[idx, 1]
        pop_id = grid[sy, sx]

        # Power-law jump, min_jump = 1.0
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
    """Incremental sliding-window SD kernel (see simulation_scratch_wall_time_pure_SD.ipynb)."""
    L = n_col.shape[0]
    widths = np.empty(ls.shape[0])
    for li in range(ls.shape[0]):
        l = ls[li]
        n_w = 0.0; sy = 0.0; sy2 = 0.0
        for k in range(l):
            n_w  += n_col[k]
            sy   += sum_y_col[k]
            sy2  += sum_y2_col[k]
        total_sd = 0.0
        valid = 0
        # for x in range(L):
        #     if n_w > 0.0:
        #         mean_y = sy / n_w
        #         var = sy2 / n_w - mean_y * mean_y
        #         if var < 0.0:
        #             var = 0.0
        #         total_sd += var ** 0.5
        #         valid += 1
        #     x_add = (x + l) % L
        #     n_w  += n_col[x_add]  - n_col[x]
        #     sy   += sum_y_col[x_add]  - sum_y_col[x]
        #     sy2  += sum_y2_col[x_add] - sum_y2_col[x]
        for x in range(L - l + 1):          # L-l+1 windows instead of L
            if n_w > 0.0:
                mean_y = sy / n_w
                var = sy2 / n_w - mean_y * mean_y
                if var < 0.0: var = 0.0
                total_sd += var ** 0.5
                valid += 1
            if x < L - l:                    # don't update past the last window
                n_w  += n_col[x + l] - n_col[x]
                sy   += sum_y_col[x + l]  - sum_y_col[x]
                sy2  += sum_y2_col[x + l] - sum_y2_col[x]
        widths[li] = total_sd / valid if valid > 0 else np.nan
    return widths


# ── Simulation class ─────────────────────────────────────────────────────────

class StochasticGrowthStripGeometry:
    """Stochastic growth on L by L SQUARE lattice with 2 populations.
    Population A_1 -> 1
    Population A_2 -> 2
    Unoccupied -> 0

    Periodic boundary conditions in x (cylinder geometry).

    Optimizations vs. original:
      - occupied_xy: pre-allocated (capacity, 2) int32 numpy array replaces
        the Python list of tuples. Eliminates heap-allocated tuple objects and
        is directly passable to Numba without conversion.
      - run() delegates to _batch_step_kernel (Numba JIT) which compiles the
        entire step loop to native code, eliminating all Python overhead per step.
    """

    def __init__(self, L: int, mu: float, seed: int = 42, initial_height: int = 500):
        if L < 2:
            raise ValueError("L must be at least 2")
        self.L       = int(L)
        self.mu      = float(mu)
        self.rng     = np.random.default_rng(seed)   # used by Python-side methods only
        self.grid    = np.zeros((max(initial_height, 2), self.L), dtype=np.int8)

        self._initialize_strip()

        capacity = self.grid.shape[0] * self.L
        self.occupied_xy = np.zeros((capacity, 2), dtype=np.int32) # holds (y, x) coordinates
        ys, xs = np.where(self.grid > 0)
        self.n_occupied = int(len(ys))
        for i, (y, x) in enumerate(zip(ys, xs)):
            self.occupied_xy[i, 0] = int(y)
            self.occupied_xy[i, 1] = int(x)

        self.time     = 0.0
        self.attempts = 0
        self.accepted = self.L   # initial strip has L occupied sites

        self.history_accepted:       List[int]        = []
        self.history_attempts:       List[int]        = []
        self.history_record_interval: List[int]       = []
        self.history_t:              List[float]      = []
        self.history_max_height:     List[float]      = []
        self.history_mean_height:    List[float]      = []
        self.history_surface_width:  List[np.ndarray] = []
        self.ls: List[int] = []

        # Seed Numba's internal RNG so _batch_step_kernel is reproducible
        _seed_numba_rng(seed)
    
    def __getstate__(self): # this is for saving and loading states
        state = self.__dict__.copy()
        # Only serialize the occupied entries, not the full pre-allocated capacity
        # Otherwise we will be saving L by L array of 32 bit ints
        # for L~10_000 thats around 400 mb per pickle!
        state['occupied_xy'] = self.occupied_xy[:self.n_occupied]
        return state

    def __setstate__(self, state): # this is for saving and loading states
        self.__dict__.update(state)
        # Restore full capacity on load
        capacity = self.grid.shape[0] * self.L
        full = np.zeros((capacity, 2), dtype=np.int32)
        full[:self.n_occupied] = self.occupied_xy
        self.occupied_xy = full
        
    def _initialize_strip(self) -> None:
        half = self.L // 2
        self.grid[0, :half] = 1
        self.grid[0, half:] = 2

    def step(self) -> bool:
        """Single Python-side growth step using the numpy occupied_xy array.

        Kept for debugging and interactive use. Production runs should use
        run(), which calls _batch_step_kernel for the entire loop.
        Note: uses self.rng (Python Generator), not Numba's RNG.
        """
        self.attempts += 1
        idx   = int(self.rng.integers(0, self.n_occupied))
        sy    = int(self.occupied_xy[idx, 0])
        sx    = int(self.occupied_xy[idx, 1])
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

    def fill_fraction(self) -> float:
        return float(np.count_nonzero(self.grid) / (self.grid.shape[0] * self.L))

    def _expand_grid(self) -> None:
        extra_rows = self.grid.shape[0]          # double current height
        extra      = np.zeros((extra_rows, self.L), dtype=np.int8)
        self.grid  = np.vstack([self.grid, extra])
        new_cap    = self.grid.shape[0] * self.L
        if new_cap > len(self.occupied_xy):
            new_occ = np.zeros((new_cap, 2), dtype=np.int32)
            new_occ[:self.n_occupied] = self.occupied_xy[:self.n_occupied]
            self.occupied_xy = new_occ

    def max_interface_height(self) -> float:
        h = np.zeros(self.L, dtype=np.float64)
        for x in range(self.L):
            ys = np.where(self.grid[:, x] > 0)[0]
            h[x] = ys.max() if ys.size else 0.0
        return float(np.max(h))

    def median_interface_height(self) -> float:
        """Do not use."""
        h = np.zeros(self.L, dtype=np.float64)
        for x in range(self.L):
            ys = np.where(self.grid[:, x] > 0)[0]
            h[x] = ys.max() if ys.size else 0.0
        return float(np.median(h))

    def mean_interface_height(self, surface: np.ndarray) -> float:
        ys_idx, _ = np.where(surface > 0)
        return float(np.mean(ys_idx))

    def extract_surface(self) -> np.ndarray:
        binary = (self.grid > 0)
        eroded = binary_erosion(binary, border_value=1)
        return (binary ^ eroded).astype(np.uint8)

    def extract_surface_width_sd_fast(self, surface: np.ndarray, ls) -> np.ndarray:
        """Compute w(l, t) for each l in ls via Numba incremental SD kernel."""
        L = self.L
        ys_idx, xs_idx = np.where(surface > 0)
        ys_f       = ys_idx.astype(np.float64)
        n_col      = np.bincount(xs_idx, minlength=L).astype(np.float64)
        sum_y_col  = np.bincount(xs_idx, weights=ys_f,      minlength=L)
        sum_y2_col = np.bincount(xs_idx, weights=ys_f ** 2, minlength=L)
        return _sd_width_kernel(n_col, sum_y_col, sum_y2_col, np.asarray(ls, dtype=np.int64))

    def save(self, path: str) -> None:
        """Pickle the full simulation state (grid, histories, RNG state).

        Note: Numba's internal RNG state is NOT preserved by pickle.
        After loading, call _seed_numba_rng(new_seed) if exact reproducibility
        of subsequent steps is required.
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Simulation saved to {path}  (t={self.time:.4f}, accepted={self.accepted})")

    @classmethod
    def load(cls, path: str) -> "StochasticGrowthStripGeometry":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"Simulation loaded from {path}  (t={obj.time:.4f}, accepted={obj.accepted})")
        return obj

    def save_snapshot(self, out_dir: Path, t: int, save_png: bool = False) -> None:
        max_height = self.max_interface_height()
        y_bound    = int(3 * max_height)
        out_dir.mkdir(parents=True, exist_ok=True)
        if save_png:
            try:
                import matplotlib.pyplot as plt
            except Exception:
                return
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
        """Shows current simulation"""
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        cmap = plt.matplotlib.colors.ListedColormap(["white", "#1f77b4", "#d62728"])
        ax.imshow(self.grid[:, :], origin="lower", cmap=cmap,
                    vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"L={self.L}, mu={self.mu}, t={self.attempts}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.tight_layout()
        return

    def run(self, n_steps: int, record_interval_true: int = 10,
            ls: list = [],
            snapshot_steps: Optional[Iterable[int]] = None,
            snapshot_dir: Optional[str] = None,
            save_snapshots: bool = False):
        """Run the simulation for n_steps with diagnostics sampled uniformly in true time.

        Uses _batch_step_kernel (Numba JIT) to execute a variable-sized batch at a time,
        then drops back to Python to record diagnostics. The batch size grows as
        sqrt(2*t*L)*Δτ + L*Δτ²/2 so that consecutive measurements are spaced by Δτ
        in true time τ (where τ ~ sqrt(t/L) asymptotically).

        Note: snapshots fire at the nearest batch boundary, not at exact attempt counts.

        Args:
            n_steps:              Total number of computational step attempts.
            record_interval_true: Desired spacing Δτ between recordings in true time.
            ls:                   Window widths for w(l, t) measurement.
            snapshot_steps:       Attempt counts at which to save PNG snapshots.
            snapshot_dir:         Directory to write snapshots into.
            save_snapshots:       Whether to save PNG images at snapshot_steps.
        """
        if not self.ls:
            if not ls:
                raise ValueError("ls has not been specified. Pass ls= to run() or set sim.ls.")
            self.ls = ls

        snapshot_set = set(snapshot_steps or [])
        out_dir      = Path(snapshot_dir) if snapshot_dir else None
        grid_rows    = self.grid.shape[0]
        steps_done   = 0
        # Initial batch: first measurement at t = L*Δτ²/2 (derived from t = L*τ²/2 with τ=Δτ)
        record_interval = round(record_interval_true**2 * self.L / 2)

        while steps_done < n_steps:
            batch = min(record_interval, n_steps - steps_done)
            print(f"Current progress: {steps_done} of {n_steps} --- {steps_done*100/n_steps:.2f} %", end="\r")

            # Entire batch runs in compiled numba code
            self.n_occupied, self.time, self.accepted = _batch_step_kernel(
                self.grid, self.occupied_xy, self.n_occupied,
                self.mu, self.L, grid_rows,
                self.time, self.accepted, batch,
            )
            self.attempts += batch
            steps_done    += batch

            # Record diagnostics (Python-side, runs infrequently)
            self.history_accepted.append(self.accepted)
            self.history_attempts.append(self.attempts)
            self.history_t.append(self.time)
            max_h = self.max_interface_height()
            self.history_max_height.append(max_h)
            if max_h > 0.67 * self.grid.shape[0]: 
                print(f"Lattice height doubled. Previous lattice height: {self.grid.shape[0]}. Max interface height: {max_h}.")
                self._expand_grid()
                grid_rows = self.grid.shape[0]
            surface = self.extract_surface()
            self.history_surface_width.append(
                self.extract_surface_width_sd_fast(surface, self.ls)
            )
            self.history_mean_height.append(self.mean_interface_height(surface))

            # Next batch size: Δt = sqrt(2*t*L)*Δτ + L*Δτ²/2, giving uniform Δτ in true time
            record_interval = int(np.ceil(
                np.sqrt(2 * self.attempts * self.L) * record_interval_true
                + record_interval_true**2 * self.L / 2
            ))
            self.history_record_interval.append(record_interval)

            # Snapshot: saves snapshot if any attempt in this batch is in snapshot_set
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

if __name__ =="__main__":
    import time as _time

    # ============================================================
    # Ensemble hyperparameters  (edit these)
    # ============================================================
    L                    = 500_000     # lattice side length
    mu                   = 3.5      # power-law jump exponent
    ls                   = list(np.unique(np.logspace(np.log10(100), np.log10(L), num=10, dtype=int)))
    n_steps              = 1_000_000_000 * 1_000  # computational steps per replica
    record_interval_true = 10     # desired spacing Δτ between recordings in true time
    N_ensemble           = 10        # number of independent replicas
    base_seed            = 42        # seeds will be base_seed, base_seed+1, ...

    # ── Derived names ─────────────────────────────────────────────────────────────
    _mu_tag   = f"{mu:.2f}".replace(".", "p")
    _t_tag    = f"{n_steps // 1_000_000_000}B"
    batch_tag = f"ensemble_L{L}_mu{_mu_tag}_t{_t_tag}_dtau{record_interval_true}_N{N_ensemble}"
    batch_dir = Path(batch_tag)
    batch_dir.mkdir(exist_ok=True)

    def _replica_path(seed: int) -> Path:
        return batch_dir / f"replica_seed{seed:06d}.pkl"

    seeds = [base_seed + i for i in range(N_ensemble)]

    print(f"Ensemble: N={N_ensemble}  L={L}  mu={mu}  n_steps={n_steps:,}  record_interval_true={record_interval_true}  ls={ls}")
    print(f"Checkpoints: {batch_dir}/")

    # ============================================================
    # Run ensemble  (resumes from checkpoint if file already exists)
    # ============================================================
    obs_list: list[dict] = []

    for rank, seed in enumerate(seeds):
        ckpt = _replica_path(seed)
        t0   = _time.perf_counter()

        # Checks if a replica already exists if it does,
        # load that replica. If not, instantiates new replica.
        if ckpt.exists():
            sim_r = StochasticGrowthStripGeometry.load(str(ckpt))
            print(f"[{rank+1}/{N_ensemble}] Resumed seed={seed}  (attempts={sim_r.attempts:,})")
            remaining = max(0, n_steps - sim_r.attempts)
        else:
            sim_r         = StochasticGrowthStripGeometry(L=L, mu=mu, seed=seed)
            sim_r.ls      = ls
            remaining     = n_steps
            print(f"[{rank+1}/{N_ensemble}] Starting  seed={seed}")

        # For the current replica, checks if it is done simulating
        # If not, simulates replica until completion
        if remaining > 0:
            obs = sim_r.run(
                n_steps              = remaining,
                record_interval_true = record_interval_true,
            )
        else:
            obs = sim_r.get_obs()

        # observations from each replica are appended to list
        sim_r.save(str(ckpt))
        obs_list.append(obs)
        print(f"    done in {_time.perf_counter()-t0:.1f}s  |  records={len(obs['t'])}") # timing

    print(f"\nAll {N_ensemble} replicas complete.")