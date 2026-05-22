import os
import json as _json
import multiprocessing as _mp
import time
from ss_true_time_npz_nonoverlap_fast import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


def _replica_filename(seed: int) -> str:
    return f"replica_seed{seed:06d}"

def _run_replica(L, mu, seed, n_steps, record_interval_true, ls, ensemble_dir):
    """Worker: build one sim, run it, save it, return its save path.

    Defined at notebook top level so `fork`-spawned children can find it.
    """
    ensemble_dir = Path(ensemble_dir)
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    sim = StochasticGrowthStripGeometry(L=L, mu=mu, seed=seed)
    sim.ls = list(ls)
    sim.run(
        n_steps=n_steps,
        record_interval_true=record_interval_true,
    )

    save_path = str(ensemble_dir / _replica_filename(seed))
    sim.save(save_path)
    return save_path


def batch_run(
        L, mu, seeds, n_steps, record_interval_true, ls,
        ensemble_dir=None, max_workers=10
        ):
    seeds = list(seeds)
    if ensemble_dir is None:
        _L_tag  = f"{L}"
        _mu_tag = f"{float(mu):.2f}".replace(".", "p")
        _t_tag  = f"{n_steps // 1_000_000}M"
        _N_tag  = f"{len(seeds)}"
        ensemble_dir = f"ensemble_L{_L_tag}_mu{_mu_tag}_t{_t_tag}_dtau{record_interval_true}_N{_N_tag}"
    ensemble_dir = Path(ensemble_dir)
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "L": int(L),
        "mu": float(mu),
        "n_steps": int(n_steps),
        "record_interval_true": int(record_interval_true),
        "ls": list(map(int, ls)),
        "seeds": list(map(int, seeds)),
    }
    with open(ensemble_dir / "ensemble_meta.json", "w") as f:
        _json.dump(meta, f, indent=2)

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(seeds))

    ctx = _mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(
                _run_replica, L, mu, int(seed),
                n_steps, record_interval_true, ls, str(ensemble_dir),
            ): int(seed)
            for seed in seeds
        }
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                path = fut.result()
                print(f"[seed {seed:06d}] done → {path}")
            except Exception as e:
                print(f"[seed {seed:06d}] FAILED: {e!r}")

if __name__ == "__main__":
    start = time.perf_counter()
    L = 2**14
    mu = 5.0
    seeds = range(20)
    n_steps = 10_000_000 * 5_000
    record_interval_true = 1
    ls = list(np.logspace(np.log10(10), np.log10(2**10), num=20, dtype=int))
    batch_run(L, mu, seeds, n_steps, record_interval_true, ls, max_workers=5)
    end = time.perf_counter()
