import os
import time
import json as _json
import multiprocessing as _mp
from stochastic_growth_true_time_simulation import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def _replica_filename(seed: int) -> str:
    return f"replica_seed{seed:06d}"

def _run_replica(L, mu, seed, n_steps, record_interval_true, ls, ensemble_dir,
                 save_every_seconds=1800, chunk_steps=100_000_000):
    """Worker: build one sim, run it, save it, return its save path.

    Saves the simulation every so often in time. This works by breaking up 
    the total n_steps into chunk_steps. Each chunk_step, it is checked
    if the elapsed time has exceeded save_every_seconds.
    """
    ensemble_dir = Path(ensemble_dir)
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(ensemble_dir / _replica_filename(seed))

    sim = StochasticGrowthStripGeometry(L=L, mu=mu, seed=seed)
    sim.ls = list(ls)

    remaining = n_steps
    last_save = time.monotonic()
    while remaining > 0:
        step = min(chunk_steps, remaining)
        sim.run(n_steps=step, record_interval_true=record_interval_true)
        remaining -= step
        if time.monotonic() - last_save >= save_every_seconds:
            sim.save(save_path)
            last_save=time.monotonic()
    
    sim.save(save_path)
    return save_path

def _run_loaded_replica(sim, seed, n_steps, record_interval_true, ensemble_dir,
                        save_every_seconds=1800, chunk_steps=100_000_000):
    save_path = str(Path(ensemble_dir) / _replica_filename(seed))
    remaining = n_steps
    last_save = time.monotonic()
    while remaining > 0:
        step = min(chunk_steps, remaining)
        sim.run(n_steps=step, record_interval_true=record_interval_true)
        remaining -= step
        if time.monotonic() - last_save >= save_every_seconds:
            sim.save(save_path)
            last_save=time.monotonic()
    sim.save(save_path)
    return save_path


def batch_run(
        L, mu, seeds, n_steps, record_interval_true, ls, root_dir,
        ensemble_dir=None, max_workers=10
        ):
    seeds = list(seeds)
    if ensemble_dir is None:
        _L_tag  = f"{L}"
        _mu_tag = f"{float(mu):.2f}".replace(".", "p")
        _t_tag  = f"{n_steps // 1_000_000}M"
        _N_tag  = f"{len(seeds)}"
        ensemble_dir = f"ensemble_L{_L_tag}_mu{_mu_tag}_t{_t_tag}_dtau{record_interval_true}_N{_N_tag}"
    ensemble_dir = root_dir / Path(ensemble_dir) 
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

def batch_load(ensemble_dir):
    """Load every replica in `ensemble_dir` into a dict {seed: sim}.

    Pairs each `.npz` with its `.json`. Replicas without both files are skipped.
    Returns (replicas_dict, ensemble_meta_dict_or_None).
    """
    ensemble_dir = Path(ensemble_dir)
    meta_path = ensemble_dir / "ensemble_meta.json"
    meta = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = _json.load(f)

    replicas = {}
    for npz_path in sorted(ensemble_dir.glob("replica_seed*.npz")):
        base = npz_path.with_suffix("")
        if not (base.with_suffix(".json")).exists():
            print(f"[skip] {npz_path.name} has no matching .json")
            continue
        seed = int(base.name.split("seed")[-1])
        replicas[seed] = StochasticGrowthStripGeometry.load(str(base))
    return replicas, meta

def batch_complete(ensemble_dir, max_workers=10):
    """Checks if any replicas are still missing steps; if so, complete them.

    Each replica is brought up to the target `n_steps` recorded in
    ensemble_meta.json, using the meta's `record_interval_true`. Updated
    replicas are saved back in place (overwriting the existing files in
    `ensemble_dir`). Replicas already at or past the target are skipped.
    """
    ensemble_dir = Path(ensemble_dir)
    replicas, meta = batch_load(ensemble_dir)
    if meta is None:
        raise FileNotFoundError(f"No ensemble_meta.json found in {ensemble_dir}")
    if not replicas:
        print(f"No replicas found in {ensemble_dir}")
        return

    target_n_steps = int(meta["n_steps"])
    record_interval_true = meta["record_interval_true"]

    # Figure out which replicas are short, and by how many steps.
    # NOTE: assumes the sim object exposes `n_steps` as the count of
    # steps already run. Adjust the attribute name if yours differs.
    todo = {}  # seed -> remaining_steps
    for seed, sim in replicas.items():
        done = int(getattr(sim, "attempts", 0))
        remaining = target_n_steps - done
        if remaining > 0:
            todo[seed] = remaining
            print(f"[seed {seed:06d}] short by {remaining} steps "
                  f"({done}/{target_n_steps})")
        else:
            print(f"[seed {seed:06d}] already complete ({done}/{target_n_steps})")

    if not todo:
        print("All replicas already complete.")
        return

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(todo))
    ctx = _mp.get_context("spawn")

    print(f"Completing {len(todo)} replica(s) with up to {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(
                _run_loaded_replica, replicas[seed], int(seed),
                remaining, record_interval_true, str(ensemble_dir),
            ): int(seed)
            for seed, remaining in todo.items()
        }
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                path = fut.result()
                print(f"[seed {seed:06d}] done → {path}")
            except Exception as e:
                print(f"[seed {seed:06d}] FAILED: {e!r}")

def batch_resume(
    ensemble_dir, n_steps, record_interval_true, max_workers=10
    ):
    """Resumes a batch. L, mu, and ls are fixed. 
    You can specify how many more steps to run as well as the desired recording interval.
    """
    replicas, meta = batch_load(ensemble_dir)
    seeds = replicas.keys()
    
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(seeds))
    ctx = _mp.get_context("spawn")

    # Makes a new directory 
    ensemble_dir = Path(ensemble_dir)
    meta_path = ensemble_dir / "ensemble_meta.json"
    meta = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = _json.load(f)
    # update meta
    meta["n_steps"] = meta["n_steps"] + n_steps
    meta["record_interval_true"] = record_interval_true
    _L_tag  = meta["L"]
    _mu_tag = f"{meta["mu"]:.2f}".replace(".", "p")
    _t_tag  = f"{(meta["n_steps"])// 1_000_000}M"
    _N_tag  = f"{len(seeds)}"
    new_ensemble_dir = f"resumed_ensemble_L{_L_tag}_mu{_mu_tag}_t{_t_tag}_dtau{record_interval_true}_N{_N_tag}"
    new_ensemble_dir = Path(new_ensemble_dir)
    new_ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Dump meta
    with open(new_ensemble_dir / "ensemble_meta.json", "w") as f:
        _json.dump(meta, f, indent=2)

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(
                _run_loaded_replica, replicas[seed], int(seed),
                n_steps, record_interval_true, str(new_ensemble_dir),
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
    resume = False
    L = 2**14
    mu = 5.0
    seeds = list(range(20,70))
    n_steps = int(1e7*5)
    record_interval_true = 3
    max_workers = 50
    ls = list(np.logspace(np.log10(2**8), np.log10(2**11), num=40, dtype=int))
    if resume:
        ensemble_dir = "/global/homes/g/ghan36/slurm_batch_scripts/ensemble_L16384_mu5p00_t50M_dtau1_N50"
        print(f"Current job: n_steps = {n_steps//1_000_000} Million, record_interval = {record_interval_true}")
        print(f"Resuming from {ensemble_dir}")
        batch_resume(ensemble_dir, n_steps, record_interval_true, max_workers=max_workers)
    else:
        print(f"Current job: L = {L}, mu = {mu}, seeds = {seeds}, n_steps = {n_steps//1_000_000} Million")
        batch_run(L, mu, seeds, n_steps, record_interval_true, ls, max_workers=max_workers)