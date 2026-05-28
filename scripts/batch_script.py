import os
import time
import json as _json
import multiprocessing as _mp
import zipfile
import contextlib, io
from stochastic_growth_true_time_simulation import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def _atomic_sim_save(sim, save_path):
    tmp_base = f"{save_path}.tmp."
    with contextlib.redirect_stdout(io.StringIO()):
        sim.save(tmp_base)                              # writes tmp_base.npz + tmp_base.json
    os.replace(f"{tmp_base}.npz",  f"{save_path}.npz")
    os.replace(f"{tmp_base}.json", f"{save_path}.json")
    print(f"Simulation saved (atomic) to {save_path}.npz/.json "
          f"(t={sim.time:.4f}, accepted={sim.accepted})\n")
    
def _is_valid_replica(base: Path) -> bool:
    npz = base.with_suffix(".npz")
    js  = base.with_suffix(".json")
    if not (npz.exists() and js.exists()):
        return False
    if not zipfile.is_zipfile(npz):
        return False
    try:
        with open(js) as f:
            _json.load(f)
    except Exception:
        return False
    return True

def _replica_filename(seed: int) -> str:
    return f"replica_seed{seed:06d}"

def _run_replica(L, mu, seed, n_steps, record_interval_true, ls, ensemble_dir,
                 save_every_seconds=2579, chunk_steps=100_000_000):
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
            _atomic_sim_save(sim, save_path)
            last_save=time.monotonic()
    
    _atomic_sim_save(sim, save_path)
    return save_path

def _run_loaded_replica(sim, seed, n_steps, record_interval_true, ensemble_dir,
                        save_every_seconds=2579, chunk_steps=100_000_000):
    save_path = str(Path(ensemble_dir) / _replica_filename(seed))
    remaining = n_steps
    last_save = time.monotonic()
    while remaining > 0:
        step = min(chunk_steps, remaining)
        sim.run(n_steps=step, record_interval_true=record_interval_true)
        remaining -= step
        if time.monotonic() - last_save >= save_every_seconds:
            _atomic_sim_save(sim, save_path)
            last_save=time.monotonic()
    _atomic_sim_save(sim, save_path)
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
        if not _is_valid_replica(base):
            print(f"[skip] {npz_path.name} is corrupt or missing pair")
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

def batch_respawn(ensemble_dir, max_workers=10):
    """Re-run any seed whose replica file is missing or corrupt.

    Seeds and parameters are taken from ensemble_meta.json. New replicas
    are written into ensemble_dir, overwriting any corrupt files in place.
    """
    ensemble_dir = Path(ensemble_dir)
    meta_path = ensemble_dir / "ensemble_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No ensemble_meta.json in {ensemble_dir}")
    with open(meta_path) as f:
        meta = _json.load(f)

    L                    = meta["L"]
    mu                   = meta["mu"]
    n_steps              = meta["n_steps"]
    record_interval_true = meta["record_interval_true"]
    ls                   = meta["ls"]
    all_seeds            = meta["seeds"]

    bad_seeds = [
        s for s in all_seeds
        if not _is_valid_replica(ensemble_dir / _replica_filename(s))
    ]
    if not bad_seeds:
        print("All replicas valid; nothing to respawn.")
        return
    print(f"Respawning {len(bad_seeds)} seed(s): {bad_seeds}")

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(bad_seeds))
    ctx = _mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(
                _run_replica, L, mu, int(seed),
                n_steps, record_interval_true, ls, str(ensemble_dir),
            ): int(seed)
            for seed in bad_seeds
        }
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                path = fut.result()
                print(f"[seed {seed:06d}] respawned → {path}")
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
    complete = False
    respawn = False
    L = 2**14
    mu = 5.0
    seeds = list(range(20,70))
    n_steps = int(1e7*5)
    record_interval_true = 3
    max_workers = 50
    root_dir = "Your directory here"
    ls = list(np.logspace(np.log10(2**8), np.log10(2**11), num=40, dtype=int))
    if complete:
        ensemble_dir = "/global/homes/g/ghan36/slurm_batch_scripts/ensemble_L16384_mu5p00_t50M_dtau1_N50"
        print(f"Current job: n_steps = {n_steps//1_000_000} Million, record_interval = {record_interval_true}")
        print(f"Completing/resuming from {ensemble_dir}")
        batch_complete(ensemble_dir, max_workers=max_workers)
    elif respawn:
        ensemble_dir = "/global/homes/g/ghan36/slurm_batch_scripts/ensemble_L16384_mu5p00_t50M_dtau1_N50"
        print(f"Current job: n_steps = {n_steps//1_000_000} Million, record_interval = {record_interval_true}")
        print(f"Respawning bad seeds from {ensemble_dir}")
        batch_respawn(ensemble_dir, max_workers=max_workers)
    else:
        print(f"Current job: L = {L}, mu = {mu}, seeds = {seeds}, n_steps = {n_steps//1_000_000} Million")
        batch_run(L, mu, seeds, n_steps, record_interval_true, ls, root_dir, max_workers=max_workers)