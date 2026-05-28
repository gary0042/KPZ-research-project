#!/usr/bin/env python3
"""
Script generator for KPZ stochastic growth NERSC batch jobs.

Edit the USER CONFIGURATION block, then run:
    python generate_scripts.py

For each (L, mu, n_steps, record_interval_true) combination, writes to
output_dir/:
  run_<name>.py       self-contained Python simulation script
  submit_<name>.sh    SLURM batch submission script
  submit_all.sh       master script to sbatch every generated job
"""

from pathlib import Path
import numpy as np


# =====================================================================
# USER CONFIGURATION — edit this section; leave the rest unchanged
# =====================================================================

# Lattice sizes (usually just one value)
L_values = [2**14]

# Window sizes (ls).  Evaluated once at generation time; the resulting
# list is embedded verbatim in every generated Python script.
# numpy is available as `np`.
ls_expr = "list(np.logspace(7, 13, num=100, base=2, dtype=int))"

# mu values to sweep over
mu_values = [3.5,4.0,4.5,5.0,5.5,6.0]

# (n_steps, record_interval_true, SLURM walltime "hh:mm:ss") triplets.
# n_steps and record_interval_true are paired; walltime applies per
# (L, n_steps, record_interval_true) combination.
step_configs = [
    (int(3*(10**12)), 5, "08:00:00"),
]

# Number of replicates per configuration
N_replicates = 100

# Seeds: range(seed_start, seed_start + N_replicates)
seed_start = 1

# Parallel workers per job; SLURM --cpus-per-task = 2*(max_workers + 5)
# Doubling because each physical core has 2 threads (logical cpus) 
# If we truly want max_worker number of cores we have to double what we request.
max_workers = 100

# Root directory for ensemble output on the cluster.
# Any valid Python expression; Path and os are available in generated scripts.
# Typical NERSC value: Path(os.environ["PSCRATCH"])
root_dir_expr = 'Path(os.environ["PSCRATCH"])'

# Local directory where generated scripts are written
output_dir = "generated_scripts"

# NERSC directory where you will place the scripts before submission
nersc_base = "/global/homes/g/ghan36/slurm_batch_scripts"

# SLURM settings
slurm_account    = "m3152"
slurm_queue      = "regular"
slurm_constraint = "cpu"
slurm_nodes      = 1
conda_env        = "py313-ssg"
email            = "gary_han@berkeley.edu"

# =====================================================================
# END OF CONFIGURATION
# =====================================================================


# ---------------------------------------------------------------------------
# Helper functions embedded verbatim in every generated Python script.
# ---------------------------------------------------------------------------

_HELPERS = '''\
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
    """Save sim such that an interrupted save cannot corrupt the final files.

    Writes to a temp path then atomically renames into place via os.replace.
    The trailing "." on tmp_base is intentional: it keeps Path.with_suffix('')
    a no-op inside sim.save(), so sim.save writes <tmp_base>.npz/.json rather
    than stripping ".tmp" and overwriting the real target directly.
    """
    tmp_base = f"{save_path}.tmp."
    with contextlib.redirect_stdout(io.StringIO()):
        sim.save(tmp_base)
    os.replace(f"{tmp_base}.npz",  f"{save_path}.npz")
    os.replace(f"{tmp_base}.json", f"{save_path}.json")
    print(f"Simulation saved (atomic) to {save_path}.npz/.json "
          f"(t={sim.time:.4f}, accepted={sim.accepted})")


def _is_valid_replica(base) -> bool:
    """Return True iff <base>.npz and <base>.json both exist and look intact."""
    base = Path(base)
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
                 save_every_seconds=1800, chunk_steps=100_000_000):
    """Worker: build one sim, run it in chunks, periodically save (atomic)."""
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
            last_save = time.monotonic()

    _atomic_sim_save(sim, save_path)
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
            _atomic_sim_save(sim, save_path)
            last_save = time.monotonic()
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
    """Load every replica in ensemble_dir into a dict {seed: sim}."""
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

    todo = {}
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
    """Resume a batch run from saved replicas."""
    replicas, meta = batch_load(ensemble_dir)
    seeds = replicas.keys()

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = min(max_workers, len(seeds))
    ctx = _mp.get_context("spawn")

    ensemble_dir = Path(ensemble_dir)
    meta_path = ensemble_dir / "ensemble_meta.json"
    meta = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = _json.load(f)
    meta["n_steps"] = meta["n_steps"] + n_steps
    meta["record_interval_true"] = record_interval_true
    _L_tag  = meta["L"]
    _mu_tag = f"{meta['mu']:.2f}".replace(".", "p")
    _t_tag  = f"{(meta['n_steps']) // 1_000_000}M"
    _N_tag  = f"{len(seeds)}"
    new_ensemble_dir = f"resumed_ensemble_L{_L_tag}_mu{_mu_tag}_t{_t_tag}_dtau{record_interval_true}_N{_N_tag}"
    new_ensemble_dir = Path(new_ensemble_dir)
    new_ensemble_dir.mkdir(parents=True, exist_ok=True)

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
'''


# ---------------------------------------------------------------------------
# Generator internals
# ---------------------------------------------------------------------------

def _mu_tag(mu: float) -> str:
    return f"{mu:.2f}".replace(".", "p")


def _t_tag(n_steps: int) -> str:
    return f"{n_steps // 1_000_000}M"


def _job_name(L, mu, n_steps, record_interval_true, N) -> str:
    return f"L{L}_mu{_mu_tag(mu)}_t{_t_tag(n_steps)}_dtau{record_interval_true}_N{N}"


def _make_main_block(
    L, mu, n_steps, record_interval_true, max_workers,
    seed_start, N_replicates, ls_values, root_dir_expr,
) -> str:
    seed_end = seed_start + N_replicates
    ls_repr = repr(ls_values)
    lines = [
        "",
        "",
        "if __name__ == '__main__':",
        "    complete = False",
        "    respawn  = False",
        f"    L = {L}",
        f"    mu = {mu}",
        f"    seeds = list(range({seed_start}, {seed_end}))",
        f"    n_steps = int({n_steps})",
        f"    record_interval_true = {record_interval_true}",
        f"    max_workers = {max_workers}",
        f"    root_dir = {root_dir_expr}",
        f"    ls = {ls_repr}",
        "    if complete or respawn:",
        "        _L_tag  = f'{L}'",
        "        _mu_tag = f'{float(mu):.2f}'.replace('.', 'p')",
        "        _t_tag  = f'{n_steps // 1_000_000}M'",
        "        _N_tag  = f'{len(seeds)}'",
        "        ensemble_dir = root_dir / f'ensemble_L{_L_tag}_mu{_mu_tag}_t{_t_tag}_dtau{record_interval_true}_N{_N_tag}'",
        "    if complete:",
        "        print(f'Completing from {ensemble_dir}')",
        "        batch_complete(ensemble_dir, max_workers=max_workers)",
        "    elif respawn:",
        "        print(f'Respawning bad seeds from {ensemble_dir}')",
        "        batch_respawn(ensemble_dir, max_workers=max_workers)",
        "    else:",
        "        print(f'Current job: L = {L}, mu = {mu}, "
        "n_steps = {n_steps // 1_000_000}M, record_interval = {record_interval_true}')",
        "        batch_run(L, mu, seeds, n_steps, record_interval_true, ls, root_dir, max_workers=max_workers)",
        "",
    ]
    return "\n".join(lines)


def _make_slurm_script(job_name, walltime, script_nersc_path, cpus_per_task) -> str:
    return f"""\
#!/bin/bash
#SBATCH -A {slurm_account}
#SBATCH -q {slurm_queue}
#SBATCH -C {slurm_constraint}
#SBATCH -N {slurm_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task * 2}
#SBATCH -t {walltime}
#SBATCH -J {job_name}
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user={email}

set -euo pipefail
echo "Starting job ${{SLURM_JOB_NAME}} (ID: ${{SLURM_JOB_ID}}) on host $(hostname)"
echo "Running in directory: ${{PWD}}"
echo "SLURM_NODELIST: ${{SLURM_NODELIST}}"
echo "SLURM_NTASKS:   ${{SLURM_NTASKS}}"
echo "CPUS per task:  ${{SLURM_CPUS_PER_TASK}}"

date

module load conda
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
fi
conda activate {conda_env}

echo "Using Python: $(which python)"
python --version

mkdir -p logs
srun -n 1 -c ${{SLURM_CPUS_PER_TASK}} python {script_nersc_path}

echo "Job completed at:"
date
"""


def main():
    ls_values = [int(x) for x in eval(ls_expr, {"np": np})]
    cpus_per_task = max_workers + 5

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    slurm_names = []
    total = len(L_values) * len(mu_values) * len(step_configs)
    print(f"Generating {total} job pair(s) → '{output_dir}/'")
    print()

    for L in L_values:
        for mu in mu_values:
            for n_steps, record_interval_true, walltime in step_configs:
                name    = _job_name(L, mu, n_steps, record_interval_true, N_replicates)
                py_name = f"run_{name}.py"
                sh_name = f"submit_{name}.sh"

                py_content = _HELPERS + _make_main_block(
                    L, mu, n_steps, record_interval_true,
                    max_workers, seed_start, N_replicates, ls_values,
                    root_dir_expr,
                )
                (out / py_name).write_text(py_content)

                sh_content = _make_slurm_script(
                    job_name=name,
                    walltime=walltime,
                    script_nersc_path=f"{nersc_base}/{py_name}",
                    cpus_per_task=cpus_per_task,
                )
                sh_path = out / sh_name
                sh_path.write_text(sh_content)
                sh_path.chmod(0o755)

                slurm_names.append(sh_name)
                print(f"  {py_name}")
                print(f"  {sh_name}  (walltime={walltime}, cpus={cpus_per_task})")
                print()

    master = out / "submit_all.sh"
    master.write_text("#!/bin/bash\n\n" + "".join(f"sbatch {s}\n" for s in slurm_names))
    master.chmod(0o755)
    print(f"Master script: {output_dir}/submit_all.sh")


if __name__ == "__main__":
    main()
