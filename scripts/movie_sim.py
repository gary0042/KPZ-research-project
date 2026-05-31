"""
movie_sim.py

script for generating movies of simulations
"""

import time

from stochastic_growth_true_time_simulation import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ===============================================================

sim_list = [
    "ensemble_L8192_mu3p30_t124000M_dtau3_N80/replica_seed000005.npz",
    "ensemble_L8192_mu3p50_t124000M_dtau3_N80/replica_seed000040.npz",
    "ensemble_L8192_mu3p60_t124000M_dtau3_N80/replica_seed000067.npz",
    "ensemble_L8192_mu3p70_t124000M_dtau3_N80/replica_seed000022.npz",
    "ensemble_L8192_mu4p10_t124000M_dtau3_N80/replica_seed000020.npz",
    "ensemble_L8192_mu4p30_t124000M_dtau3_N80/replica_seed000063.npz",
    "ensemble_L8192_mu4p90_t124000M_dtau3_N80/replica_seed000037.npz",
    "ensemble_L8192_mu5p00_t3000000M_dtau5_N100/replica_seed000001.npz",
    "ensemble_L8192_mu5p60_t124000M_dtau3_N80/replica_seed000034.npz",
    "ensemble_L8192_mu6p60_t124000M_dtau3_N80/replica_seed000031.npz",
]

movie_settings = [
    {'frames': 300, 'subsample': 2}
] * len(sim_list)

# ===============================================================


def _format_mu(mu):
    return str(mu).replace('.', 'p')


def float_to_p(x, ndigits=1, keep_trailing=False):
    s = f"{x:.{ndigits}f}" if keep_trailing else f"{round(x, ndigits):.{ndigits}f}".rstrip('0').rstrip('.')
    return s.replace('.', 'p')


def get_frame_markers(sim, num_frames=200):
    """Splices the attempts into num_frame entries."""
    occupied = sim.n_occupied
    # endpoint=False so every marker is a valid index into [0, n_occupied)
    return np.linspace(0, occupied, num=num_frames, endpoint=False, dtype=int)


def make_movie(sim, num_frames=200, subsample=4, verbose=False):
    outdir = Path(f"simulation_movies")
    outdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[make_movie] L={sim.L} mu={sim.mu} n_occupied={sim.n_occupied:,} "
              f"target frames={num_frames} subsample={subsample}")

    # construct frames
    frame_markers = get_frame_markers(sim, num_frames)
    marker_set = set(frame_markers.tolist())  # O(1) membership instead of O(num_frames)
    occupied_history = sim.occupied_xy[:sim.n_occupied, :]
    grid = np.zeros((sim.L, sim.L), dtype=bool)
    snapshots = []  # container for frames of the movie
    snapshot_Ns = []  # event count corresponding to each snapshot

    # print event-loop progress ~50 times total so we don't spam the log
    report_every = max(1, sim.n_occupied // 50)
    t0 = time.time()
    for i in range(sim.n_occupied):
        y, x = occupied_history[i, :]
        grid[y, x] = True
        if i in marker_set:
            # .copy() is essential -- without it every snapshot is a view of the
            # same `grid` array, so all frames render as the final state.
            snapshots.append(grid[::subsample, ::subsample].copy())
            snapshot_Ns.append(i)
        if verbose and (i + 1) % report_every == 0:
            pct = 100 * (i + 1) / sim.n_occupied
            print(f"  capture: {i+1:>12,}/{sim.n_occupied:,} "
                  f"({pct:5.1f}%)  frames={len(snapshots)}  "
                  f"elapsed={time.time()-t0:5.1f}s", end="\r", flush=True)

    if verbose:
        print(f"\n[make_movie] captured {len(snapshots)} frames "
              f"in {time.time()-t0:.1f}s")

    # construct animation
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(snapshots[0], origin="lower", animated=True, vmin=0, vmax=1)
    title = ax.set_title("")
    ax.set_axis_off()

    def update(f_idx):
        im.set_data(snapshots[f_idx])
        title.set_text(
            rf"Sim movie $L$={sim.L} $\mu$={sim.mu}  |  N={snapshot_Ns[f_idx]}"
        )
        return im, title
    
    anim = FuncAnimation(
        fig, update,
        frames=len(snapshots),
        interval=400,            # ms between frames (display only)
        blit=False,              # blit=True is faster but doesn't redraw the title cleanly
    )

    def _save_progress(current, total):
        if verbose:
            print(f"  encode:  {current+1:>5}/{total} "
                  f"({100*(current+1)/total:5.1f}%)", end="\r", flush=True)

    out_path = outdir / f"mu_{_format_mu(sim.mu)}_L{sim.L}.mp4"
    if verbose:
        print(f"[make_movie] encoding {out_path} (codec=libx264) ...")
        t_save = time.time()
    anim.save(
        out_path,
        fps=24,
        dpi=100,
        writer="ffmpeg",
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "23"],
        progress_callback=_save_progress,
    )
    if verbose:
        print(f"\n[make_movie] wrote {out_path} in {time.time()-t_save:.1f}s")
    plt.close(fig)

    


if __name__ == "__main__":
    for i, sim_path in enumerate(sim_list):
        sim = StochasticGrowthStripGeometry.load(sim_path)
        movie_setting = movie_settings[i]
        make_movie(sim, num_frames=movie_setting['frames'], subsample=movie_setting['subsample'], verbose=True)

