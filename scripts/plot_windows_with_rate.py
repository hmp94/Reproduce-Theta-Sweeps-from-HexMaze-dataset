"""Best sweep windows: decoded sweeps on the ACTUAL path (left) with the same
interval's population firing rate + theta cycles (right). Sweeps are numbered in
both panels; theta cycles are shaded by even/odd parity in both panels' colours.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

BASE = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset"
DECODE = f"{BASE}/results/glm_shift/cache/Rat6_20260716_decode_bayes_s3_b5.npz"
SESSION = f"{BASE}/results/glm_shift/cache/Rat6_20260716_extlfp_session.npz"
OUT = "/Users/sachuriga/Desktop/Rat6_20260716_sweep_windows"
os.makedirs(OUT, exist_ok=True)

PX, BIN_S = 1.2435, 0.01
EVEN, ODD = "#0072b2", "#d55e00"

d = np.load(DECODE, allow_pickle=True)
s = np.load(SESSION)

track_x, track_y = d["track_x_px"] / PX, d["track_y_px"] / PX
onsets = d["cycle_onsets"]
is_sweep = d["is_sweep"].astype(bool)
origin_cm = d["origin_error_px"] / PX
length_cm = d["length_px"] / PX
paths = d["path_xy_px"]
start_bin, stop_bin = d["start_bin"], d["stop_bin"]
true_xy = d["true_xy_px"] / PX
n_cycles, n_bins = len(onsets), len(track_x)

pop_rate_hz = gaussian_filter1d(s["spike_counts"].sum(1).astype(float), 1.0) / BIN_S
theta_cos = np.cos(s["raw_phase"])

# --- pick windows: 12 consecutive cycles with the most anchored sweeps -------
good = is_sweep & (origin_cm <= 15.0) & (length_cm >= 12.0)
WIN = 12
score = np.array([good[i:i + WIN].sum() for i in range(n_cycles - WIN)])
order = np.argsort(score)[::-1]
windows, taken = [], np.zeros(n_cycles, bool)
for i in order:
    if score[i] < 3 or taken[i:i + WIN].any():
        continue
    windows.append(i)
    taken[max(i - WIN, 0):i + 2 * WIN] = True
    if len(windows) == 6:
        break
print(f"picked {len(windows)} windows, sweeps each: "
      f"{[int(good[i:i + WIN].sum()) for i in windows]}")

for w_index, first in enumerate(windows, 1):
    last = first + WIN - 1
    drawn = [c for c in range(first, last + 1) if good[c] and paths[c] is not None]

    lo = max(onsets[first] - 15, 0)
    hi = min(onsets[last + 1] + 15 if last + 1 < n_cycles else n_bins - 1, n_bins - 1)
    t = np.arange(lo, hi) * BIN_S

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1])

    # ---------------- left: the maze, actual path, sweeps -------------------
    pts = [paths[c] / PX for c in drawn]
    all_xy = np.vstack(pts + [np.column_stack([track_x[lo:hi], track_y[lo:hi]])])
    all_xy = all_xy[np.isfinite(all_xy).all(1)]
    cx, cy = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    half = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) / 2 + 12

    inside = (np.abs(track_x - cx) < half) & (np.abs(track_y - cy) < half)
    ax.plot(track_x[inside][::5], track_y[inside][::5], ".", color="0.88", ms=1,
            zorder=0, rasterized=True)
    ax.plot(track_x[lo:hi], track_y[lo:hi], "-", color="0.15", lw=2.5, zorder=3,
            solid_capstyle="round")
    dx, dy = track_x[hi] - track_x[hi - 25], track_y[hi] - track_y[hi - 25]
    ax.annotate("", xy=(track_x[hi] + dx, track_y[hi] + dy),
                xytext=(track_x[hi], track_y[hi]),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=2), zorder=3)

    for k, c in enumerate(drawn, 1):
        xy = paths[c] / PX
        color = EVEN if c % 2 == 0 else ODD
        n = len(xy)
        ax.plot(*true_xy[c], "o", color="0.15", ms=7, zorder=4,
                markerfacecolor="w", markeredgewidth=1.6)
        for j in range(n - 1):
            ax.plot(xy[j:j + 2, 0], xy[j:j + 2, 1], "-", color=color,
                    alpha=0.35 + 0.65 * j / max(n - 2, 1), lw=2.2, zorder=5)
        ax.scatter(xy[:, 0], xy[:, 1], s=26, c=[color] * n,
                   alpha=np.linspace(0.45, 1.0, n), zorder=6, edgecolors="none")
        ax.annotate("", xy=xy[-1], xytext=xy[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2), zorder=7)
        ax.annotate(str(k), xy=xy[-1], xytext=(xy[-1][0] + 2.5, xy[-1][1] + 2.5),
                    fontsize=10, fontweight="bold", color=color, zorder=8)

    ax.set(xlim=(cx - half, cx + half), ylim=(cy - half, cy + half),
           xticks=[], yticks=[], aspect="equal")
    ax.plot([cx - half + 5, cx - half + 25], [cy - half + 5, cy - half + 5], "k-", lw=2)
    ax.text(cx - half + 15, cy - half + 8, "20 cm", ha="center", fontsize=8)
    ax.set_title("decoded sweeps on the actual path", fontsize=11)

    # ---------------- right: population rate + theta cycles -----------------
    # alternating cycle shading, matching the sweep colours
    for c in range(first, last + 1):
        c0 = onsets[c] * BIN_S
        c1 = (onsets[c + 1] if c + 1 < n_cycles else n_bins) * BIN_S
        axr.axvspan(c0, c1, color=(EVEN if c % 2 == 0 else ODD), alpha=0.06, zorder=0)
        axr.axvline(c0, color="0.8", lw=0.6, zorder=1)

    axr.plot(t, pop_rate_hz[lo:hi], "-", color="0.1", lw=1.4, zorder=3)
    rate_top = pop_rate_hz[lo:hi].max() * 1.06

    # sweep intervals as numbered bars riding above the rate trace
    for k, c in enumerate(drawn, 1):
        color = EVEN if c % 2 == 0 else ODD
        t0, t1 = start_bin[c] * BIN_S, (stop_bin[c] + 1) * BIN_S
        axr.plot([t0, t1], [rate_top, rate_top], "-", color=color, lw=5,
                 solid_capstyle="butt", zorder=4)
        axr.annotate(str(k), xy=((t0 + t1) / 2, rate_top), xytext=(0, 5),
                     textcoords="offset points", ha="center", fontsize=10,
                     fontweight="bold", color=color, zorder=5)

    # theta rhythm strip along the bottom
    theta_band = theta_cos[lo:hi]
    axr.plot(t, theta_band * 0.06 * rate_top + 0.08 * rate_top, "-",
             color="0.55", lw=0.9, zorder=2)
    axr.text(t[0], 0.15 * rate_top, "theta (5-10 Hz phase)", fontsize=7.5,
             color="0.45", va="bottom")

    axr.set(xlim=(t[0], t[-1]), ylim=(0, rate_top * 1.12),
            xlabel="time (s)", ylabel="population rate (Hz)")
    axr.spines[["top", "right"]].set_visible(False)
    axr.set_title("population firing rate, theta cycles shaded by parity", fontsize=11)

    t0_w = onsets[first] * BIN_S
    fig.suptitle(f"window {w_index}:  {t0_w:.1f} s,  {len(drawn)} sweeps in "
                 f"{WIN} theta cycles  (even cycles blue, odd vermilion)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{OUT}/window_{w_index}_t{t0_w:07.1f}s.png", dpi=140)
    plt.close(fig)
    print(f"  window {w_index} ({t0_w:.1f}s): {len(drawn)} sweeps")

print(f"wrote {len(windows)} figures -> {OUT}")
