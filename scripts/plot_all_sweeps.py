"""One PNG per detected sweep: the sweep on the ACTUAL path (left) and the same
interval's population firing rate with theta cycles (right). Plus an index CSV."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset")
BASE = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset"
DECODE = f"{BASE}/results/glm_shift/cache/Rat6_20260716_decode_bayes_s3_b5_relaxA.npz"
SESSION = f"{BASE}/results/glm_shift/cache/Rat6_20260716_extlfp_session.npz"
OUT = "/Users/sachuriga/Desktop/Rat6_20260716_all_sweeps"
os.makedirs(OUT, exist_ok=True)

PX, BIN_S = 1.2435, 0.01
BLUE = "#0072b2"

d = np.load(DECODE, allow_pickle=True)
s = np.load(SESSION)

track_x, track_y = d["track_x_px"] / PX, d["track_y_px"] / PX
speed_cm = d["speed_px_s"] / PX
onsets = d["cycle_onsets"]
is_sweep = d["is_sweep"].astype(bool)
origin_cm = d["origin_error_px"] / PX
length_cm = d["length_px"] / PX
paths = d["path_xy_px"]
start_bin, stop_bin = d["start_bin"], d["stop_bin"]
true_xy = d["true_xy_px"] / PX
n_cycles, n_bins = len(onsets), len(track_x)
cycle_centers = np.clip((onsets + np.r_[onsets[1:], n_bins]) // 2, 0, n_bins - 1)
heading = s["head_direction"][cycle_centers]

pop_rate_hz = gaussian_filter1d(s["spike_counts"].sum(1).astype(float), 1.0) / BIN_S
theta_cos = np.cos(s["raw_phase"])

sweep_ids = np.where(is_sweep)[0]
print(f"{len(sweep_ids)} sweeps -> {OUT}")

rows = []
for count, c in enumerate(sweep_ids, 1):
    xy = paths[c] / PX
    t0 = onsets[c] * BIN_S

    v = xy[-1] - true_xy[c]
    lateral = np.degrees((np.arctan2(v[1], v[0]) - heading[c] + np.pi)
                         % (2 * np.pi) - np.pi)
    speed = speed_cm[cycle_centers[c]]
    rows.append(dict(index=count, cycle=int(c), t_s=round(t0, 2),
                     length_cm=round(float(length_cm[c]), 1),
                     origin_cm=round(float(origin_cm[c]), 1),
                     lateral_deg=round(float(lateral), 1),
                     speed_cm_s=round(float(speed), 1)))

    fig = plt.figure(figsize=(10.5, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.16,
                          left=0.03, right=0.97, top=0.86, bottom=0.13)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1])

    # ---------------- left: the sweep on the actual path --------------------
    pad = int(0.5 / BIN_S)
    lo = max(onsets[c] - pad, 0)
    hi = min((onsets[c + 1] if c + 1 < n_cycles else n_bins - 1) + pad, n_bins - 1)

    all_xy = np.vstack([xy, np.column_stack([track_x[lo:hi], track_y[lo:hi]])])
    all_xy = all_xy[np.isfinite(all_xy).all(1)]
    if len(all_xy) < 2:
        plt.close(fig)
        continue
    cx, cy = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    half = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) / 2 + 12
    if not np.isfinite(half):
        half = 40.0

    inside = (np.abs(track_x - cx) < half) & (np.abs(track_y - cy) < half)
    ax.plot(track_x[inside][::5], track_y[inside][::5], ".", color="0.88", ms=1,
            zorder=0, rasterized=True)
    ax.plot(track_x[lo:hi], track_y[lo:hi], "-", color="0.15", lw=2.2, zorder=3,
            solid_capstyle="round")
    dx, dy = track_x[hi] - track_x[hi - 20], track_y[hi] - track_y[hi - 20]
    ax.annotate("", xy=(track_x[hi] + dx, track_y[hi] + dy),
                xytext=(track_x[hi], track_y[hi]),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.8), zorder=3)

    hd, (ox, oy) = heading[c], true_xy[c]
    if np.isfinite(ox) and np.isfinite(hd):
        ax.plot([ox - 6 * np.cos(hd), ox + half * 0.9 * np.cos(hd)],
                [oy - 6 * np.sin(hd), oy + half * 0.9 * np.sin(hd)],
                "--", color="0.45", lw=1.1, zorder=2, dashes=(5, 4))
        ax.plot(ox, oy, "o", color="0.15", ms=8, zorder=4,
                markerfacecolor="w", markeredgewidth=2)

    n = len(xy)
    for k in range(n - 1):
        ax.plot(xy[k:k + 2, 0], xy[k:k + 2, 1], "-", color=BLUE,
                alpha=0.35 + 0.65 * k / max(n - 2, 1), lw=2.4, zorder=5)
    ax.scatter(xy[:, 0], xy[:, 1], s=32, c=[BLUE] * n,
               alpha=np.linspace(0.45, 1.0, n), zorder=6, edgecolors="none")
    ax.annotate("", xy=xy[-1], xytext=xy[-2],
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2), zorder=7)

    ax.set(xlim=(cx - half, cx + half), ylim=(cy - half, cy + half),
           xticks=[], yticks=[], aspect="equal")
    ax.plot([cx - half + 4, cx - half + 24], [cy - half + 4, cy - half + 4],
            "k-", lw=2)
    ax.text(cx - half + 14, cy - half + 7, "20 cm", ha="center", fontsize=7)

    # ---------------- right: population rate around the sweep ---------------
    rlo = max(onsets[c] - int(0.30 / BIN_S), 0)
    rhi = min((onsets[c + 1] if c + 1 < n_cycles else n_bins - 1)
              + int(0.30 / BIN_S), n_bins - 1)
    t = np.arange(rlo, rhi) * BIN_S

    # this sweep's own theta cycle, shaded; neighbours' onsets as thin lines
    c0 = onsets[c] * BIN_S
    c1 = (onsets[c + 1] if c + 1 < n_cycles else n_bins) * BIN_S
    axr.axvspan(c0, c1, color=BLUE, alpha=0.07, zorder=0)
    near = onsets[(onsets >= rlo) & (onsets < rhi)]
    for o in near:
        axr.axvline(o * BIN_S, color="0.82", lw=0.7, zorder=1)

    axr.plot(t, pop_rate_hz[rlo:rhi], "-", color="0.1", lw=1.4, zorder=3)
    rate_top = pop_rate_hz[rlo:rhi].max() * 1.06

    ts0, ts1 = start_bin[c] * BIN_S, (stop_bin[c] + 1) * BIN_S
    axr.plot([ts0, ts1], [rate_top, rate_top], "-", color=BLUE, lw=6,
             solid_capstyle="butt", zorder=4)
    axr.text((ts0 + ts1) / 2, rate_top, "sweep", ha="center", va="bottom",
             fontsize=8, color=BLUE, fontweight="bold")

    axr.plot(t, theta_cos[rlo:rhi] * 0.06 * rate_top + 0.09 * rate_top, "-",
             color="0.55", lw=0.9, zorder=2)
    axr.text(t[0] + 0.01, 0.16 * rate_top, "theta", fontsize=7.5, color="0.45")

    axr.set(xlim=(t[0], t[-1]), ylim=(0, rate_top * 1.14),
            xlabel="time (s)", ylabel="population rate (Hz)")
    axr.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"#{count}  t={t0:.1f}s  len {length_cm[c]:.0f}cm  "
                 f"origin {origin_cm[c]:.0f}cm  lat {lateral:+.0f}\N{DEGREE SIGN}  "
                 f"{speed:.0f}cm/s", fontsize=10)
    fig.savefig(f"{OUT}/sweep_{count:03d}_t{t0:07.1f}s.png", dpi=115)
    plt.close(fig)
    if count % 50 == 0:
        print(f"  {count}/{len(sweep_ids)}")

pd.DataFrame(rows).to_csv(f"{OUT}/sweep_index.csv", index=False)
print(f"done: {len(rows)} PNGs + sweep_index.csv in {OUT}")
