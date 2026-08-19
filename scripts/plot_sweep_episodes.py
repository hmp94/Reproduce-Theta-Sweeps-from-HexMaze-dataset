"""Episodes of CONSECUTIVE sweeps, one figure per episode: every sweep of the
stretch on the actual path (left, even/odd cycle colours) and the same interval's
population firing rate with theta cycles (right). Sweeps numbered in both panels.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

BASE = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset"
DECODE = f"{BASE}/results/glm_shift/cache/Rat6_20260716_decode_bayes_s3_b5_relaxA.npz"
SESSION = f"{BASE}/results/glm_shift/cache/Rat6_20260716_extlfp_session.npz"
OUT = "/Users/sachuriga/Desktop/Rat6_20260716_sweep_episodes"
os.makedirs(OUT, exist_ok=True)

PX, BIN_S = 1.2435, 0.01
EVEN, ODD = "#0072b2", "#d55e00"
MIN_RUN = 3                      # episode = >= this many consecutive sweep-cycles

d = np.load(DECODE, allow_pickle=True)
s = np.load(SESSION)

track_x, track_y = d["track_x_px"] / PX, d["track_y_px"] / PX
onsets = d["cycle_onsets"]
is_sweep = d["is_sweep"].astype(bool)
length_cm = d["length_px"] / PX
origin_cm = d["origin_error_px"] / PX
paths = d["path_xy_px"]
start_bin, stop_bin = d["start_bin"], d["stop_bin"]
true_xy = d["true_xy_px"] / PX
n_cycles, n_bins = len(onsets), len(track_x)

pop_rate_hz = gaussian_filter1d(s["spike_counts"].sum(1).astype(float), 1.0) / BIN_S
theta_cos = np.cos(s["raw_phase"])

# --- episodes: maximal runs of consecutive sweep-cycles ----------------------
# Cycle c+1 must also start right after cycle c ends (no dropped-cycle gaps).
contiguous = np.r_[np.diff(onsets) * BIN_S < 0.25, False]
episodes = []
i = 0
while i < n_cycles:
    if not is_sweep[i]:
        i += 1
        continue
    j = i
    while j + 1 < n_cycles and is_sweep[j + 1] and contiguous[j]:
        j += 1
    if j - i + 1 >= MIN_RUN:
        episodes.append((i, j))
    i = j + 1

episodes.sort(key=lambda r: r[1] - r[0], reverse=True)
print(f"{len(episodes)} episodes of >= {MIN_RUN} consecutive sweeps; "
      f"lengths: {sorted([e - s0 + 1 for s0, e in episodes], reverse=True)[:15]} ...")

rows = []
for ep_index, (first, last) in enumerate(
        sorted(episodes, key=lambda r: r[0]), 1):
    drawn = [c for c in range(first, last + 1) if paths[c] is not None]
    n_cyc = last - first + 1
    t0 = onsets[first] * BIN_S
    t1 = (onsets[last + 1] if last + 1 < n_cycles else n_bins) * BIN_S

    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.14,
                          left=0.03, right=0.97, top=0.84, bottom=0.12)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1])

    # ---------------- left: all the episode's sweeps on the path ------------
    pad = int(0.4 / BIN_S)
    lo = max(onsets[first] - pad, 0)
    hi = min((onsets[last + 1] if last + 1 < n_cycles else n_bins - 1) + pad,
             n_bins - 1)

    pts = [paths[c] / PX for c in drawn]
    all_xy = np.vstack(pts + [np.column_stack([track_x[lo:hi], track_y[lo:hi]])])
    all_xy = all_xy[np.isfinite(all_xy).all(1)]
    cx, cy = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    half = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) / 2 + 12
    if not np.isfinite(half):
        half = 40.0

    inside = (np.abs(track_x - cx) < half) & (np.abs(track_y - cy) < half)
    ax.plot(track_x[inside][::5], track_y[inside][::5], ".", color="0.88", ms=1,
            zorder=0, rasterized=True)
    ax.plot(track_x[lo:hi], track_y[lo:hi], "-", color="0.15", lw=2.4, zorder=3,
            solid_capstyle="round")
    dx, dy = track_x[hi] - track_x[hi - 20], track_y[hi] - track_y[hi - 20]
    ax.annotate("", xy=(track_x[hi] + dx, track_y[hi] + dy),
                xytext=(track_x[hi], track_y[hi]),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=2), zorder=3)

    for k, c in enumerate(drawn, 1):
        xy = paths[c] / PX
        color = EVEN if c % 2 == 0 else ODD
        n = len(xy)
        if np.isfinite(true_xy[c, 0]):
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
    ax.plot([cx - half + 4, cx - half + 24], [cy - half + 4, cy - half + 4],
            "k-", lw=2)
    ax.text(cx - half + 14, cy - half + 7, "20 cm", ha="center", fontsize=8)

    # ---------------- right: population rate across the episode -------------
    rlo = max(onsets[first] - int(0.25 / BIN_S), 0)
    rhi = min((onsets[last + 1] if last + 1 < n_cycles else n_bins - 1)
              + int(0.25 / BIN_S), n_bins - 1)
    t = np.arange(rlo, rhi) * BIN_S

    for c in range(first, last + 1):
        c0 = onsets[c] * BIN_S
        c1 = (onsets[c + 1] if c + 1 < n_cycles else n_bins) * BIN_S
        axr.axvspan(c0, c1, color=(EVEN if c % 2 == 0 else ODD),
                    alpha=0.07, zorder=0)
    near = onsets[(onsets >= rlo) & (onsets < rhi)]
    for o in near:
        axr.axvline(o * BIN_S, color="0.84", lw=0.7, zorder=1)

    axr.plot(t, pop_rate_hz[rlo:rhi], "-", color="0.1", lw=1.4, zorder=3)
    rate_top = pop_rate_hz[rlo:rhi].max() * 1.06

    for k, c in enumerate(drawn, 1):
        color = EVEN if c % 2 == 0 else ODD
        ts0, ts1 = start_bin[c] * BIN_S, (stop_bin[c] + 1) * BIN_S
        axr.plot([ts0, ts1], [rate_top, rate_top], "-", color=color, lw=5,
                 solid_capstyle="butt", zorder=4)
        axr.annotate(str(k), xy=((ts0 + ts1) / 2, rate_top), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=9,
                     fontweight="bold", color=color, zorder=5)

    axr.plot(t, theta_cos[rlo:rhi] * 0.055 * rate_top + 0.08 * rate_top, "-",
             color="0.55", lw=0.9, zorder=2)
    axr.text(t[0] + 0.005, 0.15 * rate_top, "theta", fontsize=7.5, color="0.45")

    axr.set(xlim=(t[0], t[-1]), ylim=(0, rate_top * 1.13),
            xlabel="time (s)", ylabel="population rate (Hz)")
    axr.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"episode {ep_index}:  {t0:.1f}-{t1:.1f} s,  {len(drawn)} sweeps "
                 f"in {n_cyc} consecutive theta cycles  "
                 f"(even cycles blue, odd vermilion)", fontsize=11.5)
    fig.savefig(f"{OUT}/episode_{ep_index:03d}_t{t0:07.1f}s_{n_cyc}cyc.png", dpi=130)
    plt.close(fig)

    rows.append(dict(episode=ep_index, t_start_s=round(t0, 2), t_end_s=round(t1, 2),
                     n_cycles=n_cyc,
                     mean_length_cm=round(float(np.nanmean(length_cm[first:last + 1])), 1),
                     mean_origin_cm=round(float(np.nanmean(origin_cm[first:last + 1])), 1)))

pd.DataFrame(rows).to_csv(f"{OUT}/episode_index.csv", index=False)
print(f"wrote {len(rows)} episode figures + episode_index.csv -> {OUT}")
