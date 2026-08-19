"""Paper-style examples: LEFT/RIGHT-ALTERNATING theta sweeps around the moving
direction, decoded trajectories on the rat's path, coloured by even/odd cycle.

For each accepted sweep the signed lateral angle is the direction of its far end
(seen from the animal) minus the moving direction: positive = left of travel,
negative = right. Windows shown are runs of consecutive sweep-cycles whose sign
strictly alternates -- the phenomenon of Vollan et al. Figs 1-2. The moving
direction is drawn as a dashed midline so left/right reads directly.

Needs the decode dump written on the first run (results/glm_shift/cache/).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset")
from hexmaze_sweeps.config import Config

OUT = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset/results/glm_shift"
DUMP = f"{OUT}/cache/Rat6_20260716_decode_bayes_s3_b5_relaxA.npz"
config = Config()
px = config.px_per_cm

z = np.load(DUMP, allow_pickle=True)
track_x, track_y = z["track_x_px"] / px, z["track_y_px"] / px
onsets = z["cycle_onsets"]
is_sweep = z["is_sweep"].astype(bool)
is_running = z["is_running"].astype(bool)
origin_cm = z["origin_error_px"] / px
length_cm = z["length_px"] / px
paths = z["path_xy_px"]
true_xy = z["true_xy_px"] / px          # the animal, per cycle
n_cycles = len(onsets)
n_bins_total = len(track_x)
cycle_centers = np.clip((onsets + np.r_[onsets[1:], n_bins_total]) // 2,
                        0, n_bins_total - 1)
heading = z["head_direction"][cycle_centers]

# --- signed lateral angle of each sweep, relative to the moving direction ----
lateral = np.full(n_cycles, np.nan)
for c in range(n_cycles):
    if paths[c] is not None and len(paths[c]) >= 2 and np.isfinite(true_xy[c, 0]):
        v = paths[c][-1] / px - true_xy[c]
        lateral[c] = (np.arctan2(v[1], v[0]) - heading[c] + np.pi) % (2 * np.pi) - np.pi

# In a corridor maze the decoded positions are confined to the corridors, so
# left/right fanning around the heading can only happen where corridors DIVERGE.
# Look for junction approaches where consecutive sweeps point down clearly
# different directions (the hex-maze counterpart of the paper's L/R alternation).
import pandas as pd
node_xy = pd.read_csv(
    "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset/node_list_new.csv",
    header=None, names=["id", "x", "y"])[["x", "y"]].values / px
node_dist = np.full(n_cycles, np.inf)
ok_xy = np.isfinite(true_xy[:, 0])
node_dist[ok_xy] = np.min(
    np.hypot(true_xy[ok_xy, None, 0] - node_xy[None, :, 0],
             true_xy[ok_xy, None, 1] - node_xy[None, :, 1]), axis=1)

bearing = np.full(n_cycles, np.nan)
for c in range(n_cycles):
    if paths[c] is not None and len(paths[c]) >= 2 and ok_xy[c]:
        v = paths[c][-1] / px - true_xy[c]
        bearing[c] = np.arctan2(v[1], v[0])

eligible = (is_sweep & is_running & (origin_cm <= 15.0) & (length_cm >= 12.0)
            & (np.abs(lateral) <= np.radians(110)) & (node_dist <= 30.0))

def ang_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)

runs = []
i = 0
while i < n_cycles:
    if not eligible[i]:
        i += 1
        continue
    j = i
    while (j + 1 < n_cycles and eligible[j + 1]
           and ang_diff(bearing[j + 1], bearing[j]) >= np.radians(35)):
        j += 1
    if j > i:
        runs.append((i, j))
    i = j + 1

runs.sort(key=lambda r: (r[1] - r[0],
                         np.nanmean([ang_diff(bearing[c + 1], bearing[c])
                                     for c in range(r[0], r[1])])), reverse=True)
print(f"{eligible.sum()} eligible junction sweeps; diverging runs >=2: {len(runs)}; "
      f"lengths of best: {[e - s + 1 for s, e in runs[:10]]}")

EVEN, ODD = "#0072b2", "#d55e00"

def draw_example(ax, first, last):
    pad_bins = int(0.4 / config.bin_s)
    lo_bin = max(onsets[max(first - 1, 0)] - pad_bins, 0)
    hi_bin = min(onsets[min(last + 2, n_cycles - 1)] + pad_bins, n_bins_total - 1)

    drawn = [c for c in range(first, last + 1) if paths[c] is not None]
    pts = [paths[c] / px for c in drawn]
    all_xy = np.vstack(pts + [np.column_stack([track_x[lo_bin:hi_bin],
                                               track_y[lo_bin:hi_bin]])])
    cx, cy = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    half = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) / 2 + 12

    inside = (np.abs(track_x - cx) < half) & (np.abs(track_y - cy) < half)
    ax.plot(track_x[inside][::5], track_y[inside][::5], ".", color="0.88", ms=1,
            zorder=0, rasterized=True)

    # rat's path, arrow at its end
    ax.plot(track_x[lo_bin:hi_bin], track_y[lo_bin:hi_bin], "-", color="0.15",
            lw=2.5, zorder=3, solid_capstyle="round")
    dx = track_x[hi_bin] - track_x[hi_bin - 25]
    dy = track_y[hi_bin] - track_y[hi_bin - 25]
    ax.annotate("", xy=(track_x[hi_bin] + dx, track_y[hi_bin] + dy),
                xytext=(track_x[hi_bin], track_y[hi_bin]),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=2), zorder=3)

    # the MIDLINE: moving direction from the middle of the run
    mid = drawn[len(drawn) // 2]
    ox, oy = true_xy[mid]
    hd = heading[mid]
    ray = np.array([np.cos(hd), np.sin(hd)])
    ax.plot([ox - 8 * ray[0], ox + (half * 0.95) * ray[0]],
            [oy - 8 * ray[1], oy + (half * 0.95) * ray[1]],
            "--", color="0.45", lw=1.4, zorder=2, dashes=(5, 4))

    for c in drawn:
        cb = cycle_centers[c]
        ax.plot(track_x[cb], track_y[cb], "o", color="0.15", ms=7, zorder=4,
                markerfacecolor="w", markeredgewidth=1.8)

    for c in drawn:
        xy = paths[c] / px
        color = EVEN if c % 2 == 0 else ODD
        n = len(xy)
        for k in range(n - 1):
            ax.plot(xy[k:k + 2, 0], xy[k:k + 2, 1], "-", color=color,
                    alpha=0.35 + 0.65 * k / max(n - 2, 1), lw=2.2, zorder=4)
        ax.scatter(xy[:, 0], xy[:, 1], s=28, c=[color] * n,
                   alpha=np.linspace(0.45, 1.0, n), zorder=5, edgecolors="none")
        ax.annotate("", xy=xy[-1], xytext=xy[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2), zorder=6)
        side = "L" if lateral[c] > 0 else "R"
        ax.annotate(side, xy=xy[-1], xytext=(xy[-1][0] + 2, xy[-1][1] + 2),
                    fontsize=9, color=color, fontweight="bold", zorder=6)

    t0, t1 = onsets[first] * config.bin_s, onsets[last + 1] * config.bin_s
    speed = np.nanmean(z["speed_px_s"][lo_bin:hi_bin]) / px
    sides = "".join("L" if lateral[c] > 0 else "R" for c in drawn)
    ax.set(xlim=(cx - half, cx + half), ylim=(cy - half, cy + half),
           xticks=[], yticks=[], aspect="equal")
    ax.set_title(f"{t0:.1f}-{t1:.1f} s | {sides} | {speed:.0f} cm/s", fontsize=10)
    ax.plot([cx - half + 5, cx - half + 25], [cy - half + 5, cy - half + 5],
            "k-", lw=2)
    ax.text(cx - half + 15, cy - half + 8, "20 cm", ha="center", fontsize=8)

n_show = min(6, len(runs))
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (s, e) in zip(axes.ravel(), runs[:n_show]):
    draw_example(ax, s, e)
for ax in axes.ravel()[n_show:]:
    ax.axis("off")

fig.suptitle("Left/right-alternating theta sweeps around the moving direction "
             "(dashed midline) -- even cycles blue, odd cycles vermilion",
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out_png = "/Users/sachuriga/Desktop/Rat6_20260716_examples_evenodd.png"
fig.savefig(out_png, dpi=150)
print(f"wrote {out_png}")
