"""Rat-centred sweep overlay, the paper's Fig. 2 convention: every anchored sweep
translated to the animal's position and rotated so the moving direction points UP.
Forward sweeps then radiate upward from the centre; left/right separate sideways.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUMP = ("/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset/"
        "results/glm_shift/cache/Rat6_20260716_decode_bayes_s3_b5_relaxA.npz")
PX = 1.2435
ORIGIN_MAX_CM = 15.0

z = np.load(DUMP, allow_pickle=True)
onsets = z["cycle_onsets"]
n_cycles = len(onsets)
n_bins_total = len(z["track_x_px"])
cycle_centers = np.clip((onsets + np.r_[onsets[1:], n_bins_total]) // 2,
                        0, n_bins_total - 1)
heading = z["head_direction"][cycle_centers]
is_sweep = z["is_sweep"].astype(bool)
origin_cm = z["origin_error_px"] / PX
paths, true_xy = z["path_xy_px"], z["true_xy_px"] / PX

keep = np.where(is_sweep & (origin_cm <= ORIGIN_MAX_CM))[0]
print(f"{len(keep)} anchored sweeps (origin <= {ORIGIN_MAX_CM:.0f} cm)")

EVEN, ODD = "#0072b2", "#d55e00"
fig = plt.figure(figsize=(13.5, 6.6))
ax = fig.add_subplot(121)

# Tracking is in image pixels (y down), so rotating heading to "up" needs the
# mirrored angle; using it consistently for every sweep keeps left/right honest.
far_angles = []
for c in keep:
    xy = paths[c] / PX - true_xy[c]                  # rat at the origin
    rot = -(heading[c]) + np.pi / 2                  # moving direction -> +y
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    v = xy @ R.T
    v[:, 0] = -v[:, 0]           # image y-axis points down: mirror so left is left
    color = EVEN if c % 2 == 0 else ODD
    n = len(v)
    for k in range(n - 1):
        ax.plot(v[k:k + 2, 0], v[k:k + 2, 1], "-", color=color, lw=1.2,
                alpha=0.10 + 0.25 * k / max(n - 2, 1), zorder=2)
    ax.plot(*v[-1], ".", color=color, ms=5, alpha=0.55, zorder=3)
    far_angles.append(np.arctan2(v[-1, 1], v[-1, 0]))

far_angles = np.asarray(far_angles)

ax.plot(0, 0, "o", color="k", ms=10, zorder=5)
ax.annotate("", xy=(0, 18), xytext=(0, 8),
            arrowprops=dict(arrowstyle="-|>", color="k", lw=2.5), zorder=5)
ax.text(1.5, 12, "moving\ndirection", fontsize=9)
for r in (20, 40):
    ax.add_patch(plt.Circle((0, 0), r, fill=False, color="0.85", lw=0.8, zorder=1))
    ax.text(r * 0.71, -r * 0.74, f"{r} cm", fontsize=7, color="0.6")
lim = 45
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[],
       title=f"all {len(keep)} anchored sweeps, rat-centred, forward = up\n"
             f"(even cycles blue, odd vermilion; dot = far end)")

# polar histogram of far-end directions (0 = forward)
axp = fig.add_subplot(122, projection="polar")
rel = (far_angles - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi   # 0 = forward
bins = np.linspace(-np.pi, np.pi, 25)
counts, _ = np.histogram(rel, bins)
axp.bar((bins[:-1] + bins[1:]) / 2 + np.pi / 2, counts,          # draw forward-up
        width=np.diff(bins), color="#0072b2", alpha=0.75, edgecolor="w")
axp.set_theta_zero_location("E")
axp.set(title="far-end direction relative to heading\n(up = forward)")
axp.set_xticklabels([])

forward_frac = np.mean(np.abs(rel) <= np.radians(60))
print(f"far end within 60 deg of heading: {forward_frac * 100:.0f}%")

fig.tight_layout()
out = "/Users/sachuriga/Desktop/Rat6_20260716_sweeps_ratcentered.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
