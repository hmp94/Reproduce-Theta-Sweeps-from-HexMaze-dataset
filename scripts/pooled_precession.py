"""Pooled phase precession across all cells, from the local session cache.

Cells are chosen by in-field running spike count (not bits/spike, which favours
sparse cells). Each cell contributes (distance-along-moving-direction from its
field peak, theta phase) for every running spike; pooling across cells gives the
population-level test that single noisy cells cannot.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

CACHE = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset/results/glm_shift/cache/Rat6_20260716_extlfp_session.npz"
OUT = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset/results/glm_shift"
PX_PER_CM = 1.2435
BIN_S = 0.01

z = np.load(CACHE)
xy_cm = np.column_stack([z["track_x_px"], z["track_y_px"]]) / PX_PER_CM
speed = z["speed_px_s"] / PX_PER_CM
counts = z["spike_counts"]
moving_dir = z["head_direction"]
phase = z["raw_phase"]

running = np.isfinite(speed) & (speed >= 15.0)
print(f"{running.sum()} running bins, {counts.shape[1]} cells")

# --- rate maps to find field peaks -------------------------------------------
bs = 2.5
gx = np.arange(xy_cm[running, 0].min(), xy_cm[running, 0].max() + bs, bs)
gy = np.arange(xy_cm[running, 1].min(), xy_cm[running, 1].max() + bs, bs)
occ, _, _ = np.histogram2d(xy_cm[running, 0], xy_cm[running, 1], [gx, gy])
occ_s = gaussian_filter(occ * BIN_S, 2.0)

FIELD_RADIUS = 20.0
pooled_s, pooled_ph, per_cell = [], [], []
for i in range(counts.shape[1]):
    w = counts[running, i]
    if w.sum() < 100:
        continue
    spk, _, _ = np.histogram2d(xy_cm[running, 0], xy_cm[running, 1], [gx, gy], weights=w)
    rate = gaussian_filter(spk, 2.0) / np.maximum(occ_s, 0.25)
    rate[occ_s < 0.25] = 0
    pi, pj = np.unravel_index(np.argmax(rate), rate.shape)
    peak = np.array([gx[pi] + bs / 2, gy[pj] + bs / 2])

    near = running & (np.hypot(*(xy_cm - peak).T) <= FIELD_RADIUS)
    s = ((xy_cm[near] - peak) * np.column_stack([np.cos(moving_dir[near]),
                                                 np.sin(moving_dir[near])])).sum(1)
    wn = counts[near, i]
    keep = wn > 0
    n_spk = int(wn[keep].sum())
    if n_spk < 100:
        continue
    s_sp = np.repeat(s[keep], wn[keep].astype(int))
    ph_sp = np.repeat(phase[near][keep], wn[keep].astype(int))
    pooled_s.append(s_sp)
    pooled_ph.append(ph_sp)

    cand = np.radians(np.linspace(-30, 30, 241))
    R = [np.abs(np.exp(1j * (ph_sp - a * s_sp)).mean()) for a in cand]
    per_cell.append((i, n_spk, np.degrees(cand[int(np.argmax(R))]),
                     float(np.max(R))))

pooled_s = np.concatenate(pooled_s)
pooled_ph = np.concatenate(pooled_ph)
slopes = np.array([c[2] for c in per_cell])
print(f"{len(per_cell)} cells with >=100 in-field spikes, {len(pooled_s)} pooled spikes")
print(f"per-cell slopes: median {np.median(slopes):+.1f} deg/cm, "
      f"{np.mean(slopes < 0) * 100:.0f}% negative")

# pooled circular-linear fit
cand = np.radians(np.linspace(-30, 30, 481))
R = np.array([np.abs(np.exp(1j * (pooled_ph - a * pooled_s)).mean()) for a in cand])
a_best = cand[int(np.argmax(R))]
print(f"pooled slope {np.degrees(a_best):+.2f} deg/cm  (R = {R.max():.3f})")

# shuffle: rotate each cell's phases by a random offset (breaks s-phase pairing
# within cells but keeps each cell's phase distribution)
rng = np.random.default_rng(0)
null_R = []
lengths = [len(s) for s in np.split(pooled_s, np.cumsum([c[1] for c in per_cell])[:-1])]
for _ in range(200):
    ph_shuf = np.concatenate([
        seg + rng.uniform(0, 2 * np.pi)
        for seg in np.split(pooled_ph, np.cumsum([c[1] for c in per_cell])[:-1])])
    idx = rng.permutation(len(pooled_s))
    Rn = np.max([np.abs(np.exp(1j * (ph_shuf - a * pooled_s[idx])).mean())
                 for a in cand[::10]])
    null_R.append(Rn)
print(f"pooled R {R.max():.3f} vs shuffle 97.5th {np.percentile(null_R, 97.5):.3f}")

# --- figure ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
h, xe, ye = np.histogram2d(pooled_s, np.degrees(pooled_ph),
                           [np.arange(-20, 21, 2), np.arange(-180, 181, 15)])
h = h / np.maximum(h.sum(1, keepdims=True), 1)      # normalise per distance column
axes[0].imshow(np.tile(h, (1, 2)).T, aspect="auto", origin="lower", cmap="viridis",
               extent=[-20, 20, -180, 540])
axes[0].plot([-20, 20], np.degrees(a_best * np.array([-20, 20])) + 90, "r--", lw=1.5)
axes[0].set(xlabel="dist along moving dir from field peak (cm)",
            ylabel="theta phase (deg)",
            title=f"pooled ({len(pooled_s)} spikes, {len(per_cell)} cells), "
                  f"slope {np.degrees(a_best):+.1f} deg/cm")

axes[1].hist(slopes, bins=np.arange(-32, 33, 4), color="C0")
axes[1].axvline(0, color="0.4", lw=1)
axes[1].axvline(np.median(slopes), color="C3", lw=2,
                label=f"median {np.median(slopes):+.1f}")
axes[1].set(xlabel="per-cell precession slope (deg/cm)", ylabel="cells")
axes[1].legend(frameon=False)

axes[2].plot(np.degrees(cand), R, "k-")
axes[2].axhline(np.percentile(null_R, 97.5), color="0.6", ls="--",
                label="shuffle 97.5%")
axes[2].axvline(np.degrees(a_best), color="C3", lw=1)
axes[2].set(xlabel="slope (deg/cm)", ylabel="resultant R", title="pooled fit")
axes[2].legend(frameon=False)

fig.suptitle("pooled phase precession, cells selected by in-field spike count")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{OUT}/Rat6_20260716_pooled_precession.png", dpi=130)
print("wrote pooled_precession.png")
