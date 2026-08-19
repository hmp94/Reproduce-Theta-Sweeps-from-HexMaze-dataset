"""Visualising the two findings: (A,B) backward sweeps concentrate at the goal --
sweep arrows and %-backward map on the maze; (C,D) F/B modes persist in streaks --
episode barcode and streak-length distribution vs the permutation null.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset"
PX = 1.2435
F_COL, B_COL, L_COL = "#0072b2", "#d55e00", "0.75"

z = np.load(f"{BASE}/results/glm_shift/cache/Rat6_20260716_decode_bayes_s3_b5_relaxA.npz",
            allow_pickle=True)
df = pd.read_csv(f"{BASE}/results/glm_shift/Rat6_20260716_goal_sweeps.csv")

onsets = z["cycle_onsets"]
n_bins = len(z["track_x_px"])
cc = np.clip((onsets + np.r_[onsets[1:], n_bins]) // 2, 0, n_bins - 1)
heading = z["head_direction"][cc]
track_x, track_y = z["track_x_px"] / PX, z["track_y_px"] / PX

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

df["sweep_head"] = np.degrees(np.abs(wrap(df.bearing.values - heading[df.cycle.values])))
df["label"] = np.where(df.sweep_head <= 60, "F",
                       np.where(df.sweep_head >= 120, "B", "L"))
nodes = pd.read_csv(f"{BASE}/node_list_new.csv", header=None, names=["id", "x", "y"])
goal = nodes.loc[nodes.id == 421, ["x", "y"]].values[0] / PX

fig = plt.figure(figsize=(19, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.25, wspace=0.12)

# ---------------- A: every F/B sweep as an arrow on the maze -----------------
ax = fig.add_subplot(gs[0, 0])
ax.plot(track_x[::20], track_y[::20], ".", color="0.9", ms=1, zorder=0, rasterized=True)
for label, color, zo in [("F", F_COL, 2), ("B", B_COL, 3)]:
    sub = df[df.label == label]
    ax.quiver(sub.animal_x, sub.animal_y,
              np.cos(sub.bearing), np.sin(sub.bearing),
              color=color, scale=45, width=0.0022, alpha=0.55,
              headwidth=3.5, zorder=zo, label=f"{label} (n={len(sub)})")
ax.plot(*goal, "*", color="k", ms=26, zorder=5, markerfacecolor="#ffd700",
        markeredgewidth=1.5, label="goal")
ax.set(aspect="equal", xticks=[], yticks=[],
       title="every sweep as an arrow at the rat's position (blue = forward, "
             "vermilion = backward)")
ax.legend(frameon=False, loc="upper left", fontsize=9)

# ---------------- B: spatial fraction of backward sweeps ---------------------
ax = fig.add_subplot(gs[1, 0])
BS = 30.0
fb = df[df.label != "L"]
x_edges = np.arange(track_x.min(), track_x.max() + BS, BS)
y_edges = np.arange(track_y.min(), track_y.max() + BS, BS)
n_all, _, _ = np.histogram2d(fb.animal_x, fb.animal_y, [x_edges, y_edges])
n_b, _, _ = np.histogram2d(fb[fb.label == "B"].animal_x, fb[fb.label == "B"].animal_y,
                           [x_edges, y_edges])
frac = np.where(n_all >= 8, n_b / np.maximum(n_all, 1), np.nan)
im = ax.imshow(frac.T * 100, origin="lower", cmap="RdBu_r", vmin=0, vmax=100,
               extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
ax.plot(track_x[::20], track_y[::20], ".", color="0.75", ms=0.8, zorder=1,
        rasterized=True)
ax.plot(*goal, "*", color="k", ms=26, zorder=5, markerfacecolor="#ffd700",
        markeredgewidth=1.5)
ax.set(aspect="equal", xticks=[], yticks=[],
       title="% backward per 30 cm bin (>= 8 sweeps) -- red = retrospective zone")
fig.colorbar(im, ax=ax, label="% backward", shrink=0.8)

# ---------------- C: episode barcode -----------------------------------------
lab = pd.Series(index=df.cycle.values, data=df.label.values)
cycles = np.sort(lab.index.values)
episodes, i = [], 0
while i < len(cycles):
    j = i
    while j + 1 < len(cycles) and cycles[j + 1] == cycles[j] + 1:
        j += 1
    if j - i + 1 >= 4:
        episodes.append(cycles[i:j + 1])
    i = j + 1
episodes.sort(key=len, reverse=True)
episodes = episodes[:60]

max_len = max(len(e) for e in episodes)
code = {"F": 0, "B": 1, "L": 2}
mat = np.full((len(episodes), max_len), 3.0)
for r, ep in enumerate(episodes):
    for k, c in enumerate(ep):
        mat[r, k] = code[lab[c]]

ax = fig.add_subplot(gs[0, 1])
ax.imshow(mat, aspect="auto", cmap=ListedColormap([F_COL, B_COL, L_COL, "white"]),
          vmin=0, vmax=3, interpolation="nearest")
ax.set(xlabel="theta cycle within episode", ylabel="episode (sorted by length)",
       title=f"episode barcode ({len(episodes)} episodes of >= 4 sweeps): "
             "F/B come in streaks, not alternation")
ax.spines[["top", "right"]].set_visible(False)

# ---------------- D: streak lengths, observed vs permuted --------------------
def streaks(sequences):
    lengths = []
    for seq in sequences:
        seq = [s for s in seq if s in "FB"]
        k = 0
        while k < len(seq):
            j = k
            while j + 1 < len(seq) and seq[j + 1] == seq[j]:
                j += 1
            lengths.append(j - k + 1)
            k = j + 1
    return np.array(lengths)

ep_labels = [[lab[c] for c in ep] for ep in episodes]
obs = streaks(ep_labels)
rng = np.random.default_rng(0)
null_counts = []
edges = np.arange(0.5, 11.5)
for _ in range(500):
    shuffled = []
    for seq in ep_labels:
        s = list(seq)
        rng.shuffle(s)
        shuffled.append(s)
    null_counts.append(np.histogram(np.clip(streaks(shuffled), 0, 10), edges)[0])
null_counts = np.array(null_counts)
obs_counts = np.histogram(np.clip(obs, 0, 10), edges)[0]

ax = fig.add_subplot(gs[1, 1])
centers = np.arange(1, 11)
ax.bar(centers - 0.2, obs_counts, 0.4, color="#333", label="observed")
ax.bar(centers + 0.2, null_counts.mean(0), 0.4, color="0.75", label="shuffled order")
ax.errorbar(centers + 0.2, null_counts.mean(0),
            yerr=1.96 * null_counts.std(0), fmt="none", ecolor="0.5")
ax.set(xlabel="streak length (consecutive same-mode sweeps)", ylabel="count",
       title=f"mode streaks are longer than chance (median obs "
             f"{np.median(obs):.0f}; long streaks >= 4: {np.sum(obs >= 4)} vs "
             f"{null_counts[:, 3:].sum(1).mean():.0f} shuffled)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Backward sweeps concentrate at the goal; forward/backward modes "
             "persist in streaks (Rat6 20260716)", fontsize=14)
out = "/Users/sachuriga/Desktop/Rat6_20260716_FB_visualization.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}; {len(obs)} observed streaks, longest {obs.max()}")
