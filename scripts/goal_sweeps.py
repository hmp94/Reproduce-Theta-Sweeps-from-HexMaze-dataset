"""Goal-sweep analysis, Tang et al. 2026 style: for every accepted sweep, the
angle between the sweep direction and the animal-to-goal direction. Goal sweeps
are those aligned within 10 degrees. The spatial null uses every other maze node
as a pseudo-goal.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset"
DUMP = f"{BASE}/results/glm_shift/cache/Rat6_20260716_decode_bayes_s3_b5_relaxA.npz"
TRIALS = f"{BASE}/results/glm_shift/cache/Rat6_20260716_trials.csv"
NODES = f"{BASE}/node_list_new.csv"
PX, BIN_S = 1.2435, 0.01
GOAL_NODE = 421
ALIGN_DEG = 10.0

z = np.load(DUMP, allow_pickle=True)
onsets = z["cycle_onsets"]
n_cycles = len(onsets)
n_bins = len(z["track_x_px"])
cycle_centers = np.clip((onsets + np.r_[onsets[1:], n_bins]) // 2, 0, n_bins - 1)
heading = z["head_direction"][cycle_centers]
is_sweep = z["is_sweep"].astype(bool)
paths, true_xy = z["path_xy_px"], z["true_xy_px"] / PX

nodes = pd.read_csv(NODES, header=None, names=["id", "x", "y"])
goal_xy = nodes.loc[nodes.id == GOAL_NODE, ["x", "y"]].values[0] / PX
other_nodes = nodes[nodes.id != GOAL_NODE][["x", "y"]].values / PX
far_pseudo = other_nodes[np.hypot(*(other_nodes - goal_xy).T) > 60.0]

trials = pd.read_csv(TRIALS)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# --- per-sweep geometry ------------------------------------------------------
rows = []
for c in np.where(is_sweep)[0]:
    if paths[c] is None or not np.isfinite(true_xy[c, 0]):
        continue
    seg = paths[c] / PX
    animal = true_xy[c]
    distal = seg[np.argmax(np.hypot(*(seg - animal).T))]
    if np.hypot(*(distal - animal)) < 1e-6:
        continue
    bearing = np.arctan2(*(distal - animal)[::-1])
    to_goal = np.arctan2(*(goal_xy - animal)[::-1])
    t = onsets[c] * BIN_S
    in_trial = bool(((trials.start <= t) & (t <= trials.end)).any())
    rows.append(dict(cycle=c, t_s=t, bearing=bearing,
                     theta_goal=np.degrees(np.abs(wrap(bearing - to_goal))),
                     theta_goal_signed=np.degrees(wrap(bearing - to_goal)),
                     theta_head=np.degrees(np.abs(wrap(bearing - heading[c]))),
                     dist_goal=np.hypot(*(goal_xy - animal)),
                     animal_x=animal[0], animal_y=animal[1], in_trial=in_trial))
df = pd.DataFrame(rows)
print(f"{len(df)} sweeps with geometry")

def pct_aligned(sub):
    return (sub.theta_goal <= ALIGN_DEG).mean() * 100

far = df[df.dist_goal > 50]
print(f"aligned to goal (<= {ALIGN_DEG:.0f} deg): {pct_aligned(df):.1f}%  "
      f"(paper gdn: 16.3 +/- 4.4%)")
print(f"aligned to heading (<= {ALIGN_DEG:.0f} deg): "
      f"{(df.theta_head <= ALIGN_DEG).mean() * 100:.1f}%")
print(f"goal-aligned among sweeps > 50 cm from goal: {pct_aligned(far):.1f}% "
      f"(n = {len(far)})")
print(f"in-trial: {pct_aligned(df[df.in_trial]):.1f}% (n={df.in_trial.sum()})  "
      f"out-of-trial: {pct_aligned(df[~df.in_trial]):.1f}% (n={(~df.in_trial).sum()})")

# --- spatial null: every distant node as a pseudo-goal -----------------------
null_pct = []
for pseudo in far_pseudo:
    to_p = np.arctan2(pseudo[1] - df.animal_y, pseudo[0] - df.animal_x)
    theta_p = np.degrees(np.abs(wrap(df.bearing - to_p)))
    keep = np.hypot(pseudo[0] - df.animal_x, pseudo[1] - df.animal_y) > 20
    null_pct.append((theta_p[keep] <= ALIGN_DEG).mean() * 100)
null_pct = np.asarray(null_pct)
real = pct_aligned(df[df.dist_goal > 20])
pctile = (null_pct < real).mean() * 100
print(f"real goal {real:.1f}% vs pseudo-goal null median {np.median(null_pct):.1f}% "
      f"(95th {np.percentile(null_pct, 95):.1f}%), percentile {pctile:.0f}")

# --- figure ------------------------------------------------------------------
fig = plt.figure(figsize=(19, 4.6))

axp = fig.add_subplot(141, projection="polar")
bins = np.linspace(-180, 180, 25)
counts, _ = np.histogram(df.theta_goal_signed, bins)
axp.bar(np.radians((bins[:-1] + bins[1:]) / 2 + 90), counts,
        width=np.radians(np.diff(bins)), color="#a4243b", alpha=0.8, edgecolor="w")
axp.set_xticklabels([])
axp.set_title(f"sweep-to-GOAL angle (up = to goal)\naligned <=10deg: {pct_aligned(df):.1f}%",
              fontsize=10)

axh = fig.add_subplot(142, projection="polar")
counts, _ = np.histogram(np.degrees(wrap(df.bearing - heading[df.cycle])), bins)
axh.bar(np.radians((bins[:-1] + bins[1:]) / 2 + 90), counts,
        width=np.radians(np.diff(bins)), color="#0072b2", alpha=0.8, edgecolor="w")
axh.set_xticklabels([])
axh.set_title("sweep-to-HEADING angle (up = forward)", fontsize=10)

axn = fig.add_subplot(143)
axn.hist(null_pct, bins=20, color="0.8")
axn.axvline(real, color="#a4243b", lw=2.5, label=f"real goal {real:.1f}%")
axn.axvline(np.percentile(null_pct, 95), color="0.4", ls="--", label="null 95th")
axn.set(xlabel=f"% sweeps aligned <= {ALIGN_DEG:.0f} deg", ylabel="pseudo-goal nodes",
        title=f"vs {len(far_pseudo)} pseudo-goals (percentile {pctile:.0f})")
axn.legend(frameon=False, fontsize=8)

axd = fig.add_subplot(144)
edges = np.array([20, 60, 100, 140, 180, 260, 400])
frac, ns = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    sub = df[(df.dist_goal > lo) & (df.dist_goal <= hi)]
    frac.append(pct_aligned(sub) if len(sub) > 20 else np.nan)
    ns.append(len(sub))
centers = (edges[:-1] + edges[1:]) / 2
axd.plot(centers, frac, "o-", color="#a4243b")
axd.axhline(np.median(null_pct), color="0.6", ls="--", label="pseudo-goal median")
for x, f, n in zip(centers, frac, ns):
    if np.isfinite(f):
        axd.annotate(f"n={n}", (x, f), textcoords="offset points", xytext=(0, 6),
                     fontsize=7, ha="center")
axd.set(xlabel="distance to goal (cm)", ylabel=f"% aligned <= {ALIGN_DEG:.0f} deg",
        title="goal alignment vs distance")
axd.legend(frameon=False, fontsize=8)

fig.suptitle(f"Goal-sweep analysis (Tang et al. convention) -- goal node {GOAL_NODE}, "
             f"{len(df)} sweeps", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = "/Users/sachuriga/Desktop/Rat6_20260716_goal_sweeps.png"
fig.savefig(out, dpi=150)
df.to_csv(f"{BASE}/results/glm_shift/Rat6_20260716_goal_sweeps.csv", index=False)
print(f"wrote {out} and the per-sweep csv")
