"""Forward vs backward sweeps: (1) where do backward sweeps live (speed, distance
to goal, time since reward); (2) do F and B alternate across consecutive theta
cycles (the corridor version of alternating sampling)?
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset"
PX, BIN_S = 1.2435, 0.01

z = np.load(f"{BASE}/results/glm_shift/cache/Rat6_20260716_decode_bayes_s3_b5_relaxA.npz",
            allow_pickle=True)
df = pd.read_csv(f"{BASE}/results/glm_shift/Rat6_20260716_goal_sweeps.csv")
trials = pd.read_csv(f"{BASE}/results/glm_shift/cache/Rat6_20260716_trials.csv")

onsets = z["cycle_onsets"]
n_bins = len(z["track_x_px"])
cc = np.clip((onsets + np.r_[onsets[1:], n_bins]) // 2, 0, n_bins - 1)
heading = z["head_direction"][cc]
speed_cm = (z["speed_px_s"] / PX)[cc]

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

df["sweep_head"] = np.degrees(np.abs(wrap(df.bearing.values - heading[df.cycle.values])))
df["speed"] = speed_cm[df.cycle.values]
df["label"] = np.where(df.sweep_head <= 60, "F",
                       np.where(df.sweep_head >= 120, "B", "L"))

# time since the previous reward (= previous trial's end) and time to goal arrival
ends = trials.end.values
starts = trials.start.values
df["t_since_reward"] = [t - ends[ends < t].max() if (ends < t).any() else np.nan
                        for t in df.t_s]
df["t_to_goal"] = [starts[starts > 0].size and (ends[ends >= t].min() - t)
                   if (ends >= t).any() else np.nan for t in df.t_s]

fb = df[df.label != "L"].copy()
print(f"{len(df)} sweeps: F {(df.label == 'F').sum()}, B {(df.label == 'B').sum()}, "
      f"L {(df.label == 'L').sum()}")
print(f"median speed  F {df[df.label == 'F'].speed.median():.0f}  "
      f"B {df[df.label == 'B'].speed.median():.0f} cm/s")
print(f"median dist-to-goal  F {df[df.label == 'F'].dist_goal.median():.0f}  "
      f"B {df[df.label == 'B'].dist_goal.median():.0f} cm")
print(f"median t-since-reward  F {df[df.label == 'F'].t_since_reward.median():.0f}  "
      f"B {df[df.label == 'B'].t_since_reward.median():.0f} s")

def b_fraction(sub):
    n = (sub.label != "L").sum()
    return ((sub.label == "B").sum() / n * 100) if n >= 20 else np.nan

def binned(frame, col, edges):
    fr, ns, cs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = frame[(frame[col] > lo) & (frame[col] <= hi)]
        fr.append(b_fraction(sub))
        ns.append((sub.label != "L").sum())
        cs.append((lo + hi) / 2)
    return np.array(cs), np.array(fr), ns

# --- (2) F/B alternation across consecutive cycles ---------------------------
lab = pd.Series(index=fb.cycle.values, data=fb.label.values)
cycles = lab.index.values
pairs = [(c, c + 1) for c in cycles if c + 1 in lab.index]
switches = np.array([lab[a] != lab[b] for a, b in pairs])
observed = switches.mean() * 100
print(f"\nconsecutive F/B pairs: {len(pairs)}, observed switch rate {observed:.1f}%")

# null: permute labels WITHIN each episode of consecutive sweep-cycles, which
# keeps the local F/B mix and episode structure but destroys the ordering
episodes, i = [], 0
sorted_cycles = np.sort(cycles)
while i < len(sorted_cycles):
    j = i
    while j + 1 < len(sorted_cycles) and sorted_cycles[j + 1] == sorted_cycles[j] + 1:
        j += 1
    episodes.append(sorted_cycles[i:j + 1])
    i = j + 1

rng = np.random.default_rng(0)
null = []
for _ in range(2000):
    perm = {}
    for ep in episodes:
        labels = lab[ep].values.copy()
        rng.shuffle(labels)
        perm.update(dict(zip(ep, labels)))
    null.append(np.mean([perm[a] != perm[b] for a, b in pairs]) * 100)
null = np.asarray(null)
p_alt = (np.sum(null >= observed) + 1) / (len(null) + 1)     # alternation: high switch
p_per = (np.sum(null <= observed) + 1) / (len(null) + 1)     # persistence: low switch
print(f"within-episode permutation null: {null.mean():.1f}% "
      f"(2.5-97.5: {np.percentile(null, 2.5):.1f}-{np.percentile(null, 97.5):.1f})")
print(f"p(alternation) = {p_alt:.4f}   p(persistence) = {p_per:.4f}")

# triplet patterns F-B-F / B-F-B
trip = [(c, c + 1, c + 2) for c in cycles
        if c + 1 in lab.index and c + 2 in lab.index]
if trip:
    cyc = np.mean([lab[a] != lab[b] and lab[b] != lab[c] for a, b, c in trip]) * 100
    print(f"triplets: {len(trip)}, strict F-B-F/B-F-B: {cyc:.1f}% "
          f"(chance if independent ~25%)")

# --- figure ------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))

cs, fr, ns = binned(df, "speed", np.array([15, 25, 35, 45, 60, 80, 110]))
axes[0].plot(cs, fr, "o-", color="C3")
for x, f, n in zip(cs, fr, ns):
    if np.isfinite(f):
        axes[0].annotate(f"n={n}", (x, f), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=7)
axes[0].set(xlabel="running speed (cm/s)", ylabel="% backward (of F+B)",
            title="backward fraction vs speed")

cs, fr, ns = binned(df, "dist_goal", np.array([0, 60, 120, 180, 260, 400]))
axes[1].plot(cs, fr, "o-", color="C3")
for x, f, n in zip(cs, fr, ns):
    if np.isfinite(f):
        axes[1].annotate(f"n={n}", (x, f), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=7)
axes[1].set(xlabel="distance to goal (cm)", ylabel="% backward",
            title="backward fraction vs distance to goal")

cs, fr, ns = binned(df.dropna(subset=["t_since_reward"]), "t_since_reward",
                    np.array([0, 10, 20, 40, 80, 160, 400]))
axes[2].plot(cs, fr, "o-", color="C3")
for x, f, n in zip(cs, fr, ns):
    if np.isfinite(f):
        axes[2].annotate(f"n={n}", (x, f), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=7)
axes[2].set(xlabel="time since last reward (s)", ylabel="% backward",
            title="backward fraction vs time since reward")

axes[3].hist(null, bins=25, color="0.85")
axes[3].axvline(observed, color="C3", lw=2.5, label=f"observed {observed:.1f}%")
axes[3].set(xlabel="switch rate between consecutive cycles (%)", ylabel="permutations",
            title=f"F/B alternation test  p(alt)={p_alt:.3f}  p(persist)={p_per:.3f}")
axes[3].legend(frameon=False, fontsize=8)

fig.suptitle(f"Forward vs backward sweeps -- F {(df.label == 'F').sum()}, "
             f"B {(df.label == 'B').sum()} ({(df.label == 'B').sum() / max((df.label != 'L').sum(), 1) * 100:.0f}% of F+B)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = "/Users/sachuriga/Desktop/Rat6_20260716_forward_backward.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
