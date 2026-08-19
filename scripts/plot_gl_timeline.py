"""Learning timeline of the near-goal backward flip: x = GL[n]S[n] (goal-location
block, session within block; switch-day phases are separate points), y = %backward
among F/B sweeps, three near (<=100 cm) vs far (>100 cm) of the active goal (0-60 / 60-100 /
100-200 cm). Bins with fewer than 8 F/B sweeps are left out.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset"
PX = 1.2435
nodes = pd.read_csv(f"{BASE}/node_list_new.csv", header=None, names=["id", "x", "y"])

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def node_xy(n):
    return nodes.loc[nodes.id == n, ["x", "y"]].values[0] / PX

# (date, trial-range or None, goal, GL label). GL = the task's Repeat index,
# S = its Session index (from Trials_Data). A repeat block STARTS with the switch
# session: its first half runs the old goal (label ...a), its second half is the
# new-goal introduction, NGL (label ...b). Missing S numbers = sessions not on
# disk yet. Trial ranges for the split sessions come from the txt logs.
PLAN = [
    ("20260623", None,          204, "GL1S1"),
    ("20260624", None,          204, "GL1S2"),
    ("20260625", None,          204, "GL1S3"),
    ("20260626", None,          204, "GL1S4"),
    ("20260629", range(1, 11),  204, "GL2S1a"),
    ("20260629", range(11, 25), 109, "GL2S1b"),  # NGL introduction
    ("20260630", None,          109, "GL2S2"),
    ("20260701", None,          109, "GL2S3"),
    ("20260703", None,          109, "GL2S5"),
    ("20260706", range(1, 12),  109, "GL3S1a"),
    ("20260706", range(12, 23), 318, "GL3S1b"),  # NGL introduction
    ("20260707", None,          318, "GL3S2"),
    ("20260708", None,          318, "GL3S3"),
    ("20260709", None,          318, "GL3S4"),
    ("20260710", None,          318, "GL3S5"),
    ("20260716", None,          421, "GL4S4"),
    ("20260717", None,          421, "GL4S5"),
]
RINGS = [(0, 100), (100, 10000)]
MIN_N = 8

rows = []
for date, trial_range, goal_id, label in PLAN:
    name = f"Rat6_{date}"
    try:
        z = np.load(f"{BASE}/results/glm_shift/cache/{name}_decode_b612_s04_t180.npz",
                    allow_pickle=True)
    except FileNotFoundError:
        continue
    onsets = z["cycle_onsets"]
    nb = len(z["track_x_px"])
    cc = np.clip((onsets + np.r_[onsets[1:], nb]) // 2, 0, nb - 1)
    heading = z["head_direction"][cc]
    is_sweep = z["is_sweep"].astype(bool)
    paths, true_xy = z["path_xy_px"], z["true_xy_px"] / PX
    t_cycle = onsets * 0.01

    if trial_range is None:
        t0, t1 = -1.0, 1e12
    else:
        trials = pd.read_csv(f"{BASE}/results/glm_shift/cache/{name}_trials.csv")
        sel = trials[trials.iloc[:, 0].isin(list(trial_range))]
        t0, t1 = sel.start.min(), sel.end.max()

    goal = node_xy(goal_id)
    labs = []
    for c in np.where(is_sweep)[0]:
        if not (t0 <= t_cycle[c] <= t1) or paths[c] is None \
                or not np.isfinite(true_xy[c, 0]):
            continue
        seg = paths[c] / PX
        animal = true_xy[c]
        distal = seg[np.argmax(np.hypot(*(seg - animal).T))]
        if np.hypot(*(distal - animal)) < 1e-6:
            continue
        dev = np.degrees(np.abs(wrap(np.arctan2(*(distal - animal)[::-1]) - heading[c])))
        if dev <= 60:
            lab = "F"
        elif dev >= 120:
            lab = "B"
        else:
            continue
        labs.append((lab, np.hypot(*(goal - animal))))
    df = pd.DataFrame(labs, columns=["lab", "d"])

    row = dict(label=label, date=date, goal=goal_id, n_sweeps=len(df))
    for lo, hi in RINGS:
        m = df[(df.d > lo) & (df.d <= hi)]
        key = f"{lo}_{hi}"
        row[f"pctB_{key}"] = (m.lab == "B").mean() * 100 if len(m) >= MIN_N else np.nan
        row[f"n_{key}"] = len(m)
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(f"{BASE}/results/glm_shift/gl_timeline_rings.csv", index=False)
print(table.to_string(index=False))

# ---------------------------------------------------------------- figure
COLORS = {"0_100": "#a4243b", "100_10000": "#8f9bb3"}
NAMES = {"0_100": "0-100 cm", "100_10000": ">100 cm"}

fig, ax = plt.subplots(figsize=(13.5, 5.6))
x = np.arange(len(table))

for key in ["0_100", "100_10000"]:
    y = table[f"pctB_{key}"].values
    ax.plot(x, y, "o-", color=COLORS[key], lw=1.8, ms=7, label=NAMES[key],
            zorder=3 if key == "0_100" else 2)
    for xi, (yi, ni) in enumerate(zip(y, table[f"n_{key}"])):
        if np.isfinite(yi) and key == "0_100":
            ax.annotate(f"{ni}", (xi, yi), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7, color=COLORS[key])

# goal-block boundaries and shading
block_edges = [i for i in range(1, len(table))
               if table.label.iloc[i][:3] != table.label.iloc[i - 1][:3]]
for e in block_edges:
    ax.axvline(e - 0.5, color="0.75", lw=1, ls="--")
for i, lab in enumerate(table.label):
    if lab.endswith("b"):
        ax.axvspan(i - 0.4, i + 0.4, color="0.92", zorder=0)
        ax.annotate("NGL", (i, 2), ha="center", fontsize=8, color="0.5")

ax.axhline(50, color="0.85", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(table.label, rotation=45, ha="right", fontsize=9)
ax.set(ylabel="% backward sweeps (of F+B)", ylim=(0, 105),
       title="Near-goal retrospective sweeps across learning — "
             "near (<=100 cm) vs far (>100 cm) of the active goal\n"
             "(GL = goal-location block, S = session; a/b = old-goal / NGL phase of a switch session; "
             "numbers = n sweeps in the 0-100 cm zone)")
goal_of_block = {lab[:3]: g for lab, g in zip(table.label, table.goal)}
for block, g in goal_of_block.items():
    xs = [i for i, l in enumerate(table.label) if l.startswith(block)]
    ax.annotate(f"goal {g}", (np.mean(xs), 100), ha="center", fontsize=9,
                color="0.35")
ax.legend(frameon=False, loc="center left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "/Users/sachuriga/Desktop/Rat6_GL_timeline_FB.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
