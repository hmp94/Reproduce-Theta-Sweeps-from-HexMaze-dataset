"""Across sessions: when does the near-goal backward flip appear?

For every session dir given: preflight (auto-detect LFP rate, read goal node and
learning day, export trials), run the sweep pipeline if its dump is missing, then
compute — with no figures — the backward fraction by distance-to-goal, the F/B
persistence test, and the basic sweep stats. One row per session into
results/glm_shift/batch_near_goal.csv.

    python scripts/batch_near_goal.py <session_dir> [<session_dir> ...]
"""
import glob
import os
import subprocess
import sys

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
CACHE = os.path.join(REPO, "results", "glm_shift", "cache")
TAG = "b612_s04_t180"
PX = 1.2435
PIPE_ARGS = ["--theta-band", "6", "12", "--speed-smooth-s", "0.4",
             "--turn-max-deg", "180", "--tag", TAG]

NODES = pd.read_csv(os.path.join(REPO, "node_list_new.csv"), header=None,
                    names=["id", "x", "y"])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def preflight(session_dir):
    """LFP rate, goal node, learning day, unit count; writes the trials csv."""
    from pynwb import NWBHDF5IO

    nwb_path = glob.glob(os.path.join(session_dir, "Rat*_*.nwb"))[0]
    name = os.path.basename(nwb_path).replace(".nwb", "")

    io = NWBHDF5IO(nwb_path, "r")
    nwb = io.read()
    last_spike = max(np.asarray(nwb.units["spike_times"][i])[-1]
                     for i in range(len(nwb.units)))
    quality = np.array([q.decode() if isinstance(q, bytes) else q
                        for q in nwb.units["quality_label"][:]])
    cell_type = np.array([c.decode() if isinstance(c, bytes) else c
                          for c in nwb.units["cell_type"][:]])
    n_units = int(((quality == "good") & (cell_type == "pyramidal")).sum())

    goal_node, day, repeat, sess = None, None, None, None
    trials_csv = os.path.join(CACHE, f"{name}_trials.csv")
    try:
        df = nwb.processing["Behavior"]["Trials_Data"].to_dataframe()
        goal_node = int(df.Goal_node.astype(int).iloc[0])
        day = int(df.Day.astype(int).iloc[0]) if "Day" in df else None
        repeat = int(df.Repeat.astype(int).iloc[0]) if "Repeat" in df else None
        sess = int(df.Session.astype(int).iloc[0]) if "Session" in df else None
        if "Trial_start_s" in df.columns:
            out = df.groupby("Trial_Num").agg(start=("Trial_start_s", "first"),
                                              end=("Trial_end_s", "first"))
        else:
            t0 = nwb.session_start_time.timestamp()
            out = df.groupby("Trial_Num").agg(start=("Trial_start_time", "first"),
                                              end=("Trial_end_time", "first")) - t0
        out.to_csv(trials_csv)
    except Exception as error:
        print(f"  {name}: trials table problem: {error}")
    io.close()

    # The timestamps files lie; the sampling rate that makes the LFP's duration
    # match the spike record is the true one.
    lfp_file = glob.glob(os.path.join(session_dir, "LFP_Output", "*lfp_data.npy"))[0]
    n_samples = np.load(lfp_file, mmap_mode="r").shape[0]
    fs = min((1000.0, 1250.0, 1500.0),
             key=lambda r: abs(n_samples / r - last_spike))
    mismatch = abs(n_samples / fs - last_spike) / last_spike
    if mismatch > 0.05:
        print(f"  {name}: WARNING lfp duration mismatch {mismatch * 100:.0f}% at fs={fs}")
    return name, fs, goal_node, day, repeat, sess, n_units


def session_stats(name, goal_node):
    """Backward-by-distance and persistence, straight from the decode dump."""
    z = np.load(os.path.join(CACHE, f"{name}_decode_{TAG}.npz"), allow_pickle=True)
    onsets = z["cycle_onsets"]
    n_bins = len(z["track_x_px"])
    centers = np.clip((onsets + np.r_[onsets[1:], n_bins]) // 2, 0, n_bins - 1)
    heading = z["head_direction"][centers]
    is_sweep = z["is_sweep"].astype(bool)
    paths, true_xy = z["path_xy_px"], z["true_xy_px"] / PX
    goal = NODES.loc[NODES.id == goal_node, ["x", "y"]].values[0] / PX

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
        dev = np.degrees(np.abs(wrap(bearing - heading[c])))
        label = "F" if dev <= 60 else ("B" if dev >= 120 else "L")
        rows.append((c, label, np.hypot(*(goal - animal))))
    sweep_df = pd.DataFrame(rows, columns=["cycle", "label", "dist_goal"])

    def pct_b(lo, hi):
        m = (sweep_df.dist_goal > lo) & (sweep_df.dist_goal <= hi) & (sweep_df.label != "L")
        return ((sweep_df.label[m] == "B").mean() * 100 if m.sum() >= 10 else np.nan,
                int(m.sum()))

    b_near, n_near = pct_b(0, 60)
    b_mid, n_mid = pct_b(60, 120)
    b_far, n_far = pct_b(120, 1e9)

    # persistence: switch rate between consecutive sweep cycles vs shuffled order
    lab = pd.Series(index=sweep_df.cycle.values, data=sweep_df.label.values)
    lab = lab[lab != "L"]
    cyc = np.sort(lab.index.values)
    pairs = [(c, c + 1) for c in cyc if c + 1 in lab.index]
    p_persist, switch, null_mean = np.nan, np.nan, np.nan
    if len(pairs) >= 30:
        switch = float(np.mean([lab[a] != lab[b] for a, b in pairs])) * 100
        episodes, i = [], 0
        while i < len(cyc):
            j = i
            while j + 1 < len(cyc) and cyc[j + 1] == cyc[j] + 1:
                j += 1
            episodes.append(cyc[i:j + 1])
            i = j + 1
        rng = np.random.default_rng(0)
        null = []
        for _ in range(1000):
            perm = {}
            for ep in episodes:
                labels = lab[ep].values.copy()
                rng.shuffle(labels)
                perm.update(dict(zip(ep, labels)))
            null.append(np.mean([perm[a] != perm[b] for a, b in pairs]) * 100)
        null = np.asarray(null)
        null_mean = float(null.mean())
        p_persist = float((np.sum(null <= switch) + 1) / (len(null) + 1))

    length_cm = z["length_px"][is_sweep] / PX
    return dict(n_sweeps=int(is_sweep.sum()),
                n_running=int(z["is_running"].astype(bool).sum()),
                prevalence=round(is_sweep.sum() / max(z["is_running"].astype(bool).sum(), 1), 3),
                length_cm=round(float(np.nanmean(length_cm)), 1),
                n_F=int((sweep_df.label == "F").sum()),
                n_B=int((sweep_df.label == "B").sum()),
                pctB_0_60=None if np.isnan(b_near) else round(b_near),
                n_0_60=n_near,
                pctB_60_120=None if np.isnan(b_mid) else round(b_mid),
                n_60_120=n_mid,
                pctB_far=None if np.isnan(b_far) else round(b_far),
                switch_pct=None if np.isnan(switch) else round(switch, 1),
                null_switch=None if np.isnan(null_mean) else round(null_mean, 1),
                p_persist=None if np.isnan(p_persist) else round(p_persist, 4))


def main(session_dirs):
    out_rows = []
    for index, session_dir in enumerate(session_dirs, 1):
        session_dir = session_dir.rstrip("/")
        print(f"[{index}/{len(session_dirs)}] {os.path.basename(session_dir)}")
        try:
            name, fs, goal_node, day, repeat, sess, n_units = preflight(session_dir)
            print(f"  {name}: fs={fs:.0f} goal={goal_node} day={day} "
                  f"repeat={repeat} units={n_units}")
            if goal_node is None:
                print("  no goal -> skipping")
                continue

            dump = os.path.join(CACHE, f"{name}_decode_{TAG}.npz")
            if not os.path.exists(dump):
                result = subprocess.run(
                    [sys.executable, "-u", os.path.join(REPO, "scripts", "run_pipeline_extlfp.py"),
                     "--session-dir", session_dir, "--lfp-fs", str(fs)] + PIPE_ARGS,
                    capture_output=True, text=True)
                if result.returncode != 0 or not os.path.exists(dump):
                    print(f"  PIPELINE FAILED:\n{result.stdout[-500:]}\n{result.stderr[-500:]}")
                    continue

            stats = session_stats(name, goal_node)
            row = dict(session=name.replace("Rat6_", ""), day=day, repeat=repeat, task_session=sess,
                       goal=goal_node, units=n_units, lfp_fs=int(fs), **stats)
            out_rows.append(row)
            print(f"  sweeps={stats['n_sweeps']} prev={stats['prevalence']} "
                  f"len={stats['length_cm']} | %B near/mid/far = "
                  f"{stats['pctB_0_60']}/{stats['pctB_60_120']}/{stats['pctB_far']} "
                  f"(n={stats['n_0_60']}/{stats['n_60_120']}) | "
                  f"switch {stats['switch_pct']} vs {stats['null_switch']} "
                  f"p={stats['p_persist']}")
            pd.DataFrame(out_rows).to_csv(
                os.path.join(REPO, "results", "glm_shift", os.environ.get("BATCH_OUT", "batch_near_goal.csv")),
                index=False)
        except Exception as error:
            import traceback
            print(f"  FAILED: {type(error).__name__}: {error}")
            traceback.print_exc()

    print(f"\n{len(out_rows)} sessions done -> results/glm_shift/batch_near_goal.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
