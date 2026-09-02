"""Per-session near-goal forward/backward numbers — the rows behind the GL timeline.

One session in, one row of numbers out, plus the per-sweep table those numbers were
counted from. Written to be reproducible from scratch: every convention that the
timeline figure depends on is a named constant here, and `checklist()` prints them so
two people can diff conventions before diffing results.

    python scripts/session_fb_near_goal.py --session-dir /path/to/20260716
    python scripts/session_fb_near_goal.py --cache Rat6_20260716        # dump exists
    python scripts/session_fb_near_goal.py --session-dir ... --sweeps-out per_sweep.csv

or from Python:

    from session_fb_near_goal import analyse_session
    row, sweeps = analyse_session("Rat6_20260716", goal_node=421)

WHY A REIMPLEMENTATION FAILS TO MATCH — the six things that are not obvious. Each is
marked [Nx] at the constant or line that implements it.

  [N1] F/B is a 60/120 degree split with the middle DISCARDED, not a 90 degree cut.
       Sweeps deviating 60-120 deg from heading are "lateral" and enter neither the
       numerator nor the denominator. A 90 deg split silently changes both.
  [N2] Direction is measured animal -> most distal point, recomputed in the ANIMAL
       frame. It is NOT sweeps["direction"], which is measured in the anchor frame.
  [N3] "The animal" is true_xy_px, the tracked position at the sweep's own start bin.
       Not the anchor, not the tracked position at cycle centre.
  [N4] Distance to goal is straight-line Euclidean between node coordinates, in cm.
       It is NOT the geodesic along corridors.
  [N5] The pipeline settings are not the repo defaults. Stock settings find ~28 sweeps
       on a session where these find ~1400. See PIPELINE_ARGS.
  [N6] LFP sampling rate differs per session (1000 vs 1500 Hz) and the timestamps file
       lies. Some files are zero-padded, which hides the duration check. See detect_fs.

Goal node: read from the NWB, but Trials_Data.Goal_node is WRONG on goal-switch days
(it broadcasts the first goal to every trial). On those sessions pass --goal explicitly;
the per-session txt log is ground truth (grep goal_node=).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "results", "glm_shift", "cache")

# ---------------------------------------------------------------- conventions
PX_PER_CM = 1.2435          # measured from node spacing; sets every distance below
TAG = "b612_s04_t180"       # cache filename tag; must match the pipeline settings

FORWARD_MAX_DEG = 60.0      # [N1] dev <= this  -> forward
BACKWARD_MIN_DEG = 120.0    # [N1] dev >= this  -> backward; in between is dropped

RINGS = [(0.0, 100.0), (100.0, 1e4)]   # (lo, hi] in cm, distance from the active goal
MIN_N = 8                              # a ring with fewer F+B sweeps reports NaN

# [N5] the settings that produced the cached dumps. The repo defaults are different.
PIPELINE_ARGS = ["--theta-band", "6", "12", "--speed-smooth-s", "0.4",
                 "--turn-max-deg", "180", "--tag", TAG]
PIPELINE_DEFAULTS = dict(decoder="bayes", spike_smooth_bins=3.0, spatial_bin_cm=5.0,
                         max_sweep_origin_cm=20.0, shuffle_pct=95.0,
                         min_valid_samples=3, jump_max_cm=25.0,
                         straightness_min=0.4, sweep_convention="tang")


def checklist() -> str:
    """Every convention, as text. Print this next to a collaborator's before comparing."""
    return "\n".join([
        "conventions",
        f"  px per cm                 {PX_PER_CM}",
        f"  forward                   deviation <= {FORWARD_MAX_DEG:g} deg   [N1]",
        f"  backward                  deviation >= {BACKWARD_MIN_DEG:g} deg  [N1]",
        f"  lateral (60-120 deg)      DISCARDED, not counted either way      [N1]",
        "  sweep direction           animal -> most distal path point       [N2]",
        "  animal position           true_xy_px, tracked at sweep start bin [N3]",
        "  distance to goal          straight-line Euclidean, cm            [N4]",
        f"  rings (lo, hi] cm         {RINGS}",
        f"  min sweeps per ring       {MIN_N}",
        "  pipeline                  " + " ".join(PIPELINE_ARGS) + "        [N5]",
        "                            " + ", ".join(f"{k}={v}" for k, v in
                                                   PIPELINE_DEFAULTS.items()),
    ])


def wrap(angle):
    """Fold radians into [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def node_xy_cm(node_id: int) -> np.ndarray:
    """Maze node coordinates in cm."""
    nodes = pd.read_csv(os.path.join(REPO, "node_list_new.csv"), header=None,
                        names=["id", "x", "y"])
    match = nodes.loc[nodes.id == int(node_id), ["x", "y"]].values
    if not len(match):
        raise ValueError(f"node {node_id} is not in node_list_new.csv")
    return match[0] / PX_PER_CM


# ---------------------------------------------------------------- LFP rate [N6]
def _last_real_sample(path: str, chunk: int = 1 << 20) -> int:
    """Index after the last non-zero sample of channel 0.

    Some LFP files are zero-padded past the end of the real recording, which makes
    n_samples / fs overshoot and defeats the duration check. Scans backwards so a
    750 MB file costs one chunk in the common case.
    """
    data = np.load(path, mmap_mode="r")
    n = data.shape[0]
    end = n
    while end > 0:
        start = max(0, end - chunk)
        block = np.asarray(data[start:end, 0])
        nz = np.nonzero(block)[0]
        if len(nz):
            return start + int(nz[-1]) + 1
        end = start
    return n


def detect_fs(session_dir: str, last_spike_s: float,
              candidates=(1000.0, 1250.0, 1500.0)) -> tuple[float, float, int, int]:
    """Sampling rate whose implied duration matches the spike record. [N6]

    Returns (fs, relative mismatch, n_samples, n_real_samples). Trust the returned
    mismatch: anything above a few percent means the rate is wrong, not the data.
    """
    lfp_file = glob.glob(os.path.join(session_dir, "LFP_Output", "*lfp_data.npy"))[0]
    n_samples = np.load(lfp_file, mmap_mode="r").shape[0]
    n_real = _last_real_sample(lfp_file)
    fs = min(candidates, key=lambda r: abs(n_real / r - last_spike_s))
    mismatch = abs(n_real / fs - last_spike_s) / last_spike_s
    return fs, mismatch, n_samples, n_real


def read_nwb_meta(session_dir: str) -> dict:
    """Unit count, last spike time, goal node and learning indices; writes trials csv."""
    from pynwb import NWBHDF5IO

    nwb_path = glob.glob(os.path.join(session_dir, "Rat*_*.nwb"))[0]
    name = os.path.basename(nwb_path).replace(".nwb", "")
    io = NWBHDF5IO(nwb_path, "r")
    nwb = io.read()

    last_spike = max(np.asarray(nwb.units["spike_times"][i])[-1]
                     for i in range(len(nwb.units)))
    decode = lambda v: np.array([x.decode() if isinstance(x, bytes) else x for x in v])
    quality = decode(nwb.units["quality_label"][:])
    cell_type = decode(nwb.units["cell_type"][:])
    n_units = int(((quality == "good") & (cell_type == "pyramidal")).sum())

    meta = dict(name=name, nwb_path=nwb_path, last_spike_s=float(last_spike),
                n_units=n_units, goal_node=None, day=None, repeat=None, session=None)
    try:
        trials = nwb.processing["Behavior"]["Trials_Data"].to_dataframe()
        meta["goal_node"] = int(trials.Goal_node.astype(int).iloc[0])
        for key, col in [("day", "Day"), ("repeat", "Repeat"), ("session", "Session")]:
            if col in trials:
                meta[key] = int(trials[col].astype(int).iloc[0])
        # Session-relative seconds. Trial_start_time is epoch seconds in some files.
        if "Trial_start_s" in trials.columns:
            out = trials.groupby("Trial_Num").agg(start=("Trial_start_s", "first"),
                                                  end=("Trial_end_s", "first"))
        else:
            t0 = nwb.session_start_time.timestamp()
            out = trials.groupby("Trial_Num").agg(start=("Trial_start_time", "first"),
                                                  end=("Trial_end_time", "first")) - t0
        os.makedirs(CACHE, exist_ok=True)
        out.to_csv(os.path.join(CACHE, f"{name}_trials.csv"))
        meta["n_trials"] = len(out)
    except Exception as error:                       # keep going; goal can be passed in
        print(f"  {name}: trials table problem: {error}")
    io.close()
    return meta


# ---------------------------------------------------------------- the measurement
def sweep_table(name: str, goal_node: int, trial_range=None,
                tag: str = TAG) -> pd.DataFrame:
    """One row per accepted sweep: its F/B label and its distance to the active goal.

    This is the whole measurement. Everything downstream is counting.
    """
    dump = os.path.join(CACHE, f"{name}_decode_{tag}.npz")
    if not os.path.exists(dump):
        raise FileNotFoundError(
            f"{dump} not found — run the pipeline first (see build_cache / --session-dir)")
    z = np.load(dump, allow_pickle=True)

    onsets = z["cycle_onsets"]
    n_bins = len(z["track_x_px"])
    centers = np.clip((onsets + np.r_[onsets[1:], n_bins]) // 2, 0, n_bins - 1)
    heading = z["head_direction"][centers]           # heading at the cycle centre
    is_sweep = z["is_sweep"].astype(bool)
    paths = z["path_xy_px"]
    true_xy = z["true_xy_px"] / PX_PER_CM            # [N3] animal at sweep start, cm
    t_cycle = onsets * 0.01                          # 10 ms bins

    if trial_range is None:
        t0, t1 = -np.inf, np.inf
    else:                                            # switch days: one goal per phase
        trials = pd.read_csv(os.path.join(CACHE, f"{name}_trials.csv"))
        sel = trials[trials.iloc[:, 0].isin(list(trial_range))]
        if not len(sel):
            raise ValueError(f"no trials of {name} in the requested range")
        t0, t1 = float(sel.start.min()), float(sel.end.max())

    goal = node_xy_cm(goal_node)
    rows = []
    for c in np.where(is_sweep)[0]:
        if not (t0 <= t_cycle[c] <= t1):
            continue
        if paths[c] is None or not np.isfinite(true_xy[c, 0]):
            continue

        path_cm = paths[c] / PX_PER_CM
        animal = true_xy[c]
        # [N2] most distal point FROM THE ANIMAL, and the bearing to it
        offsets = path_cm - animal
        distal = path_cm[np.argmax(np.hypot(offsets[:, 0], offsets[:, 1]))]
        step = distal - animal
        if np.hypot(*step) < 1e-6:
            continue

        bearing = np.arctan2(step[1], step[0])
        dev_deg = float(np.degrees(np.abs(wrap(bearing - heading[c]))))

        # [N1] three-way label; lateral sweeps are dropped entirely
        if dev_deg <= FORWARD_MAX_DEG:
            label = "F"
        elif dev_deg >= BACKWARD_MIN_DEG:
            label = "B"
        else:
            continue

        rows.append(dict(
            cycle=int(c), t_s=float(t_cycle[c]), label=label, dev_deg=dev_deg,
            animal_x_cm=float(animal[0]), animal_y_cm=float(animal[1]),
            distal_x_cm=float(distal[0]), distal_y_cm=float(distal[1]),
            sweep_len_cm=float(np.hypot(*step)),
            goal_dist_cm=float(np.hypot(*(goal - animal))),   # [N4] Euclidean
        ))
    return pd.DataFrame(rows, columns=["cycle", "t_s", "label", "dev_deg",
                                       "animal_x_cm", "animal_y_cm", "distal_x_cm",
                                       "distal_y_cm", "sweep_len_cm", "goal_dist_cm"])


def summarise(sweeps: pd.DataFrame, label: str, date: str, goal_node: int) -> dict:
    """Collapse the per-sweep table into the one row the timeline plots."""
    row = dict(label=label, date=date, goal=goal_node, n_sweeps=len(sweeps))
    for lo, hi in RINGS:
        ring = sweeps[(sweeps.goal_dist_cm > lo) & (sweeps.goal_dist_cm <= hi)]
        key = f"{lo:g}_{hi:g}"
        row[f"pctB_{key}"] = ((ring.label == "B").mean() * 100
                              if len(ring) >= MIN_N else np.nan)
        row[f"n_{key}"] = len(ring)
    return row


def build_cache(session_dir: str, fs: float, channel: int | None = None) -> None:
    """Run the sweep pipeline for one session, producing the decode dump. [N5]"""
    cmd = [sys.executable, os.path.join(REPO, "scripts", "run_pipeline_extlfp.py"),
           "--session-dir", session_dir, "--lfp-fs", str(fs)] + PIPELINE_ARGS
    if channel is not None:
        cmd += ["--lfp-channel", str(channel)]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def analyse_session(name_or_dir: str, goal_node: int | None = None, trial_range=None,
                    label: str | None = None, rebuild: bool = False,
                    lfp_channel: int | None = None) -> tuple[dict, pd.DataFrame]:
    """Everything for one session: build the dump if needed, then measure.

    Args:
        name_or_dir: a session directory (holding the NWB and LFP_Output/) or a bare
            session name like "Rat6_20260716" when its dump already exists.
        goal_node: overrides the NWB. REQUIRED on goal-switch days, where the NWB
            broadcasts the first goal to every trial.
        trial_range: restrict to these trial numbers, for switch days analysed in two
            phases.
        rebuild: re-run the pipeline even if the dump exists.

    Returns (summary row, per-sweep table).
    """
    if os.path.isdir(name_or_dir):
        meta = read_nwb_meta(name_or_dir)
        name = meta["name"]
        fs, mismatch, n_samples, n_real = detect_fs(name_or_dir, meta["last_spike_s"])
        print(f"{name}: {meta['n_units']} good pyramidal units, "
              f"last spike {meta['last_spike_s']:.1f} s")
        print(f"  LFP fs {fs:.0f} Hz, duration mismatch {mismatch * 100:.1f}% "
              f"({n_real}/{n_samples} samples real)")
        if mismatch > 0.05:
            print("  WARNING mismatch > 5% — the rate is probably wrong; do not trust "
                  "theta from this session until it is resolved  [N6]")
        if n_real < n_samples:
            print(f"  note: file is zero-padded past sample {n_real}  [N6]")
        if goal_node is None:
            goal_node = meta["goal_node"]
            print(f"  goal node {goal_node} (from NWB — verify against the txt log on "
                  f"goal-switch days)")
        dump = os.path.join(CACHE, f"{name}_decode_{TAG}.npz")
        if rebuild or not os.path.exists(dump):
            build_cache(name_or_dir, fs, lfp_channel)
    else:
        name = name_or_dir
        if goal_node is None:
            raise ValueError("pass goal_node when starting from a cached dump")

    date = name.split("_")[-1]
    sweeps = sweep_table(name, goal_node, trial_range)
    row = summarise(sweeps, label or name, date, goal_node)
    return row, sweeps


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--session-dir", help="folder with Rat*_<date>.nwb and LFP_Output/")
    src.add_argument("--cache", help="session name whose decode dump already exists")
    ap.add_argument("--goal", type=int, default=None,
                    help="active goal node; REQUIRED on goal-switch days")
    ap.add_argument("--trials", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    help="restrict to this inclusive trial range (switch-day phases)")
    ap.add_argument("--label", default=None, help="row label, e.g. GL4S4")
    ap.add_argument("--rebuild", action="store_true", help="re-run the pipeline")
    ap.add_argument("--lfp-channel", type=int, default=None)
    ap.add_argument("--sweeps-out", default=None, help="write the per-sweep table here")
    ap.add_argument("--row-out", default=None, help="append the summary row to this csv")
    args = ap.parse_args()

    print(checklist(), "\n")
    trial_range = range(args.trials[0], args.trials[1] + 1) if args.trials else None
    row, sweeps = analyse_session(args.session_dir or args.cache, goal_node=args.goal,
                                  trial_range=trial_range, label=args.label,
                                  rebuild=args.rebuild, lfp_channel=args.lfp_channel)

    counts = sweeps.label.value_counts()
    print(f"\n{row['label']}: {len(sweeps)} F/B sweeps "
          f"(F {counts.get('F', 0)}, B {counts.get('B', 0)})")
    for lo, hi in RINGS:
        key = f"{lo:g}_{hi:g}"
        pct = row[f"pctB_{key}"]
        shown = f"{pct:.1f}%" if np.isfinite(pct) else f"n/a (< {MIN_N} sweeps)"
        print(f"  {lo:g}-{hi:g} cm from goal: n={row[f'n_{key}']:5d}  %backward {shown}")

    if args.sweeps_out:
        sweeps.to_csv(args.sweeps_out, index=False)
        print(f"wrote {args.sweeps_out}")
    if args.row_out:
        frame = pd.DataFrame([row])
        header = not os.path.exists(args.row_out)
        frame.to_csv(args.row_out, mode="a", header=header, index=False)
        print(f"appended to {args.row_out}")


if __name__ == "__main__":
    main()
