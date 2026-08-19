"""The per-cycle sweep pipeline with the external 1500 Hz LFP, for any session:

    python scripts/run_pipeline_extlfp.py --session-dir <dir> [gate options]

Prints the standard stats plus a breakdown of WHY running cycles were rejected.
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexmaze_sweeps.config import Config
from hexmaze_sweeps.data import Result, load_session
from hexmaze_sweeps.decoding import (decode, lowpass_trajectory, prepare_log_prior,
                                     prepare_tuning, rate_maps, shuffle_threshold,
                                     theta_cycles)
from hexmaze_sweeps.plotting import plot_sweeps
from hexmaze_sweeps.sweeps import alternation, extract_sweeps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--session-dir", required=True,
                help="folder holding Rat*_<date>.nwb and LFP_Output/")
ap.add_argument("--lfp-channel", type=int, default=None,
                help="default: auto-pick by theta/delta power ratio")
ap.add_argument("--lfp-fs", type=float, default=1500.0,
                help="external LFP sampling rate; some sessions are 1000 Hz — "
                     "verify with a theta-peak check, the timestamps file lies")
ap.add_argument("--decoder", default="bayes", choices=["pv", "bayes"])
ap.add_argument("--spike-smooth-bins", type=float, default=3.0)
ap.add_argument("--spatial-bin-cm", type=float, default=5.0)
ap.add_argument("--max-sweep-origin-cm", type=float, default=20.0)
ap.add_argument("--shuffle-pct", type=float, default=95.0)
ap.add_argument("--min-valid-samples", type=int, default=3)
ap.add_argument("--jump-max-cm", type=float, default=25.0)
ap.add_argument("--straightness-min", type=float, default=0.4)
ap.add_argument("--turn-max-deg", type=float, default=90.0)
ap.add_argument("--theta-band", type=float, nargs=2, default=[5.0, 10.0],
                metavar=("LO", "HI"))
ap.add_argument("--speed-smooth-s", type=float, default=1.0)
ap.add_argument("--sweep-convention", default="tang", choices=["tang", "vollan"])
ap.add_argument("--tag", default="tang_relaxA_o20")
args = ap.parse_args()

nwb_path = glob.glob(os.path.join(args.session_dir, "Rat*_*.nwb"))[0]
name = os.path.basename(nwb_path).replace(".nwb", "")
out_dir = os.path.join(REPO, "results", "glm_shift")
os.makedirs(out_dir, exist_ok=True)

config = Config(decoder=args.decoder, pv_smooth_bins=args.spike_smooth_bins,
                spatial_bin_cm=args.spatial_bin_cm,
                max_sweep_origin_cm=args.max_sweep_origin_cm,
                shuffle_percentile=args.shuffle_pct,
                min_valid_samples=args.min_valid_samples,
                jump_max_cm=args.jump_max_cm,
                straightness_min=args.straightness_min,
                turn_max_rad=np.radians(args.turn_max_deg),
                theta_band_hz=tuple(args.theta_band),
                speed_smooth_s=args.speed_smooth_s,
                sweep_convention=args.sweep_convention,
                external_lfp_npy=os.path.join(args.session_dir, "LFP_Output"),
                external_lfp_fs=args.lfp_fs,
                external_lfp_channel=args.lfp_channel)
print(f"[{args.tag}] {name}: decoder={args.decoder} smooth={args.spike_smooth_bins} "
      f"bin={args.spatial_bin_cm}cm shuffle={args.shuffle_pct} "
      f"minvalid={args.min_valid_samples} jump={args.jump_max_cm} "
      f"straight={args.straightness_min} origin={args.max_sweep_origin_cm} "
      f"conv={args.sweep_convention}")

session, nwb_io = load_session(nwb_path, os.path.join(REPO, "node_list_new.csv"), config)
nwb_io.close()
print(f"{session.n_units} units, {session.n_bins} bins "
      f"({session.n_bins * config.bin_s / 60:.1f} min), LFP {session.lfp_rate_hz:.0f} Hz")

cycle_onsets = theta_cycles(session, config)
print(f"{len(cycle_onsets)} theta cycles")

tuning_hz, bx, by, occupancy_s = rate_maps(session, config)
tuning = prepare_tuning(tuning_hz, config)
log_prior = prepare_log_prior(occupancy_s, config)
shuffle_99 = shuffle_threshold(session, config, tuning, log_prior)
decoded_xy_px, peak_score = decode(session, config, tuning, bx, by, log_prior)
lowpass_xy_px = lowpass_trajectory(session, config, tuning, bx, by, cycle_onsets, log_prior)
sweeps = extract_sweeps(session, config, decoded_xy_px, peak_score,
                        lowpass_xy_px, cycle_onsets, shuffle_99)
observed, null_mean, null_high, n_triplets = alternation(sweeps)

# --- why do running cycles fail? ---------------------------------------------
run = sweeps["is_running"]
nv = sweeps["n_valid_samples"][run]
st = sweeps["straightness"][run]
enough = nv >= config.min_valid_samples
print(f"\nrejection breakdown over {run.sum()} running cycles:")
print(f"  fewer than {config.min_valid_samples} contiguous valid bins: "
      f"{(~enough).mean() * 100:.0f}%   (median valid bins {np.median(nv):.0f})")
print(f"  enough bins but straightness <= {config.straightness_min}: "
      f"{(enough & ~(st > config.straightness_min)).mean() * 100:.0f}%")
print(f"  accepted: {sweeps['is_sweep'][run].mean() * 100:.1f}%")

is_sweep = sweeps["is_sweep"]
mean_length_cm = (np.nanmean(sweeps["length_px"][is_sweep]) / config.px_per_cm
                  if is_sweep.any() else np.nan)
stats = dict(shuffle_99=shuffle_99, n_cycles=len(cycle_onsets),
             n_units=session.n_units, n_sweeps=int(is_sweep.sum()),
             n_running=int(sweeps["is_running"].sum()),
             prevalence=sweeps["prevalence"], mean_length_cm=mean_length_cm,
             alternation=observed, alternation_null=null_mean,
             alternation_null_high=null_high, n_triplets=n_triplets,
             rate_smooth_cm=config.rate_smooth_cm, cv_errors_cm=None)

print(f"\nsweeps {stats['n_sweeps']} / {stats['n_running']} running cycles "
      f"= prevalence {stats['prevalence']:.3f}   (paper: 0.48)")
if stats["n_sweeps"]:
    print(f"mean sweep length {mean_length_cm:.1f} cm   (paper: 22.5)")
if n_triplets:
    print(f"alternation {observed * 100:.1f}% vs shuffle {null_mean * 100:.1f}% "
          f"(99.9th pct {null_high * 100:.1f}%), {n_triplets} triplets")
else:
    print("alternation undefined: no sweep triplets")

result = Result(session, config, sweeps, stats, decoded_xy_px, lowpass_xy_px, cycle_onsets)
import matplotlib.pyplot as plt
png = os.path.join(out_dir, f"{name}_pipeline_{args.tag}_sweeps.png")
fig = plot_sweeps(result, verbose=False, save_path=png)
plt.close(fig)

# dump for the downstream analyses (goal sweeps, F/B, plotting)
dump = os.path.join(out_dir, "cache", f"{name}_decode_{args.tag}.npz")
os.makedirs(os.path.dirname(dump), exist_ok=True)
np.savez_compressed(
    dump,
    track_x_px=session.track_x_px, track_y_px=session.track_y_px,
    speed_px_s=session.speed_px_s, head_direction=session.head_direction,
    cycle_onsets=cycle_onsets,
    is_sweep=sweeps["is_sweep"], is_running=sweeps["is_running"],
    length_px=sweeps["length_px"], origin_error_px=sweeps["origin_error_px"],
    true_xy_px=sweeps["true_xy_px"],
    path_xy_px=np.array(sweeps["path_xy_px"], dtype=object),
    start_bin=sweeps["start_bin"], stop_bin=sweeps["stop_bin"],
    spike_counts=session.spike_counts)
print(f"wrote {png}\nwrote {dump}")
