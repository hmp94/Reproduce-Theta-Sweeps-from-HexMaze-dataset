"""Decode + sweep extraction under the relaxA gates, dumped for the plot scripts."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset")
from hexmaze_sweeps.config import Config
from hexmaze_sweeps.data import load_session
from hexmaze_sweeps.decoding import (decode, lowpass_trajectory, prepare_log_prior,
                                     prepare_tuning, rate_maps, shuffle_threshold,
                                     theta_cycles)
from hexmaze_sweeps.sweeps import extract_sweeps

SESSION_DIR = "/Volumes/genzel/Rat/HM/Rat_HM_Neuron/task/rat6_491391/20260716"
NODES = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset/node_list_new.csv"
DUMP = ("/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset/"
        "results/glm_shift/cache/Rat6_20260716_decode_bayes_s3_b5_relaxA.npz")

config = Config(decoder="bayes", pv_smooth_bins=3.0, spatial_bin_cm=5.0,
                shuffle_percentile=95.0, min_valid_samples=3, max_sweep_origin_cm=20.0,
                jump_max_cm=25.0, straightness_min=0.4,
                external_lfp_npy=f"{SESSION_DIR}/LFP_Output",
                external_lfp_channel=23)

session, nwb_io = load_session(f"{SESSION_DIR}/Rat6_20260716.nwb", NODES, config)
nwb_io.close()
cycle_onsets = theta_cycles(session, config)
tuning_hz, bx, by, occupancy_s = rate_maps(session, config)
tuning = prepare_tuning(tuning_hz, config)
log_prior = prepare_log_prior(occupancy_s, config)
shuffle_99 = shuffle_threshold(session, config, tuning, log_prior)
decoded_xy_px, peak_score = decode(session, config, tuning, bx, by, log_prior)
lowpass_xy_px = lowpass_trajectory(session, config, tuning, bx, by, cycle_onsets, log_prior)
sweeps = extract_sweeps(session, config, decoded_xy_px, peak_score,
                        lowpass_xy_px, cycle_onsets, shuffle_99)

np.savez_compressed(
    DUMP,
    track_x_px=session.track_x_px, track_y_px=session.track_y_px,
    speed_px_s=session.speed_px_s, head_direction=session.head_direction,
    cycle_onsets=cycle_onsets,
    is_sweep=sweeps["is_sweep"], is_running=sweeps["is_running"],
    length_px=sweeps["length_px"], origin_error_px=sweeps["origin_error_px"],
    true_xy_px=sweeps["true_xy_px"],
    path_xy_px=np.array(sweeps["path_xy_px"], dtype=object),
    start_bin=sweeps["start_bin"], stop_bin=sweeps["stop_bin"])
print(f"{int(sweeps['is_sweep'].sum())} sweeps under relaxA -> {DUMP}")
