"""Decoded positions to sweeps, and sweeps to the left-right alternation statistic.

    extract_sweeps -> alternation

Within each theta cycle: find the bin of peak population firing, grow outwards from it for
as long as the decoded trajectory stays smooth, then take the stretch from the point
nearest the anchor to the point furthest from it, and measure its length, direction and
straightness (`chunkThetaPosSweeps.m`).

The decoder is NOT constrained to the maze. Sweeps reaching never-visited, physically
inaccessible locations is the paper's central claim, so snapping decoded positions onto the
corridors would delete the phenomenon. The maze helpers here are for measuring how far a
sweep left the travelled path, not for stopping it doing so.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .config import Config, SEGMENT_CM
from .data import Session


# Visited and never-visited space
#
# The decoder is NOT constrained to the maze. Sweeps reaching never-visited, physically
# inaccessible locations is the paper's central claim, so snapping decoded positions onto
# the corridors would delete the phenomenon. Bins are a plain rectangular grid; places
# the animal never went carry almost no firing and so attract almost no decoding weight.
#
# What the geometry is good for is telling those places apart afterwards, to ask how far
# =============================================================================
# a sweep left the travelled path.
# =============================================================================
def maze_corridors(node_xy_px, segment_px) -> np.ndarray:
    """Corridors between neighbouring nodes, as (n, 2, 2). For drawing the maze only.

    Incomplete by construction: `node_list_new.csv` holds the hexagon vertices but not the
    long corridors joining one cluster to the next, so those are missing here. Never use
    it to decide where the animal can be -- use `visited_mask`, which comes from tracking.
    """
    distance = cdist(node_xy_px, node_xy_px)
    i, j = np.where(np.triu((distance > 0) & (distance < 1.3 * segment_px)))
    return np.stack([node_xy_px[i], node_xy_px[j]], axis=1)


def visited_mask(track_x_px, track_y_px, config: Config, dilate: int = 2, close: int = 1):
    """Which parts of the arena the animal actually reached.

    Marks the bins it occupied, then dilates and closes them, so that the ragged edge of a
    trajectory does not read as a hole. Everything outside is never-visited space -- the
    same definition the paper uses (`imdilate` then `imclose`, both radius 1).

    Returns:
        (mask, x_edges_px, y_edges_px), the mask being True where the animal has been.
    """
    bin_size_px = config.px(config.spatial_bin_cm)
    x_edges_px = np.arange(np.nanmin(track_x_px), np.nanmax(track_x_px) + bin_size_px, bin_size_px)
    y_edges_px = np.arange(np.nanmin(track_y_px), np.nanmax(track_y_px) + bin_size_px, bin_size_px)

    counts = np.histogram2d(track_x_px, track_y_px, bins=[x_edges_px, y_edges_px])[0]
    mask = counts > 0

    if dilate:
        mask = binary_dilation(mask, iterations=dilate)
    if close:
        mask = binary_dilation(mask, iterations=close)
        mask = ~binary_dilation(~mask, iterations=close)     # erosion completes the closing

    return mask, x_edges_px, y_edges_px


def distance_outside_visited_cm(points_px, track_x_px, track_y_px, config: Config,
                                subsample: int = 5) -> np.ndarray:
    """How far each point lies beyond anywhere the animal has been. Zero if inside.

    Use this to measure how far a sweep reaches into never-visited space, not to stop it
    doing so.
    """
    track_xy = np.c_[track_x_px, track_y_px]
    track_xy = track_xy[np.isfinite(track_xy).all(1)][::subsample]

    distance_px = np.full(len(points_px), np.nan)
    finite = np.where(np.isfinite(points_px).all(1))[0]
    if len(finite):
        distance_px[finite] = cKDTree(track_xy).query(points_px[finite])[0]

    return distance_px / config.px_per_cm


# =============================================================================
# Sweep extraction (chunkThetaPosSweeps.m)
# =============================================================================
def _mark_unreliable_bins(session, config, peak_score, lowpass_error_px, shuffle_99):
    """Which decoded time bins to discard. See `Config` for the thresholds."""
    n_active_cells = (session.spike_counts > 0).sum(1)

    too_few_cells = n_active_cells < config.min_active_cells
    decoder_lost_the_animal = lowpass_error_px > config.px(config.max_lowpass_error_cm)

    # A posterior probability is never negative, so the hippocampal branch's floor of 0
    # would be a gate that never fires -- i.e. silently no gate at all. Bayesian decoding
    # therefore always measures its peak against the shuffled null instead.
    if config.decoder == "bayes" or config.min_peak_correlation is None:
        poor_match = peak_score < shuffle_99
    else:
        poor_match = peak_score < config.min_peak_correlation

    return too_few_cells | decoder_lost_the_animal | poor_match


def _smoothness_breaks(decoded, config):
    """Per bin: was the step to the next too long, the step from the previous too long,
    or did the direction turn too sharply. A sweep may not cross any of these."""
    step = np.diff(decoded, axis=0)

    step_to_next_px = np.r_[np.hypot(step[:, 0], step[:, 1]), np.nan]
    step_from_prev_px = np.r_[np.nan, step_to_next_px[:-1]]

    travel_direction = np.arctan2(step[:, 1], step[:, 0])
    turn_angle = np.r_[0.0, np.abs((np.diff(travel_direction) + np.pi) % (2 * np.pi) - np.pi), 0.0]
    is_sharp_turn = turn_angle > config.turn_max_rad

    return step_to_next_px, step_from_prev_px, is_sharp_turn


def _tang_sweep(sweeps, cycle, path, is_finite, cycle_bins, cycle_start,
                lowpass_smoothed, session, config,
                step_to_next_px, is_sharp_turn, jump_max_px) -> None:
    """One cycle's sweep by the Tang et al. 2026 definition.

    The candidate is the LONGEST smooth stretch of consecutive valid bins, truncated
    to the sub-segment with the greatest net start-to-end displacement. The sweep
    vector runs from the lowpass decoded position at the CYCLE START to the
    stretch's most distal point; direction and length are its angle and magnitude.
    """
    n = len(cycle_bins)

    # --- maximal smooth runs of consecutive valid bins ------------------------
    runs = []
    k = 0
    while k < n:
        if not is_finite[k]:
            k += 1
            continue
        j = k
        while (j + 1 < n and is_finite[j + 1]
               and step_to_next_px[cycle_bins[j]] <= jump_max_px
               and not is_sharp_turn[cycle_bins[j + 1]]):
            j += 1
        runs.append((k, j))
        k = j + 1
    if not runs:
        return

    run_start, run_stop = max(runs, key=lambda r: r[1] - r[0])

    # --- truncate to the sub-segment with the largest net displacement --------
    best_i, best_j, best_net = run_start, run_stop, -1.0
    for i in range(run_start, run_stop + 1):
        for j in range(i + 1, run_stop + 1):
            net = np.hypot(*(path[j] - path[i]))
            if net > best_net:
                best_i, best_j, best_net = i, j, net
    if best_net <= 0:
        return

    # --- the sweep vector: cycle-start anchor -> most distal point ------------
    anchor0 = lowpass_smoothed[cycle_start]
    if not np.isfinite(anchor0).all():
        return
    segment = path[best_i:best_j + 1]
    from_anchor = np.hypot(segment[:, 0] - anchor0[0], segment[:, 1] - anchor0[1])
    distal = int(np.argmax(from_anchor))
    near = int(np.argmin(from_anchor))
    sweep_vector_px = segment[distal] - anchor0
    if np.hypot(*sweep_vector_px) < 1e-9:
        return

    sweeps["n_valid_samples"][cycle] = best_j - best_i + 1
    sweeps["length_px"][cycle] = np.hypot(*sweep_vector_px)
    sweeps["direction"][cycle] = np.arctan2(sweep_vector_px[1], sweep_vector_px[0])
    sweeps["path_xy_px"][cycle] = segment
    sweeps["path_frame_px"][cycle] = segment - anchor0
    sweeps["start_bin"][cycle] = cycle_start + best_i
    sweeps["stop_bin"][cycle] = cycle_start + best_j
    sweeps["true_xy_px"][cycle] = (session.track_x_px[cycle_start + best_i],
                                   session.track_y_px[cycle_start + best_i])
    sweeps["origin_error_px"][cycle] = np.hypot(
        *(segment[near] - sweeps["true_xy_px"][cycle]))

    # r^2 against the sweep vector's axis, the distal point excluded because it
    # defines the axis (same convention as the vollan branch).
    body = np.delete(segment, distal, axis=0) - anchor0
    if len(body) < 2:
        return
    axis = sweep_vector_px / np.hypot(*sweep_vector_px)
    total_variance = body[:, 0].var() + body[:, 1].var()
    if total_variance > 0:
        sweeps["straightness"][cycle] = (body @ axis).var() / total_variance


def extract_sweeps(session, config, decoded_xy_px, peak_correlation,
                   lowpass_xy_px, cycle_onsets, shuffle_99) -> dict:
    """Pull one candidate sweep out of each theta cycle.

    Within a cycle: find the bin of peak population firing, grow outwards from it for as
    long as the decoded trajectory stays smooth, then take the stretch from the point
    nearest the anchor to the point furthest from it, and measure its length, direction
    and straightness.

    Returns:
        dict of arrays with one entry per theta cycle. `is_sweep` masks the cycles that
        passed; `path_xy_px` and `path_frame_px` hold the trajectories.
    """
    n_bins = session.n_bins

    # The peak of population firing marks the middle of the sweep.
    population_rate = gaussian_filter1d(session.spike_counts.sum(1).astype(float), 1.0)

    lowpass_error_px = np.hypot(lowpass_xy_px[:, 0] - session.track_x_px,
                                lowpass_xy_px[:, 1] - session.track_y_px)

    # --- blank out the decoded positions we do not trust -----------------------
    decoded = gaussian_filter1d(decoded_xy_px, config.decoded_smooth_bins, axis=0)
    lowpass_smoothed = gaussian_filter1d(lowpass_xy_px, config.lowpass_smooth_bins, axis=0)

    decoded[_mark_unreliable_bins(session, config, peak_correlation,
                                  lowpass_error_px, shuffle_99)] = np.nan

    # --- work relative to the anchor -------------------------------------------
    sweep_frame = decoded - lowpass_smoothed

    step_to_next_px, step_from_prev_px, is_sharp_turn = _smoothness_breaks(decoded, config)
    jump_max_px = config.px(config.jump_max_cm)

    # --- somewhere to put the answers ------------------------------------------
    n_cycles = len(cycle_onsets)
    cycle_centers = np.clip((cycle_onsets + np.r_[cycle_onsets[1:], n_bins]) // 2, 0, n_bins - 1)

    sweeps = dict(
        n_valid_samples=np.zeros(n_cycles, int),
        straightness=np.full(n_cycles, np.nan),
        length_px=np.full(n_cycles, np.nan),
        direction=np.full(n_cycles, np.nan),
        origin_error_px=np.full(n_cycles, np.nan),   # near end of the sweep, to the animal
        speed_px_s=session.speed_px_s[cycle_centers],
        head_direction=session.head_direction[cycle_centers],
        true_xy_px=np.full((n_cycles, 2), np.nan),
        path_xy_px=[None] * n_cycles,           # trajectory in maze coordinates
        path_frame_px=[None] * n_cycles,        # trajectory relative to the anchor
        start_bin=np.full(n_cycles, -1, int),   # chunkThetaPosSweeps.m: s.iStart
        stop_bin=np.full(n_cycles, -1, int),    # chunkThetaPosSweeps.m: s.iStop
        cycle_onsets=cycle_onsets,
    )

    for cycle in range(n_cycles):
        cycle_start = cycle_onsets[cycle]
        cycle_stop = cycle_onsets[cycle + 1] if cycle + 1 < n_cycles else n_bins
        cycle_bins = np.arange(cycle_start, min(cycle_stop, n_bins))
        if len(cycle_bins) < 3:
            continue

        path = decoded[cycle_bins]
        path_frame = sweep_frame[cycle_bins]
        is_finite = np.isfinite(path[:, 0])
        if not is_finite.any():
            continue

        if config.sweep_convention == "tang":
            _tang_sweep(sweeps, cycle, path, is_finite, cycle_bins, cycle_start,
                        lowpass_smoothed, session, config,
                        step_to_next_px, is_sharp_turn, jump_max_px)
            continue

        # --- the smooth stretch containing the peak of population firing --------
        # Note: the stretch containing that peak, not the longest one in the cycle.
        peak_activity = int(np.nanargmax(population_rate[cycle_bins]))

        breaks_after = np.where((step_to_next_px[cycle_bins] > jump_max_px)
                                | is_sharp_turn[cycle_bins] | ~is_finite)[0]
        breaks_before = np.where((step_from_prev_px[cycle_bins] > jump_max_px)
                                 | is_sharp_turn[cycle_bins] | ~is_finite)[0]

        run_start = breaks_before[breaks_before < peak_activity].max() \
            if np.any(breaks_before < peak_activity) else 0
        run_stop = breaks_after[breaks_after >= peak_activity].min() \
            if np.any(breaks_after >= peak_activity) else len(cycle_bins) - 1

        # --- from nearest the anchor to furthest from it ------------------------
        masked_frame = path_frame.copy()
        masked_frame[:run_start] = np.nan
        masked_frame[run_stop + 1:] = np.nan

        distance_from_anchor_px = np.hypot(masked_frame[:, 0], masked_frame[:, 1])
        if not np.isfinite(distance_from_anchor_px).any():
            continue

        distal_index = int(np.nanargmax(distance_from_anchor_px))
        proximal_index = int(np.nanargmin(distance_from_anchor_px))
        if proximal_index > distal_index:
            proximal_index = 0          # heading back inwards; take the whole stretch

        sweeps["n_valid_samples"][cycle] = distal_index - proximal_index + 1

        # --- length: greatest separation between any two of its points ----------
        valid_points = path[run_start:run_stop + 1]
        valid_points = valid_points[np.isfinite(valid_points[:, 0])]
        if len(valid_points) >= 2:
            sweeps["length_px"][cycle] = cdist(valid_points, valid_points).max()

        # --- direction: towards its furthest point from the anchor --------------
        sweep_vector_px = masked_frame[distal_index]
        if not np.isfinite(sweep_vector_px).all() or np.hypot(*sweep_vector_px) < 1e-9:
            continue
        sweeps["direction"][cycle] = np.arctan2(sweep_vector_px[1], sweep_vector_px[0])

        sweep_slice = slice(proximal_index, distal_index + 1)
        sweeps["path_xy_px"][cycle] = path[sweep_slice]
        sweeps["path_frame_px"][cycle] = masked_frame[sweep_slice]
        sweeps["start_bin"][cycle] = cycle_start + proximal_index
        sweeps["stop_bin"][cycle] = cycle_start + distal_index
        sweeps["true_xy_px"][cycle] = (session.track_x_px[cycle_start + proximal_index],
                                       session.track_y_px[cycle_start + proximal_index])

        # How far the near end of the sweep is from the animal itself. A sweep is supposed
        # to set off from where the animal is; one that never comes near it is not a sweep.
        sweeps["origin_error_px"][cycle] = np.hypot(
            *(path[proximal_index] - sweeps["true_xy_px"][cycle]))

        # --- straightness: variance along the sweep's own axis / total variance --
        # Equals the paper's r^2 = 1 - var(perpendicular) / var(total). The furthest
        # point is excluded because it defines the axis.
        body_points = masked_frame[proximal_index:distal_index]
        body_points = body_points[np.isfinite(body_points[:, 0])]
        if len(body_points) < 2:
            continue

        sweep_axis = sweep_vector_px / np.hypot(*sweep_vector_px)
        total_variance = body_points[:, 0].var() + body_points[:, 1].var()
        if total_variance > 0:
            sweeps["straightness"][cycle] = (body_points @ sweep_axis).var() / total_variance

    # --- which cycles actually contain a sweep ---------------------------------
    sweeps["is_running"] = sweeps["speed_px_s"] > config.px(config.speed_sweep_cm_s)
    sweeps["starts_at_animal"] = sweeps["origin_error_px"] <= config.px(config.max_sweep_origin_cm)

    # A sweep should point ahead of the animal; the head-centred angle says how far off the
    # head direction it is. Off by default (see Config.max_sweep_head_angle_deg).
    head_centred = _wrap_angle(sweeps["direction"] - sweeps["head_direction"])
    if config.max_sweep_head_angle_deg is None:
        sweeps["points_forward"] = np.ones(n_cycles, bool)
    else:
        sweeps["points_forward"] = np.abs(head_centred) <= np.radians(config.max_sweep_head_angle_deg)

    sweeps["is_sweep"] = (sweeps["n_valid_samples"] >= config.min_valid_samples) \
        & (sweeps["straightness"] > config.straightness_min) \
        & sweeps["is_running"] \
        & sweeps["starts_at_animal"] \
        & sweeps["points_forward"]

    sweeps["prevalence"] = sweeps["is_sweep"].sum() / max(sweeps["is_running"].sum(), 1)
    return sweeps


# =============================================================================
# Left-right alternation -- the paper's central result
# =============================================================================
def _wrap_angle(angle):
    """Fold an angle into [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def head_centred_direction(sweeps) -> np.ndarray: # consider to be fixed when dlc_coordinates included
    """Each sweep's direction relative to the animal's heading; positive is left.

    Tracking is in image pixels, whose y axis points DOWN, so angles there run
    clockwise and a leftward sweep gives a negative difference. Negating restores a
    right-handed, left-positive angle. Alternation counts sign flips and is therefore
    unaffected by this; only left/right labels and figures are.
    """
    ego = -_wrap_angle(sweeps["direction"] - sweeps["head_direction"])
    ego[~sweeps["is_sweep"]] = np.nan
    return ego


def alternation(sweeps, n_shuffle=1000, seed=0):
    """Fraction of consecutive sweep triplets that go left-right-left, or the reverse.

    Returns:
        (observed, shuffle_mean, shuffle_99.9th_percentile, n_triplets), where
        n_triplets counts runs of three adjacent theta cycles that all contain a sweep.
        If it is zero the statistic is undefined, not zero; use
        `alternation_consecutive_sweeps` instead.
    """
    is_sweep = sweeps["is_sweep"]
    n_triplets = int(np.sum(is_sweep[:-2] & is_sweep[1:-1] & is_sweep[2:]))
    ego = head_centred_direction(sweeps)

    def alternating_fraction(angles):
        # Alternation means the sign of the turn keeps flipping, so consecutive signs
        # differ by 2 (from +1 to -1, or back).
        turn_sign = np.sign(_wrap_angle(np.diff(angles)))
        sign_change = np.diff(turn_sign)
        usable = np.isfinite(sign_change)
        if usable.sum() == 0:
            return np.nan
        return (np.abs(sign_change[usable]) == 2).sum() / usable.sum()

    observed = alternating_fraction(ego)

    # Null: same directions, same cycles, reordered. Any alternation left is a property
    # of the numbers rather than of their sequence.
    rng = np.random.default_rng(seed)
    present = np.where(np.isfinite(ego))[0]
    null = np.empty(n_shuffle)
    for k in range(n_shuffle):
        shuffled = ego.copy()
        shuffled[present] = ego[present][rng.permutation(len(present))]
        null[k] = alternating_fraction(shuffled)

    return observed, float(np.nanmean(null)), float(np.nanpercentile(null, 99.9)), n_triplets


def alternation_consecutive_sweeps(sweeps, n_shuffle=1000, seed=0):
    """Weaker test for when `alternation` finds no adjacent triplets.

    Compares consecutive detected sweeps, ignoring the theta cycles between them. Sweeps
    separated by seconds have no reason to alternate, so treat the result as indicative.
    """
    present = np.where(sweeps["is_sweep"])[0]
    ego = _wrap_angle(sweeps["direction"][present] - sweeps["head_direction"][present])

    def alternating_fraction(angles):
        sign_change = np.diff(np.sign(_wrap_angle(np.diff(angles))))
        return (np.abs(sign_change) == 2).sum() / len(sign_change) if len(sign_change) else np.nan

    rng = np.random.default_rng(seed)
    null = np.array([alternating_fraction(rng.permutation(ego)) for _ in range(n_shuffle)])
    return alternating_fraction(ego), float(null.mean()), float(np.percentile(null, 99.9)), len(ego)
