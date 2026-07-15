"""Spikes to a decoded position, once per 10 ms bin.

    theta_cycles -> rate_maps -> decode -> lowpass_trajectory

Two decoders, because the paper's Methods text and the paper's released code do not agree
about which one it used. `Config.decoder` picks:

* "pv"    -- the authors' population-vector correlation (`decodePv.m`), which is what their
             code actually runs: for each time bin, correlate the population's activity
             ACROSS CELLS against its average activity at each maze position;
* "bayes" -- Poisson rate-map reconstruction, which is what their Methods text describes.

They differ only in how the (n_time_bins, n_positions) score matrix is produced.
`prepare_tuning` feeds each the form it needs -- and they need different forms, so do not
cross the wires. Everything downstream just wants the score to be higher where the position
fits better, so it is shared: a thresholded centre of mass (`processDec`) rather than the
single best position, which would quantise onto the grid.

`lowpass_trajectory` is the anchor -- a slowly-moving decoded position, taken from the
first 40 ms of each theta cycle before the sweep departs. Sweeps are measured against it,
not against the tracked position, because the encoded and actual positions can drift apart.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert

from .config import Config
from .data import Session


# =============================================================================
# Theta cycles
# =============================================================================
def _theta_phase_from_lfp(session: Session, config: Config) -> np.ndarray:
    """Theta phase per time bin, band-passed from the field potential."""
    nyquist_hz = session.lfp_rate_hz / 2
    filter_b, filter_a = butter(2, [config.theta_band_hz[0] / nyquist_hz,
                                    config.theta_band_hz[1] / nyquist_hz], "band")
    lfp_phase = np.angle(hilbert(filtfilt(filter_b, filter_a, session.lfp_theta_channel)))
    lfp_t_s = np.arange(len(lfp_phase)) / session.lfp_rate_hz

    # np.interp repeats its last value rather than extrapolating, which would freeze
    # the phase and destroy every cycle past the end of the LFP. Refuse instead.
    coverage = lfp_t_s[-1] / session.bin_centers_s[-1]
    if coverage < 0.95:
        raise ValueError(
            f"LFP covers only {coverage * 100:.0f}% of the session "
            f"({lfp_t_s[-1]:.0f} s of {session.bin_centers_s[-1]:.0f} s). "
            f"Set Config.lfp_rate_hz explicitly, or use theta_source='pca'.")

    phase = np.interp(session.bin_centers_s, lfp_t_s, np.unwrap(lfp_phase))
    return (phase + np.pi) % (2 * np.pi) - np.pi


def _theta_phase_from_population(session: Session, config: Config) -> np.ndarray:
    """Theta phase per time bin, reconstructed from spiking (the paper's method).

    Band-passed spike counts trace a circle in their first two principal components
    once per theta cycle; the angle around that circle is the phase.
    """
    nyquist_hz = config.bin_rate_hz / 2
    filter_b, filter_a = butter(2, [config.theta_band_hz[0] / nyquist_hz,
                                    config.theta_band_hz[1] / nyquist_hz], "band")
    filtered = filtfilt(filter_b, filter_a, session.spike_counts, axis=0)

    centered = filtered - filtered.mean(0)
    svd_u, svd_s, _ = np.linalg.svd(centered, full_matrices=False)
    pc_scores = svd_u[:, :2] * svd_s[:2]

    phase = np.arctan2(pc_scores[:, 1], pc_scores[:, 0])

    # A principal component's sign is arbitrary, so the circle may be traced backwards.
    if np.median(np.diff(np.unwrap(phase))) < 0:
        phase = -phase
    return phase


def theta_cycles(session: Session, config: Config) -> np.ndarray:
    """
    Bin index at which each theta cycle starts.

    Phase is only defined up to an offset, so it is re-zeroed at the phase where the
    population fires least (the paper's convention). Cycles then begin at upward
    zero-crossings of the re-zeroed phase.
    
    """
    if config.theta_source == "lfp" and session.lfp_theta_channel is not None:
        phase = _theta_phase_from_lfp(session, config)
    else:
        phase = _theta_phase_from_population(session, config)

    # --- find the phase at which the population fires least, and call it zero --
    population_spikes = session.spike_counts.sum(1)
    n_phase_bins = 60 

    phase_bin = (((phase + np.pi) / (2 * np.pi)) * n_phase_bins).astype(int) % n_phase_bins # conv to [0;2pi]
    mean_rate_per_phase_bin = np.array([
        population_spikes[phase_bin == k].mean() if np.any(phase_bin == k) else np.inf
        for k in range(n_phase_bins)])

    quietest_bin = np.argmin(mean_rate_per_phase_bin)
    min_firing_phase = (quietest_bin + 0.5) / n_phase_bins * 2 * np.pi - np.pi 
    phase = (phase - min_firing_phase + np.pi) % (2 * np.pi) - np.pi # revert

    # --- cycles start at upward zero-crossings; drop non-theta durations -------
    onsets = np.where((phase[:-1] < 0) & (phase[1:] >= 0))[0] + 1

    # The last onset is dropped: there is no following onset to measure it against.
    duration_s = np.diff(onsets) * config.bin_s
    is_theta = (duration_s >= config.cycle_min_s) & (duration_s <= config.cycle_max_s)
    return onsets[:-1][is_theta]


# =============================================================================
# Rate maps -- each cell's average firing rate at each position
# =============================================================================
def rate_maps(session: Session, config: Config, include_bins: np.ndarray | None = None):
    """Build the tuning curves the decoder compares against.

    Rates are returned in spikes per second, NOT yet in the form either decoder wants --
    `prepare_tuning` does that, because the two want different things (the PV decoder
    mean-normalises each cell; the Bayesian one needs the rates themselves).

    Args:
        include_bins: optional boolean mask over time bins. When given, only those bins
            contribute to the maps -- this is how `cross_validate_smoothing` holds a fold
            out. The spatial grid still spans the whole session, so bin centres stay
            comparable across folds; only which bins are counted changes.

    Returns:
        tuning_hz: (n_positions, n_units), each cell's firing rate at each position.
        bin_center_x_px, bin_center_y_px: (n_positions,).
        occupancy_s: (n_positions,) -- seconds spent at each, for the Bayesian prior.
    """
    bin_size_px = config.px(config.spatial_bin_cm)
    smooth_sigma_bins = config.px(config.rate_smooth_cm) / bin_size_px

    is_running = (session.speed_px_s > config.px(config.speed_spatial_cm_s)) \
        & np.isfinite(session.speed_px_s)
    if include_bins is not None:
        is_running = is_running & include_bins

    # --- diving maze into smaller grid ---------------------------------------------
    x_edges_px = np.arange(np.nanmin(session.track_x_px),
                           np.nanmax(session.track_x_px) + bin_size_px, bin_size_px)
    y_edges_px = np.arange(np.nanmin(session.track_y_px),
                           np.nanmax(session.track_y_px) + bin_size_px, bin_size_px)
    n_x_bins, n_y_bins = len(x_edges_px) - 1, len(y_edges_px) - 1

    # --- how long the animal spent in each square ------------------------------
    occupancy_s = np.histogram2d(session.track_x_px[is_running], session.track_y_px[is_running],
                                 bins=[x_edges_px, y_edges_px])[0] * config.bin_s
    occupancy_smoothed = gaussian_filter(occupancy_s, smooth_sigma_bins, mode="constant")

    # --- firing rate = smoothed spike count / smoothed time spent --------------
    maps = np.zeros((session.n_units, n_x_bins, n_y_bins), np.float32)
    for unit in range(session.n_units):
        spikes_while_running = session.spike_counts[:, unit] * is_running
        spike_map = gaussian_filter(
            np.histogram2d(session.track_x_px, session.track_y_px,
                           bins=[x_edges_px, y_edges_px], weights=spikes_while_running)[0],
            smooth_sigma_bins, mode="constant")
        with np.errstate(divide="ignore", invalid="ignore"):
            maps[unit] = np.where(occupancy_smoothed > 0.01, spike_map / occupancy_smoothed, 0.0)

    # --- which positions the decoder may choose from ---------------------------
    # Bins the animal visited, plus a margin of unvisited space around them so a sweep can
    # leave the travelled path. Their rate comes from the smoothing above, which carries
    # each cell's tuning a little way past the edge of where the animal actually went.
    visited = occupancy_s > config.min_occupancy_s

    if config.unvisited_margin_cm > 0:
        margin_bins = int(round(config.unvisited_margin_cm / config.spatial_bin_cm))
        visited = binary_dilation(visited, iterations=margin_bins)

        # Keep only bins where the rate maps are actually defined. Below this the loop above
        # sets every cell's rate to exactly zero, so the bin can never be decoded -- but it
        # would still count towards the decoder's 99th-percentile threshold and drag it
        # down, smearing the centroid across the maze.
        visited &= occupancy_smoothed > 0.01

    visited = visited.ravel()
    bin_center_x_px = (0.5 * (x_edges_px[:-1] + x_edges_px[1:]))[:, None].repeat(n_y_bins, 1).ravel()[visited]
    bin_center_y_px = (0.5 * (y_edges_px[:-1] + y_edges_px[1:]))[None, :].repeat(n_x_bins, 0).ravel()[visited]

    tuning_hz = maps.reshape(session.n_units, -1)[:, visited].T
    return tuning_hz, bin_center_x_px, bin_center_y_px, occupancy_s.ravel()[visited]


# =============================================================================
# Putting the rate maps into the form the chosen decoder wants
# =============================================================================
def prepare_tuning(tuning_hz: np.ndarray, config: Config) -> np.ndarray:
    """Rate maps -> what the decoder actually correlates or integrates against.

    The two decoders want different things from the same rate maps, and giving either the
    other's form is silently wrong rather than loud:

    * PV correlation divides each cell by its own mean rate (`decodePv.m` line 24), so that
      a high-firing cell cannot dominate a correlation taken ACROSS cells. This is a
      per-cell rescaling, and Pearson r across cells is not invariant to it, so it matters.
    * Bayesian reconstruction needs the rates themselves, in spikes per second, because the
      Poisson likelihood is a statement about how many spikes a rate produces in `bin_s`.
      Mean-normalising first would make `exp(-tau * sum(f))` meaningless.
    """
    if config.decoder == "bayes":
        return np.maximum(tuning_hz, config.bayes_min_rate_hz)
    return tuning_hz / (tuning_hz.mean(0, keepdims=True) + 1e-9)


def prepare_log_prior(occupancy_s: np.ndarray, config: Config) -> np.ndarray | None:
    """log P(x) for the Bayesian decoder. None means a flat prior (and for PV, unused)."""
    if config.decoder != "bayes" or config.bayes_prior != "occupancy":
        return None

    prior = occupancy_s / occupancy_s.sum()
    return np.log(prior + 1e-12)


def cross_validate_smoothing(session: Session, config: Config, candidates_cm=None,
                             n_folds=None, max_test_bins=4000, seed=0, on_progress=None):
    """Pick the rate-map smoothing width that best predicts held-out position.

    The paper's Methods step 1: "select smoothing kernel size that maximises
    cross-validated prediction quality". For each candidate sigma, build the tuning curves
    on the training folds, decode the held-out fold with whichever decoder `config` selects,
    and score by the median decoding error. The width with the lowest mean error wins.

    Folds are BLOCKED (contiguous spans of running bins), not interleaved. At 10 ms adjacent
    bins are almost identical, so a random split would put a bin's near-copy in the training
    set and make decoding look far better than it is. Blocked folds keep train and test
    genuinely separated in time.

    Returns:
        (best_cm, errors_cm) -- the chosen width, and {sigma_cm: mean CV error in cm} for all
        candidates so the whole curve can be inspected, not just the argmin.
    """
    from dataclasses import replace

    candidates_cm = candidates_cm or config.cv_smoothing_candidates_cm
    n_folds = n_folds or config.cv_folds
    rng = np.random.default_rng(seed)

    running = (session.speed_px_s > config.px(config.speed_spatial_cm_s)) \
        & np.isfinite(session.speed_px_s)
    running_idx = np.where(running)[0]                       # time-ordered
    if len(running_idx) < n_folds:
        raise ValueError("too few running bins to cross-validate")

    # Blocked folds: contiguous slices of the running bins, so each test fold is one span.
    fold_slices = np.array_split(np.arange(len(running_idx)), n_folds)

    # The smoothed activity the decoder sees does not depend on sigma, so smooth once.
    activity = gaussian_filter1d(session.spike_counts, config.pv_smooth_bins, axis=0)
    true_xy_px = np.column_stack([session.track_x_px, session.track_y_px])

    errors_cm: dict[float, float] = {}
    for s, sigma_cm in enumerate(candidates_cm, 1):
        cfg = replace(config, rate_smooth_cm=sigma_cm)
        fold_errors = []

        for fold, test_positions in enumerate(fold_slices, 1):
            test_bins = running_idx[test_positions]
            train_mask = np.ones(session.n_bins, bool)
            train_mask[test_bins] = False

            tuning_hz, bx, by, occ = rate_maps(session, cfg, include_bins=train_mask)
            tuning = prepare_tuning(tuning_hz, cfg)
            log_prior = prepare_log_prior(occ, cfg)

            if len(test_bins) > max_test_bins:                # subsample for speed
                test_bins = rng.choice(test_bins, max_test_bins, replace=False)

            scores = decoder_scores(activity[test_bins], tuning, cfg, log_prior)
            decoded, _ = _thresholded_centroid(scores, bx, by, cfg)

            error_px = np.hypot(decoded[:, 0] - true_xy_px[test_bins, 0],
                                decoded[:, 1] - true_xy_px[test_bins, 1])
            fold_errors.append(np.nanmedian(error_px))

            if on_progress:
                on_progress(f"cv sigma {sigma_cm:g}cm  fold {fold}/{n_folds}  "
                            f"({s}/{len(candidates_cm)})")

        errors_cm[sigma_cm] = float(np.mean(fold_errors)) / config.px_per_cm

    best_cm = min(errors_cm, key=errors_cm.get)
    return best_cm, errors_cm


# =============================================================================
# The decoder
#
# For each time bin, correlate the population's activity against its average activity
# at each maze position. The correlation runs across cells, so it asks which position
# =============================================================================
# the current pattern of firing looks like.
# =============================================================================
def _correlate_across_units(activity: np.ndarray, tuning_curves: np.ndarray) -> np.ndarray:
    """Pearson r across cells: (n_time_bins, n_units) x (n_positions, n_units) -> (T, P)."""
    activity_z = (activity - activity.mean(1, keepdims=True)) / (activity.std(1, keepdims=True) + 1e-12)
    tuning_z = (tuning_curves - tuning_curves.mean(1, keepdims=True)) / (tuning_curves.std(1, keepdims=True) + 1e-12)
    return (activity_z @ tuning_z.T) / activity.shape[1]


def _bayes_posterior(activity, tuning_hz, config: Config, log_prior=None) -> np.ndarray:
    """Poisson posterior over position, one row per time bin. (T, n_units) -> (T, P).

    The standard rate-map reconstruction (Zhang et al. 1998; the same formulation
    pynapple's `decode_2d` uses), which is what the paper's Methods text describes even
    though the released code ships the PV-correlation decoder instead:

        log P(x | n)  =  sum_i n_i log f_i(x)  -  tau sum_i f_i(x)  +  log P(x)  +  const

    `const` absorbs `-sum_i log(n_i!)`, which does not depend on position and therefore
    cannot move the decoded position. That is what lets the SMOOTHED spike counts the PV
    decoder uses be passed in here unchanged: they are not integers, so the factorial term
    would be ill-defined, but it drops out of the posterior anyway.
    """
    log_likelihood = (activity @ np.log(tuning_hz).T
                      - config.bin_s * tuning_hz.sum(1)[None, :])

    if log_prior is not None:
        log_likelihood = log_likelihood + log_prior[None, :]

    # Subtract the row max before exponentiating, or 83 cells' worth of log-rates underflow.
    log_likelihood -= log_likelihood.max(1, keepdims=True)
    posterior = np.exp(log_likelihood)
    return posterior / posterior.sum(1, keepdims=True)


def decoder_scores(activity, tuning, config: Config, log_prior=None) -> np.ndarray:
    """(n_time_bins, n_units) -> (n_time_bins, n_positions). Higher means a better match.

    A correlation for `decoder="pv"`, a posterior probability for `decoder="bayes"`.
    Everything downstream -- the centroid, the shuffle, the anchor -- only asks that the
    score be higher where the position fits better, so it does not care which it is.
    """
    if config.decoder == "bayes":
        return _bayes_posterior(activity, tuning, config, log_prior)
    return _correlate_across_units(activity, tuning)


def _bin_distance_matrix_px(bin_center_x_px, bin_center_y_px) -> np.ndarray:
    """Distance between every pair of position bins. Computed once, reused per chunk."""
    return np.hypot(bin_center_x_px[:, None] - bin_center_x_px[None, :],
                    bin_center_y_px[:, None] - bin_center_y_px[None, :]).astype(np.float32)


def _thresholded_centroid(scores, bin_center_x_px, bin_center_y_px, config,
                          bin_distance_px=None):
    """Turn a map of decoder scores into one decoded position per time bin.

    A weighted average of the good positions, rather than the single best one, which
    would quantise onto the grid (processDec, inside runPvPosDecoding.m). A position
    counts as good if it is among the top few percent anywhere, or lies close to the peak.

    Returns:
        (decoded position per bin, peak score per bin).
    """
    if bin_distance_px is None:
        bin_distance_px = _bin_distance_matrix_px(bin_center_x_px, bin_center_y_px)

    peak_bin = scores.argmax(1)
    peak_score = scores[np.arange(len(scores)), peak_bin]

    n_positions = scores.shape[1]
    kth = int(np.clip(round(config.centroid_percentile / 100.0 * (n_positions - 1)),
                      0, n_positions - 1))
    threshold = np.partition(scores, kth, axis=1)[:, kth][:, None]

    weights = scores.copy()
    is_far_from_peak = bin_distance_px[peak_bin] > config.px(config.centroid_radius_cm)
    weights[(scores < threshold) & is_far_from_peak] = 0.0

    # `processDec` stops here, so by default so do we -- which leaves a negative weight on
    # any position that is anti-correlated but close to the peak, dragging the centroid
    # away from it. See `Config.clip_negative_weights`.
    if config.clip_negative_weights:
        np.clip(weights, 0, None, out=weights)

    weight_sum = weights.sum(1)
    weight_sum[weight_sum == 0] = np.nan            # nothing matched
    centroid = np.stack([(weights * bin_center_x_px).sum(1) / weight_sum,
                         (weights * bin_center_y_px).sum(1) / weight_sum], 1)
    return centroid, peak_score


def decode(session, config, tuning, bin_center_x_px, bin_center_y_px, log_prior=None,
           chunk_size=8000, on_progress=None):
    """Decode the encoded position in every time bin.

    `tuning` must already have been through `prepare_tuning`. Chunked only to bound
    memory: the full (n_time_bins, n_positions) score matrix would not fit. This is the
    slowest stage, so it reports its progress.
    """
    activity = gaussian_filter1d(session.spike_counts, config.pv_smooth_bins, axis=0)
    bin_distance_px = _bin_distance_matrix_px(bin_center_x_px, bin_center_y_px)

    decoded_xy_px = np.full((session.n_bins, 2), np.nan)
    peak_score = np.full(session.n_bins, np.nan)

    starts = range(0, session.n_bins, chunk_size)
    report_every = max(1, len(starts) // 10)        # ~10 updates, not one per chunk

    for done, start in enumerate(starts, 1):
        chunk = slice(start, start + chunk_size)
        scores = decoder_scores(activity[chunk], tuning, config, log_prior)
        decoded_xy_px[chunk], peak_score[chunk] = _thresholded_centroid(
            scores, bin_center_x_px, bin_center_y_px, config, bin_distance_px)

        if on_progress and (done % report_every == 0 or done == len(starts)):
            on_progress(f"decoding {100 * done / len(starts):.0f}%")

    return decoded_xy_px, peak_score


def shuffle_threshold(session, config, tuning, log_prior=None, seed=0) -> float:
    """The score the decoder reaches by chance.

    Rotating each cell's spike train in time keeps its rate and rhythmicity but destroys
    its relationship to place and to the other cells. Returns the 99th percentile of the
    resulting scores -- a correlation for `decoder="pv"`, a posterior for `"bayes"`.
    """
    rng = np.random.default_rng(seed)
    shifted = np.stack([np.roll(session.spike_counts[:, unit], rng.integers(session.n_bins))
                        for unit in range(session.n_units)], 1)

    n_samples = min(config.n_shuffle_bins, session.n_bins)
    scores = decoder_scores(
        gaussian_filter1d(shifted[:n_samples], config.pv_smooth_bins, axis=0),
        tuning, config, log_prior)
    return float(np.percentile(scores, config.shuffle_percentile))


def lowpass_trajectory(session, config, tuning, bin_center_x_px, bin_center_y_px,
                       cycle_onsets, log_prior=None) -> np.ndarray:
    """The anchor: a slowly-moving decoded position that sweeps depart from.

    Decoded from the first 40 ms of each theta cycle, before the sweep has travelled,
    then smoothed across cycles and interpolated back onto every time bin. Sweeps are
    measured against this rather than the tracked position, because the encoded and
    actual positions can drift apart.
    """
    # --- one spike-count vector per cycle, from its first few bins -------------
    window = np.clip(cycle_onsets[:, None] + np.arange(config.anchor_n_bins)[None, :],
                     0, session.n_bins - 1)
    cycle_spike_counts = session.spike_counts[window].sum(1)

    # Smooth across cycles, not time bins: neighbouring cycles see similar places.
    cycle_spike_counts = gaussian_filter1d(cycle_spike_counts.astype(float),
                                           config.anchor_smooth_cycles, axis=0)

    cycle_xy_px, _ = _thresholded_centroid(
        decoder_scores(cycle_spike_counts, tuning, config, log_prior),
        bin_center_x_px, bin_center_y_px, config)

    # Fill unmatched cycles before smoothing, or one NaN spreads across its neighbours.
    fill_value = np.nanmean(cycle_xy_px, 0)
    cycle_xy_px = np.where(np.isfinite(cycle_xy_px), cycle_xy_px, fill_value)
    cycle_xy_px = gaussian_filter1d(cycle_xy_px, config.anchor_post_smooth_cycles, axis=0)

    # --- back onto the time-bin grid, timestamped mid-window -------------------
    cycle_t_s = session.bin_centers_s[cycle_onsets] + 0.5 * config.bin_s * config.anchor_n_bins
    return np.stack([np.interp(session.bin_centers_s, cycle_t_s, cycle_xy_px[:, 0]),
                     np.interp(session.bin_centers_s, cycle_t_s, cycle_xy_px[:, 1])], 1)
