"""Reading one NWB file, and deciding whether to trust what is in it.

The LFP sampling rate in this dataset lies about itself: 6 of 10 files stamp a 1 kHz LFP
with the 30 kHz acquisition rate, and `np.interp` then silently freezes theta phase rather
than extrapolating. `check_lfp_rates` is a preflight that runs before the analysis and
prints a table.

Head direction comes either from two DLC body parts (`head_direction_source="dlc"`) or, by
default, from the direction of travel. The DLC keypoints are not trustworthy on this
dataset -- they jump around, because DeepLabCut's likelihood column is dropped when the
NWB is written, so nothing filters low-confidence frames; see docs/data-issues.md. Travel
direction is only a good proxy while the animal runs, hence the 15 cm/s gate on anything
using it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from scipy.spatial.distance import cdist

from .config import Config, SEGMENT_CM, status


# =============================================================================
# Data containers
# =============================================================================
@dataclass
class Session:
    """One recording, resampled onto a regular grid of `bin_s` time bins."""

    bin_centers_s: np.ndarray
    n_bins: int
    track_x_px: np.ndarray                  # where the animal actually was
    track_y_px: np.ndarray
    speed_px_s: np.ndarray                  # NaN where tracking is missing
    head_direction: np.ndarray              # direction of travel; see module docstring
    spike_counts: np.ndarray                # (n_bins, n_units)
    unit_ids: np.ndarray
    lfp_theta_channel: np.ndarray | None
    lfp_rate_hz: float
    node_xy_px: np.ndarray                  # maze node coordinates, (n_nodes, 2)
    config: Config = field(repr=False)

    @property
    def n_units(self) -> int:
        return self.spike_counts.shape[1]


@dataclass
class Result:
    """Everything one call to `run` produced, so figures can be drawn afterwards."""

    session: Session
    config: Config
    sweeps: dict                            # per-theta-cycle arrays; see extract_sweeps
    stats: dict                             # one row of the results table
    decoded_xy_px: np.ndarray
    lowpass_xy_px: np.ndarray
    cycle_onsets: np.ndarray


# =============================================================================
# LFP sampling rate
# =============================================================================
PLAUSIBLE_LFP_RATES_HZ = (1000.0, 1250.0, 1500.0, 2000.0, 2500.0)


class LfpRate(NamedTuple):
    """How one session's LFP sampling rate was resolved.

    `method` is one of "timestamps" (the file's own were consistent), "length"
    (n_samples / duration), "spectrum" (only that rate shows a theta peak),
    "fallback" (nothing worked), or "config" (supplied by the caller).
    """

    rate_hz: float
    timestamp_rate_hz: float
    timestamp_span_s: float
    ephys_end_s: float
    n_samples: int
    method: str
    note: str

    @property
    def timestamps_ok(self) -> bool:
        return self.method in ("timestamps", "config")


def _theta_peak_score(lfp, rate_hz: float, seconds: float = 300.0) -> tuple[float, float]:
    
    """
    Returns:
        (score, peak_hz). Score is the 4-12 Hz peak height relative to surrounding
        power, or 0.0 if that peak is not inside 5.5-9.5 Hz.
    """
    n_samples = int(min(seconds * rate_hz, lfp.data.shape[0]))
    channel = lfp.data.shape[1] // 2
    signal = lfp.data[:n_samples, channel].astype(float)

    segment_length = int(min(n_samples, max(1024, 8 * rate_hz)))
    freq, power = welch(signal, fs=rate_hz, nperseg=segment_length)

    band = (freq >= 4) & (freq <= 12)
    broadband = (freq >= 1) & (freq <= 25)
    if band.sum() < 3 or broadband.sum() < 3:
        return 0.0, np.nan

    peak_hz = float(freq[band][np.argmax(power[band])])
    prominence = float(power[band].max() / np.median(power[broadband]))
    return (prominence if 5.5 <= peak_hz <= 9.5 else 0.0), peak_hz


def infer_lfp_rate_hz(lfp, ephys_end_s: float, config: Config) -> LfpRate:
    """Resolve the LFP's sampling rate. Computes only; `print_lfp_check` reports.

    Some files stamp the downsampled LFP with the raw acquisition rate, so
    `1 / diff(timestamps)` is too fast and the timestamps cover only a fraction of the
    session. This fails silently downstream, because `np.interp` repeats its last value
    rather than extrapolating, freezing the theta phase.

    Falls back to `n_samples / ephys_end_s`, and if the LFP outlasts the spike record
    (making its length uninformative) to whichever candidate rate reveals a theta peak.

    Args:
        lfp: NWB TimeSeries with `.data` (n_samples, n_channels) and `.timestamps`.
        ephys_end_s: Duration of the electrical recording, i.e. the last spike time.
            Not the tracking duration, which can start or stop at a different moment.
    """
    n_samples = int(lfp.data.shape[0])
    rate_from_timestamps = 1.0 / float(np.median(np.diff(lfp.timestamps[:5000])))
    timestamp_span_s = float(lfp.timestamps[-1])

    def outcome(rate_hz, method, note=""):
        return LfpRate(float(rate_hz), rate_from_timestamps, timestamp_span_s,
                       float(ephys_end_s), n_samples, method, note)

    if config.lfp_rate_hz is not None:
        return outcome(config.lfp_rate_hz, "config", "supplied by the caller")

    if timestamp_span_s >= 0.9 * ephys_end_s:
        return outcome(rate_from_timestamps, "timestamps")

    rate_from_length = n_samples / ephys_end_s
    for standard_rate in PLAUSIBLE_LFP_RATES_HZ:
        if abs(rate_from_length - standard_rate) / standard_rate < 0.01:
            return outcome(standard_rate, "length",
                           f"n_samples/duration = {rate_from_length:.1f} Hz")

    # The LFP outlasts the spikes, so its length proves nothing. Ask the signal.
    scored = [(*_theta_peak_score(lfp, rate_hz), rate_hz) for rate_hz in PLAUSIBLE_LFP_RATES_HZ]
    best_score, peak_hz, best_rate = max(scored)

    if best_score == 0.0:
        return outcome(rate_from_length, "fallback",
                       "no candidate rate reveals a theta peak; "
                       "consider theta_source='pca' for this session")

    return outcome(best_rate, "spectrum",
                   f"length implies {rate_from_length:.0f} Hz (the LFP outlasts the "
                   f"spikes); theta peak at {peak_hz:.2f} Hz")


def check_lfp_rates(nwb_paths, config: Config | None = None) -> dict[str, LfpRate]:
    """Resolve every session's LFP rate up front, keeping warnings out of the results."""
    config = config or Config()
    records: dict[str, LfpRate] = {}

    for index, path in enumerate(nwb_paths, 1):
        name = os.path.basename(path)
        status(f"  LFP check [{index}/{len(nwb_paths)}] {name} ...")

        nwb_io = NWBHDF5IO(path, "r")
        try:
            nwb = nwb_io.read()
            if "lfp" not in nwb.acquisition:
                continue

            # `.target.data` is the flat concatenation of every unit's spike times.
            last_spike_s = float(np.max(nwb.units["spike_times"].target.data[:]))

            records[name] = infer_lfp_rate_hz(nwb.acquisition["lfp"], last_spike_s, config)
        finally:
            nwb_io.close()

    status()
    return records


def print_lfp_check(records: dict[str, LfpRate]) -> int:
    """Print `check_lfp_rates` as a table. Returns how many sessions were suspect."""

    # --- one row per session --------------------------------------------------
    columns = (f"{'session':<24}{'n_samples':>11}{'stamped fs':>12}"
               f"{'stamps span':>13}{'ephys':>9}{'resolved fs':>13}  {'source':<10}")

    print("LFP sampling-rate check")
    print(columns)
    print("-" * len(columns))

    for name, record in records.items():
        print(f"{name:<24}"
              f"{record.n_samples:>11}"
              f"{record.timestamp_rate_hz:>12.0f}"
              f"{record.timestamp_span_s:>12.1f}s"
              f"{record.ephys_end_s:>8.1f}s"
              f"{record.rate_hz:>13.0f}"
              f"  {record.method:<10}")

    print("-" * len(columns))

    # --- then explain any session whose timestamps were overridden ------------
    suspect = {name: r for name, r in records.items() if not r.timestamps_ok}

    if not suspect:
        print("all sessions: LFP timestamps span the recording")
        return 0

    print(f"{len(suspect)} of {len(records)} sessions have LFP timestamps that do not "
          f"span the recording; the stamped rate was overridden. Notes:")

    for name, record in suspect.items():
        if record.note:
            print(f"  {name}: {record.note}")

    return len(suspect)


# =============================================================================
# Loading one session
# =============================================================================
def _decode_strings(values) -> np.ndarray:
    """NWB text columns come back as bytes on some files and str on others."""
    return np.array([v.decode() if isinstance(v, bytes) else v for v in np.asarray(values)])


def _running_speed_px_s(track_x_px, track_y_px, has_tracking, config: Config) -> np.ndarray:
    """Speed per time bin. NaN where tracking is missing, so speed gates reject it."""
    n_bins = len(track_x_px)

    speed_px_s = np.empty(n_bins)
    speed_px_s[0] = 0.0
    speed_px_s[1:] = np.hypot(np.diff(track_x_px), np.diff(track_y_px)) / config.bin_s

    # Tracking glitches appear as isolated impossible speeds; interpolate over them.
    if config.speed_despike_cm_s is not None:
        is_jump = speed_px_s > config.px(config.speed_despike_cm_s)
        if is_jump.any():
            index = np.arange(n_bins)
            speed_px_s[is_jump] = np.interp(index[is_jump], index[~is_jump], speed_px_s[~is_jump])

    speed_px_s = gaussian_filter1d(speed_px_s, config.speed_smooth_s / config.bin_s)
    speed_px_s[~has_tracking] = np.nan
    return speed_px_s


def _travel_direction(track_x_px, track_y_px, config: Config) -> np.ndarray:
    """Direction of motion per time bin, standing in for head direction."""
    smooth_bins = config.head_dir_smooth_s / config.bin_s
    velocity_x = gaussian_filter1d(np.gradient(track_x_px), smooth_bins)
    velocity_y = gaussian_filter1d(np.gradient(track_y_px), smooth_bins)
    return np.arctan2(velocity_y, velocity_x)


def calc_head_direction(positions: np.ndarray) -> np.ndarray:
    """Head direction from a front and a back body part, in radians.

    Args:
        positions: (n_samples, 5) -- timestamp, front x, front y, back x, back y.

    Returns:
        The angle from the back part to the front part, one per sample.
    """
    if positions.shape[1] < 5:
        raise ValueError("positions must be (n, 5): t, front_x, front_y, back_x, back_y")

    front_x, front_y = positions[:, 1], positions[:, 2]
    back_x, back_y = positions[:, 3], positions[:, 4]

    degrees = np.remainder(
        np.degrees(np.arctan2(back_y - front_y, back_x - front_x)) + 180, 360)
    return np.radians(degrees)


def _dlc_head_direction(nwb, bin_centers_s, config: Config) -> np.ndarray:
    """Head direction per time bin, from two DLC body parts.

    DLC body parts are stored relative to the animal rather than in maze coordinates,
    but head direction is the angle between two of them, so the offset cancels.
    """
    if "DLC_Position" not in nwb.processing["Behavior"].data_interfaces:
        raise ValueError("head_direction_source='dlc' but this file has no DLC_Position")

    dlc = nwb.processing["Behavior"]["DLC_Position"]
    front_name, back_name = config.dlc_head_parts

    front_xy = dlc[front_name].data[:].astype(float)
    back_xy = dlc[back_name].data[:].astype(float)
    dlc_t_s = dlc[front_name].timestamps[:].astype(float)

    tracked = np.isfinite(front_xy).all(1) & np.isfinite(back_xy).all(1)
    if tracked.sum() < 2:
        raise ValueError(f"DLC parts {front_name}/{back_name} are empty in this file")

    head_direction = calc_head_direction(
        np.column_stack([dlc_t_s[tracked], front_xy[tracked], back_xy[tracked]]))

    # An angle cannot be interpolated across the +-pi wrap, so carry it as a unit vector
    # and smooth that. Smoothing the vector also shrinks it where the angle is unsteady,
    # which is harmless: only the direction is read back out.
    smooth_bins = config.head_dir_smooth_s / config.bin_s
    cosine = gaussian_filter1d(
        np.interp(bin_centers_s, dlc_t_s[tracked], np.cos(head_direction)), smooth_bins)
    sine = gaussian_filter1d(
        np.interp(bin_centers_s, dlc_t_s[tracked], np.sin(head_direction)), smooth_bins)

    return np.arctan2(sine, cosine)


def _external_lfp_channel(config: Config, n_bins: int) -> np.ndarray:
    """One channel of an externally exported (n_samples, n_channels) LFP .npy.

    Used when the NWB's own LFP is unusable (see docs/data-issues.md). The export is
    assumed to start on the ephys clock's zero, like the spike times.
    """
    import glob

    path = config.external_lfp_npy
    if os.path.isdir(path):
        # some sessions prefix the file with the session name, some don't
        matches = sorted(glob.glob(os.path.join(path, "*lfp_data.npy")))
        if not matches:
            raise FileNotFoundError(f"no *lfp_data.npy in {path}")
        path = matches[0]

    data = np.load(path, mmap_mode="r")
    fs = config.external_lfp_fs

    channel = config.external_lfp_channel
    if channel is None:
        probe = np.asarray(data[: int(min(400 * fs, data.shape[0]))], np.float32)
        freq, power = welch(probe, fs=fs, nperseg=int(8 * fs), axis=0)
        theta = power[(freq >= 6) & (freq <= 10)].mean(0)
        delta = power[(freq >= 2) & (freq <= 4)].mean(0)
        channel = int(np.argmax(theta / delta))

    n_samples = min(int(n_bins * config.bin_s * fs), data.shape[0])
    signal = np.empty(n_samples)
    step = int(fs * 600)          # column of a row-major mmap, in chunks
    for start in range(0, n_samples, step):
        stop = min(start + step, n_samples)
        signal[start:stop] = data[start:stop, channel]
    return signal


def _pick_theta_channel(lfp, lfp_rate_hz: float, n_bins: int, config: Config) -> np.ndarray:
    """Return the channel with the most theta (6-10 Hz) power relative to delta (2-4 Hz)."""
    n_probe_samples = int(min(400 * lfp_rate_hz, lfp.data.shape[0]))
    probe = lfp.data[:n_probe_samples, :].astype(np.float32)

    # The window must be long enough that both bands contain frequency bins; at a high
    # sampling rate a short window resolves nothing below ~15 Hz.
    segment_length = int(min(n_probe_samples, max(2048, 8 * lfp_rate_hz)))
    freq, power = welch(probe, fs=lfp_rate_hz, nperseg=segment_length, axis=0)

    theta_power = power[(freq >= 6) & (freq <= 10)].mean(0)
    delta_power = power[(freq >= 2) & (freq <= 4)].mean(0)
    best_channel = int(np.argmax(theta_power / delta_power))

    n_samples = min(int(n_bins * config.bin_s * lfp_rate_hz), lfp.data.shape[0])
    return lfp.data[:n_samples, best_channel].astype(float)


def load_session(nwb_path: str, node_csv_path: str, config: Config) -> tuple[Session, NWBHDF5IO]:
    """Read one NWB file onto a common grid of time bins.

    Returns the `Session` and the open file handle, which the caller must close.
    """
    nwb_io = NWBHDF5IO(nwb_path, "r")
    nwb = nwb_io.read()

    # --- 1. tracking: keep only frames where the animal was located -----------
    position = nwb.processing["Behavior"]["Position"]["Rat"]
    position_xy_px = position.data[:].astype(float)
    position_t_s = position.timestamps[:].astype(float)

    tracked = np.isfinite(position_xy_px[:, 0]) & np.isfinite(position_xy_px[:, 1])
    tracked_t_s = position_t_s[tracked]
    tracked_x_px = position_xy_px[tracked, 0]
    tracked_y_px = position_xy_px[tracked, 1]

    # --- 2. choose which cells to analyse -------------------------------------
    units = nwb.units
    quality_label = _decode_strings(units["quality_label"][:])
    cell_type = _decode_strings(units["cell_type"][:])
    spike_times_s = [np.asarray(units["spike_times"][i]) for i in range(len(quality_label))]

    keep_unit = np.isin(quality_label, config.quality)
    if config.cell_types is not None:
        keep_unit &= np.isin(cell_type, config.cell_types)
    unit_ids = np.where(keep_unit)[0]

    # --- 3. time grid, and spike counts per bin -------------------------------
    last_spike_s = max(times[-1] for times in spike_times_s if len(times))
    session_end_s = min(tracked_t_s[-1], last_spike_s)

    bin_edges_s = np.arange(0.0, session_end_s + config.bin_s, config.bin_s)
    bin_centers_s = bin_edges_s[:-1] + config.bin_s / 2
    n_bins = len(bin_centers_s)

    spike_counts = np.zeros((n_bins, len(unit_ids)), np.float32)
    for column, unit in enumerate(unit_ids):
        spike_counts[:, column] = np.histogram(spike_times_s[unit], bins=bin_edges_s)[0]

    # --- 4. tracking onto the same grid, then speed and heading ---------------
    track_x_px = np.interp(bin_centers_s, tracked_t_s, tracked_x_px)
    track_y_px = np.interp(bin_centers_s, tracked_t_s, tracked_y_px)
    has_tracking = (bin_centers_s >= tracked_t_s[0]) & (bin_centers_s <= tracked_t_s[-1])

    speed_px_s = _running_speed_px_s(track_x_px, track_y_px, has_tracking, config)

    if config.head_direction_source == "dlc":
        head_direction = _dlc_head_direction(nwb, bin_centers_s, config)
    else:
        head_direction = _travel_direction(track_x_px, track_y_px, config)

    # --- 5. LFP --------------------------------------
    lfp_theta_channel, lfp_rate_hz = None, 0.0
    if config.external_lfp_npy is not None:
        lfp_theta_channel = _external_lfp_channel(config, n_bins)
        lfp_rate_hz = config.external_lfp_fs
    elif "lfp" in nwb.acquisition:
        lfp = nwb.acquisition["lfp"]
        # The rate follows from how long the *electrical* recording ran, so pass the
        # last spike time rather than session_end_s (the camera may stop earlier).
        lfp_rate_hz = infer_lfp_rate_hz(lfp, last_spike_s, config).rate_hz
        lfp_theta_channel = _pick_theta_channel(lfp, lfp_rate_hz, n_bins, config)

    # --- 6. the maze -----------------------------------------------------------
    node_xy_px = pd.read_csv(node_csv_path, header=None,
                             names=["id", "x", "y"])[["x", "y"]].values.astype(float)

    session = Session(bin_centers_s, n_bins, track_x_px, track_y_px, speed_px_s,
                      head_direction, spike_counts, unit_ids,
                      lfp_theta_channel, lfp_rate_hz, node_xy_px, config)
    return session, nwb_io


def measure_px_per_cm(node_xy_px: np.ndarray, segment_cm: float = SEGMENT_CM) -> float:
    """Pixels per centimetre, from the spacing of neighbouring maze nodes."""
    distances = cdist(node_xy_px, node_xy_px)
    np.fill_diagonal(distances, np.inf)         # a node is not its own neighbour
    return float(np.median(distances.min(1)) / segment_cm)
