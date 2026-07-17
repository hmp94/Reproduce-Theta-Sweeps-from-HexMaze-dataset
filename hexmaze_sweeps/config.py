"""Every tunable number, and the live status line.

Units: the data lives in pixels; suffixes say what a variable is (_px, _cm, _s, _hz).
Parameters are in centimetres and `Config.px_per_cm` converts them, so the scale enters
the spatial bin size, the rate-map smoothing width, the jump limit, the centroid radius,
both speed gates, and the reported numbers. Only the dimensionless parameters escape it:
the turn limit (an angle), straightness (a variance ratio), sample counts, percentiles.

Results are NOT scale-invariant. `bin_size_px = spatial_bin_cm * px_per_cm` and the maze
is a fixed number of pixels wide, so the scale sets how many bins tile it (1200 visited
positions at 1.2435 px/cm, 939 at 1.658). `px_per_cm` is measured from the maze rather
than assumed; see `measure_px_per_cm`.
"""
from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")

# Distance between neighbouring maze nodes, i.e. one corridor segment.
SEGMENT_CM = 40.0

# The paper's headline numbers, quoted in printouts for comparison.
PAPER_PREVALENCE = 0.48
PAPER_SWEEP_LENGTH_CM = 22.5
PAPER_ALTERNATION_PCT = 79.8
PAPER_ALTERNATION_SHUFFLE_PCT = 61.1


# Live progress
#
# A session takes minutes, so each stage reports as it starts. Progress goes to stderr
# and overwrites itself in place, which keeps it off stdout: the results table stays
# =============================================================================
# clean and can still be piped to a file.
# =============================================================================
def status(message: str = "") -> None:
    """Overwrite the one-line status on stderr. An empty message erases it."""
    if sys.stderr.isatty():
        sys.stderr.write("\r\x1b[2K" + message)
    elif message:
        sys.stderr.write(message + "\n")     # not a terminal: one line each, no control codes
    sys.stderr.flush()


def _elapsed(since: float) -> str:
    """Seconds since `since`, as m:ss."""
    seconds = int(time.time() - since)
    return f"{seconds // 60}:{seconds % 60:02d}"


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    """Every tunable number. Values are the authors' unless a comment says otherwise."""

    # -- scaling factors ------
    px_per_cm: float = 1.2435

    # -- time base (SweepsSettings.m: dt = 10e-3) ------------------------------
    bin_s: float = 0.010

    # -- theta -----------------------------------------------------------------
    theta_band_hz: tuple = (5.0, 10.0)
    theta_source: str = "lfp"           # "pca" reconstructs theta from spiking, as the paper does
    lfp_rate_hz: float | None = None    # None -> work it out; see infer_lfp_rate_hz
    cycle_min_s: float = 0.08           # 0.08-0.22 s = 4.5-12.5 Hz; anything else is not theta
    cycle_max_s: float = 0.22

    # -- choosing cell types ----------------------------------------------------
    # Multi-unit clusters are off
 
    quality: tuple = ("good",)              # ("good", "mua") to include MUA
    cell_types: tuple = ("pyramidal",)      # None to also keep interneurons

    # -- speed and head direction ----------------------------------------------
    speed_smooth_s: float = 1.0             # SweepsSettings.m: tsm.speed = 100 samples
    speed_despike_cm_s: float | None = None  # e.g. 150.0 to drop tracking jumps; not in the paper
    head_dir_smooth_s: float = 0.10
    head_direction_source: str = "dlcl"   # "dlc" reads two body parts from DLC_Position
    dlc_head_parts: tuple = ("nose", "neck")  # (front, back); the angle runs back -> front

    # -- rate maps (runPvPosDecoding.m) -----------------------------------------
    spatial_bin_cm: float = 5.0
    rate_smooth_cm: float = 7.5
    speed_spatial_cm_s: float = 5.0         # a rate map describes where a cell fires while moving
    min_occupancy_s: float = 0.25           # drop positions the animal barely visited

    # The paper picks the smoothing width by cross-validation, not a fixed value (Methods
    # step 1). `--cv-smoothing` does this per session; off by default because it is slow.
    cv_smoothing: bool = False
    cv_smoothing_candidates_cm: tuple = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 50.0)
    cv_folds: int = 10                      # the paper's 10-fold


    unvisited_margin_cm: float = 22.5       # = 3 * rate_smooth_cm

    # -- decoder (decodePv.m, processDec) ---------------------------------------
    # "pv"    -- population-vector correlation across cells (decodePv.m), original authors'
    #            method
    # "bayes" -- Poisson Bayesian reconstruction.
    decoder: str = "pv"
    bayes_prior: str = "uniform"            # or "occupancy": P(x) from time spent at x
    bayes_min_rate_hz: float = 1e-3         # rate floor, so log(f) is finite in empty bins

    pv_smooth_bins: float = 1.0             # a raw 10 ms bin is mostly zeros
    centroid_percentile: float = 99.0       # which positions enter the decoded average
    centroid_radius_cm: float = 10.0
    decoded_smooth_bins: float = 1.0        # 0.8 for entorhinal cortex

    # `processDec` takes a weighted mean and never clips, so an anti-correlated position near
    # the peak keeps its negative weight. Clipping is ours, off by default; no-op for "bayes".
    clip_negative_weights: bool = False

    # -- the anchor, a.k.a. lowpass trajectory (runPvPosDecoding.m) -------------
    anchor_n_bins: int = 4                  # first 40 ms of a cycle, before the sweep departs
    anchor_smooth_cycles: float = 1.7
    anchor_post_smooth_cycles: float = 0.5
    lowpass_smooth_bins: float = 1.0

    # -- which decoded bins to trust (chunkThetaPosSweeps.m; hippocampus values) -
    n_shuffle_bins: int = 10000
    shuffle_percentile: float = 99.0
    min_active_cells: int = 1
    max_lowpass_error_cm: float = 50.0      # beyond this the decoder has lost the animal
    # A floor on the decoder's peak score. None -> compare against the shuffled null
    # instead. Ignored by decoder="bayes": a fixed floor is meaningless for a posterior
    # probability, so Bayesian decoding always uses the shuffled null.
    min_peak_correlation: float | None = 0.0

    # -- sweep extraction (chunkThetaPosSweeps.m) -------------------------------
    jump_max_cm: float = 20.0
    turn_max_rad: float = np.pi / 2
    min_valid_samples: int = 4
    straightness_min: float = 0.5
    speed_sweep_cm_s: float = 15.0          # SweepsSettings.m: minSpeed = 0.15 m/s

    # This to maintain the sweep orginate from the animal sweep forward
    max_sweep_origin_cm: float = float("inf")
    max_sweep_head_angle_deg: float | None = None

    def px(self, cm: float) -> float:
        """Convert centimetres to pixels."""
        return cm * self.px_per_cm

    @property
    def bin_rate_hz(self) -> float:
        """Spike-count bins per second."""
        return 1.0 / self.bin_s
