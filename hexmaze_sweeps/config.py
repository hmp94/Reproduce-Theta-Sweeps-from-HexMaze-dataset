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

    # -- physical scale: the only place centimetres enter the computation ------
    px_per_cm: float = 1.2435

    # -- time base (SweepsSettings.m: dt = 10e-3) ------------------------------
    bin_s: float = 0.010

    # -- theta -----------------------------------------------------------------
    theta_band_hz: tuple = (5.0, 10.0)
    theta_source: str = "lfp"           # "pca" reconstructs theta from spiking, as the paper does
    lfp_rate_hz: float | None = None    # None -> work it out; see infer_lfp_rate_hz
    cycle_min_s: float = 0.08           # 0.08-0.22 s = 4.5-12.5 Hz; anything else is not theta
    cycle_max_s: float = 0.22

    # -- which cells to use ----------------------------------------------------
    # Multi-unit clusters are off: they double the sweep count, but the extra sweeps
    # are longer than the paper's and the rate maps become far less reproducible
    # between halves of the session (r = 0.35 -> r = 0.16).
    quality: tuple = ("good",)              # ("good", "mua") to include MUA
    cell_types: tuple = ("pyramidal",)      # None to also keep interneurons

    # -- speed and head direction ----------------------------------------------
    speed_smooth_s: float = 1.0             # SweepsSettings.m: tsm.speed = 100 samples
    speed_despike_cm_s: float | None = None  # e.g. 150.0 to drop tracking jumps; not in the paper
    head_dir_smooth_s: float = 0.10
    head_direction_source: str = "travel"   # "dlc" reads two body parts from DLC_Position
    dlc_head_parts: tuple = ("nose", "neck")  # (front, back); the angle runs back -> front

    # -- rate maps (runPvPosDecoding.m) -----------------------------------------
    spatial_bin_cm: float = 5.0
    rate_smooth_cm: float = 7.5
    speed_spatial_cm_s: float = 5.0         # a rate map describes where a cell fires while moving
    min_occupancy_s: float = 0.25           # drop positions the animal barely visited

    # How far beyond the visited area the decoder may place the animal. A sweep is
    # supposed to be able to leave the travelled path -- the paper's central claim is that
    # sweeps reach never-visited, inaccessible locations -- so the position bins must not
    # stop at the edge of where the animal went. Dilating the visited region by roughly a
    # sweep length reproduces the authors' rectangular arena grid without carrying the
    # 92% of a hex maze that is wall.
    #
    # Note the ceiling: a PV-correlation decoder can only reach as far out as the rate-map
    # smoothing carries a cell's tuning, because unvisited bins have no data of their own.
    # The paper hits this limit too, and switches to the LMT model to go further.
    unvisited_margin_cm: float = 0.0        # 0 = visited bins only

    # -- decoder (decodePv.m, processDec) ---------------------------------------
    pv_smooth_bins: float = 1.0             # a raw 10 ms bin is mostly zeros
    centroid_percentile: float = 99.0       # which positions enter the decoded average
    centroid_radius_cm: float = 10.0
    decoded_smooth_bins: float = 1.0        # 0.8 for entorhinal cortex

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
    min_peak_correlation: float | None = 0.0  # None -> compare against the shuffled null

    # -- sweep extraction (chunkThetaPosSweeps.m) -------------------------------
    jump_max_cm: float = 20.0
    turn_max_rad: float = np.pi / 2
    min_valid_samples: int = 4
    straightness_min: float = 0.5
    speed_sweep_cm_s: float = 15.0          # SweepsSettings.m: minSpeed = 0.15 m/s

    # A sweep departs from the animal, so its near end must be somewhere near the animal.
    # Not in the paper: their decoder is accurate enough that this never binds. Here it
    # rejects "sweeps" made of decoded positions that drift around the far side of the
    # maze, never touching the animal at all. Set very large to switch it off.
    max_sweep_origin_cm: float = 30.0

    def px(self, cm: float) -> float:
        """Convert centimetres to pixels."""
        return cm * self.px_per_cm

    @property
    def bin_rate_hz(self) -> float:
        """Spike-count bins per second."""
        return 1.0 / self.bin_s
