# Methods: each stage, and where it comes from

Every stage of the port, mapped to the paper's Methods and to the authors' released MATLAB.

```
load_session → theta_cycles → rate_maps → decode
             → lowpass_trajectory → extract_sweeps → alternation
```

---

## Config — the port of `SweepsSettings.m`

| `Config` field | source | note |
|---|---|---|
| `bin_s = 0.010` | `SweepsSettings.m: dt = 10e-3` | the decoding bin |
| `speed_sweep_cm_s = 15.0` | `minSpeed = 0.15` m/s | the running gate |
| `speed_smooth_s = 1.0` | `tsm.speed = 100` samples × 10 ms | |
| `anchor_smooth_cycles = 1.7` | `tsm.pos_slow` | see [divergences](divergences.md) |
| `jump_max_cm = 20.0` | Methods: "jumped < 20 cm" | |
| `straightness_min = 0.5` | Methods: r² > 0.5 | |
| `spatial_bin_cm = 5.0`, `rate_smooth_cm = 7.5` | `runPvPosDecoding.m` | |
| `px_per_cm = 1.2435` | **ours** — measured, not in the paper | `measure_px_per_cm` |
| `max_sweep_origin_cm = 30.0` | **ours** — not in the paper | see below |

---

## 1. `load_session` — one NWB onto a 10 ms grid

Tracking from `Behavior/Position/Rat` (the animal's centroid), interpolated onto the bin grid.
Cells filtered to `quality="good"`, `cell_type="pyramidal"`. Spikes histogrammed into 10 ms bins →
the `(n_bins, n_units)` matrix everything downstream reads.

**Head direction** comes from the **direction of travel** by default (`_travel_direction`), because
the DLC keypoints are unusable ([data-issues §2](data-issues.md)). The paper has true head direction
from LEDs. Travel direction is only a good proxy while the animal runs — hence the 15 cm/s gate on
everything that uses it.

**Divergence: the LFP-rate inference block is entirely ours** and has no counterpart in the paper.
It exists to work around corrupt NWB timestamps ([data-issues §1](data-issues.md)).

## 2. `theta_cycles` — cutting time into theta cycles

Phase from the LFP (band-pass 5–10 Hz + Hilbert) **or**, matching the paper, from the population:
band-pass the spike-count matrix, take the first two PCs, and read the angle around the circle the
population traces once per theta cycle.

Phase is then **re-zeroed at the population's firing minimum** (the paper's convention), so "the
start of a cycle" means the same thing regardless of source. Cycles begin at upward zero-crossings;
inter-onset intervals outside 0.08–0.22 s (4.5–12.5 Hz) are discarded as not-theta.

**Divergence: we default to `theta_source="lfp"`; the paper uses PCA.** Our populations are too small
for a clean PCA reconstruction (its phase advances at 0.89 Hz with only 59% of bins moving forwards).
`--theta pca` reproduces them exactly, and changes no result.

## 3. `rate_maps` — each cell's tuning curve (`decodePv.m`)

Occupancy-normalised, Gaussian-smoothed 2D firing map per cell, built **only from running bins**
(> 5 cm/s), keeping positions visited > 0.25 s.

The load-bearing line is the last one:

```python
tuning_curves = tuning_curves / (tuning_curves.mean(0, keepdims=True) + 1e-9)
```

`decodePv.m`'s **mean-normalisation** — each cell divided by its own mean rate, so a high-firing cell
cannot dominate the across-cell correlation.

**The decoder is not constrained to the maze.** Sweeps reaching never-visited, inaccessible locations
is the paper's central claim; snapping decoded positions onto corridors would delete the phenomenon.
`unvisited_margin_cm` optionally dilates the bins past the visited area so a sweep can leave the
travelled path.

## 4. `decode` — where is the population "saying" it is? (`decodePv.m`, `processDec`)

Two pieces:

* `_correlate_across_units` — **Pearson r across cells** between the current activity vector and each
  position's tuning vector → a `(time, position)` correlation map. This is a PV-correlation decoder,
  not Bayesian.
* `_thresholded_centroid` — `processDec` with `fast=0`. The decoded position is **not the argmax**;
  it is the weighted centroid of correlations that are either above the per-bin **99th percentile**
  **or** within **10 cm of the peak**. Argmax would quantise every step onto the 5 cm grid; the
  centroid gives continuous positions. Negative weights are clipped so the centroid cannot be dragged
  across the maze.

## 5. `shuffle_threshold` — what does chance reach? (`shuff99`)

Circularly roll each cell's spike train — preserving rate and rhythmicity, destroying place tuning
and cell-cell structure — then take the **99th percentile of all resulting correlations** as one
global scalar (`chunkThetaPosSweeps.m`).

## 6. `lowpass_trajectory` — the anchor (`runPvPosDecoding.m`)

Decode a position from **only the first 4 bins (40 ms) of each theta cycle** — before the sweep has
departed — smooth *across cycles* (σ = 1.7, then 0.5), and interpolate back onto every bin.

Sweeps are measured as `decoded − anchor`, **not** `decoded − tracked position`, because the encoded
map can drift from the real one. Everything about sweep direction is anchor-relative.

## 7. `extract_sweeps` — one candidate sweep per cycle (`chunkThetaPosSweeps.m`)

Per theta cycle:

1. blank untrustworthy bins (too few active cells / decoder lost the animal / poor correlation);
2. find the bin of **peak population firing** — the middle of the sweep;
3. grow outward while the trajectory stays smooth: no step > 20 cm, no turn > 90°. **The run
   containing the peak, not the longest run in the cycle** — a subtle point, and a bug in the first
   version of this code;
4. take from nearest-the-anchor to furthest-from-anchor; measure **length** (max pairwise distance),
   **direction** (angle of the distal point *relative to the anchor*), and **straightness r²**
   (`var(along axis) / var(total)`, distal point excluded because it defines the axis).

A cycle is a sweep if: ≥ 4 valid samples, r² > 0.5, running > 15 cm/s — **and** (ours) its near end
is within `max_sweep_origin_cm` of the animal.

> **Divergence: `max_sweep_origin_cm` is not in the paper.** Their decoder is accurate enough that a
> sweep always departs from the animal, so the criterion never binds. Here it rejects "sweeps" made
> of decoded positions that drift around the far side of the maze and never come near the animal at
> all. It cuts the 10-session count from 84 to 33. Set it very large to switch it off.

## 8. `alternation` — the paper's central result (`computeAlternationPercent.m`, `egoRightLeft.m`)

Convert each sweep's direction to **head-centred** coordinates, then measure the fraction of
**adjacent theta-cycle triplets** whose turn-sign flips (left-right-left or the reverse), against a
1000-iteration shuffle.

Two honesty notes baked into the code:

* if there are **zero adjacent triplets**, the statistic is **undefined, not zero** — this is the
  situation in all 10 sessions, and the CLI says so;
* tracking is in image pixels, whose y axis points **down**, so a raw angle difference is mirrored.
  `head_centred_direction` negates it. Alternation counts sign *flips* and is invariant to a global
  sign, so this affects labels and figures only.

## 9. `plot_sweeps` — Fig. 4a/d

The maze; the session's trajectory faint; the segment as a black arrow; then per theta cycle, the
decoded positions of its sweep as circles running **cyan → magenta with sweep time**, and a **green
arrow for the direction — drawn from the anchor**, because that is what the direction is measured
against.

Rejected cycles are drawn faint grey **and labelled with why**. A figure showing only what survived
cannot tell you whether the criteria chose sensibly.
