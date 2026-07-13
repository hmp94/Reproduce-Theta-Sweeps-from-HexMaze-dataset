# Methods: each stage, and where it comes from

```
load_session → theta_cycles → rate_maps → decode
             → lowpass_trajectory → extract_sweeps → alternation
```

## Config — the port of `SweepsSettings.m`

| `Config` field | source |
|---|---|
| `bin_s = 0.010` | `dt = 10e-3` |
| `speed_sweep_cm_s = 15.0` | `minSpeed = 0.15` m/s |
| `speed_smooth_s = 1.0` | `tsm.speed = 100` samples × 10 ms |
| `jump_max_cm = 20.0` | Methods: "jumped < 20 cm" |
| `straightness_min = 0.5` | Methods: r² > 0.5 |
| `spatial_bin_cm = 5.0`, `rate_smooth_cm = 7.5` | `runPvPosDecoding.m` |
| `px_per_cm = 1.2435` | **ours** — measured, not in the paper |
| `max_sweep_origin_cm = 30.0` | **ours** — not in the paper |

## 1. `load_session`

Tracking from `Behavior/Position/Rat` (the animal's centroid) onto a 10 ms grid. Cells filtered to
`quality="good"`, `cell_type="pyramidal"`. Spikes histogrammed into 10 ms bins.

Head direction comes from **direction of travel** by default, because the DLC keypoints are unusable
([data-issues §2](data-issues.md)). The paper has LED-tracked head direction. Travel direction is
only a good proxy while running — hence the 15 cm/s gate on everything using it.

The LFP-rate inference block is **entirely ours**, working around corrupt NWB timestamps
([data-issues §1](data-issues.md)).

## 2. `theta_cycles`

Phase from the LFP (band-pass 5–10 Hz + Hilbert) **or**, matching the paper, from the population
(band-pass the spike-count matrix, take the first two PCs, read the angle around the circle the
population traces once per theta cycle).

Phase is then **re-zeroed at the population's firing minimum** — the paper's convention, which makes
"the start of a cycle" mean the same thing regardless of source. Cycles begin at upward
zero-crossings; intervals outside 0.08–0.22 s are discarded as not-theta.

## 3. `rate_maps` (`decodePv.m`)

Occupancy-normalised, Gaussian-smoothed 2D firing map per cell, from **running bins only** (> 5 cm/s),
keeping positions visited > 0.25 s. The load-bearing line is the last one:

```python
tuning_curves = tuning_curves / (tuning_curves.mean(0, keepdims=True) + 1e-9)
```

`decodePv.m`'s **mean-normalisation** — each cell divided by its own mean rate, so a high-firing cell
cannot dominate the across-cell correlation.

**The decoder is not constrained to the maze.** Sweeps reaching never-visited, inaccessible locations
is the paper's central claim; snapping decoded positions onto corridors would delete the phenomenon.

## 4. `decode` (`decodePv.m`, `processDec`)

* `_correlate_across_units` — **Pearson r across cells** between the current activity vector and each
  position's tuning vector. A PV-correlation decoder, not Bayesian.
* `_thresholded_centroid` — `processDec` with `fast=0`. The decoded position is **not the argmax**;
  it is the weighted centroid of correlations either above the per-bin **99th percentile** **or**
  within **10 cm of the peak**. Argmax would quantise every step onto the 5 cm grid; the centroid
  gives continuous positions.

## 5. `shuffle_threshold` (`shuff99`)

Circularly roll each cell's spike train — preserving rate and rhythmicity, destroying place tuning —
then take the **99th percentile of all resulting correlations** as one global scalar.

## 6. `lowpass_trajectory` — the anchor (`runPvPosDecoding.m`)

Decode a position from **only the first 4 bins (40 ms) of each theta cycle**, before the sweep has
departed; smooth *across cycles* (σ = 1.7, then 0.5); interpolate back onto every bin.

Sweeps are measured as `decoded − anchor`, **not** `decoded − tracked position`, because the encoded
map can drift from the real one. **Everything about sweep direction is anchor-relative.**

## 7. `extract_sweeps` (`chunkThetaPosSweeps.m`)

Per theta cycle:

1. blank untrustworthy bins (too few active cells / decoder lost the animal / poor correlation);
2. find the bin of **peak population firing** — the middle of the sweep;
3. grow outward while the trajectory stays smooth: no step > 20 cm, no turn > 90°. **The run
   containing the peak, not the longest run in the cycle** — subtle, and wrong in the first version
   of this port;
4. take from nearest-the-anchor to furthest-from-anchor; measure **length** (max pairwise distance),
   **direction** (angle of the distal point *relative to the anchor*), and **straightness r²**
   (`var(along axis) / var(total)`, distal point excluded because it defines the axis).

A cycle is a sweep if: ≥ 4 valid samples, r² > 0.5, running > 15 cm/s — **and** (ours) its near end
is within `max_sweep_origin_cm` of the animal. See [divergences](divergences.md).

## 8. `alternation` (`computeAlternationPercent.m`, `egoRightLeft.m`)

Convert each sweep's direction to **head-centred** coordinates, then measure the fraction of
**adjacent theta-cycle triplets** whose turn-sign flips, against a 1000-iteration shuffle.

Two honesty notes in the code:

* if there are **zero adjacent triplets**, the statistic is **undefined, not zero** — the situation
  in all 10 sessions, and the CLI says so;
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
