# Methods: each stage and its source

```
load_session → theta_cycles → rate_maps → decode
             → lowpass_trajectory → extract_sweeps → alternation
```

The authors' MATLAB works in **metres**, so every threshold is checkable exactly. Defaults are
paper-faithful; `Config` fields marked **ours** are additions or data-forced re-scalings
([divergences](divergences.md)).

## Parameters

| `Config` field | value | source |
|---|---|---|
| `bin_s` | 0.010 s | `SweepsSettings.dt = 10e-3` |
| `speed_sweep_cm_s` | 15 | `minSpeed = 0.15` m/s |
| `jump_max_cm`, `turn_max_rad` | 20 cm, π/2 | `chunkThetaPosSweeps.m: maxDist=.2, maxAngleDiff=pi/2` |
| `straightness_min`, `min_valid_samples` | 0.5, 4 | `fig1.m:35` |
| `centroid_percentile`, `centroid_radius_cm` | 99, 10 cm | `processDec` |
| `max_lowpass_error_cm` | 50 / 15 (hc/mec) | `dec.err > .5 / .15` |
| `min_active_cells` | 1 / 5 (hc/mec) | `dec.nactive < 1 / 5` |
| `anchor_n_bins`, `anchor_smooth_cycles` | 4, 1.7 | `runPvPosDecoding.m` (`nbins=3`, `smoothSpikes=1.7`) |
| `spatial_bin_cm`, `rate_smooth_cm` | 5, 7.5 cm | **ours** — re-scaled from their 2.5 cm (1.5 m field → 9×5 m maze) |
| `px_per_cm` | 1.2435 | **ours** — measured from node spacing |
| `decoder` | `"pv"` | option: `pv` (decodePv) or `bayes` (Poisson; Methods text / Tang et al.) |
| `cv_smoothing` | off | option: pick `rate_smooth_cm` by 10-fold CV, as the paper does |
| `unvisited_margin_cm` | 22.5 cm | **ours** — dilate visited region so a sweep can leave the path |
| `max_sweep_origin_cm` | off (∞) | **ours** — sweep must start near the animal |
| `max_sweep_head_angle_deg` | off | **ours** — sweep must point forward (≤ this from heading) |
| `clip_negative_weights` | off | matches `processDec` (never clips) |

## Stages

1. **`load_session`** — spikes → 10 ms bin counts; tracking interpolated onto the same grid; good
   pyramidal cells only. Head direction from travel (DLC unusable, data-issues §2). LFP sampling rate
   inferred to work around corrupt timestamps (data-issues §1).
2. **`theta_cycles`** — theta phase from LFP (5–10 Hz + Hilbert) or, matching the paper, from the
   spike-count PCA. Re-zeroed at the population firing minimum; cycles = upward zero-crossings with
   period 0.08–0.22 s.
3. **`rate_maps`** — occupancy-normalised, smoothed firing map per cell, running bins only (>5 cm/s),
   in Hz. `--cv-smoothing` selects the smoothing width by 10-fold blocked CV. State space = visited
   bins dilated by `unvisited_margin_cm`, **not** snapped to the maze (a sweep must be able to leave it).
4. **`decode`** — for each 10 ms bin, score every position, then take a **thresholded centroid**
   (`processDec`, not argmax → continuous positions). `decoder="pv"`: Pearson r across cells, each cell
   mean-normalised. `decoder="bayes"`: Poisson posterior `Σ nᵢ log fᵢ(x) − τ Σ fᵢ(x)` on raw-Hz rates.
5. **`shuffle_threshold`** — circular spike-shift null; 99th percentile of decoder scores.
6. **`lowpass_trajectory`** (the anchor) — decode from the first 40 ms of each cycle, smooth across
   cycles. Sweeps are measured as **decoded − anchor**, so direction is anchor-relative.
7. **`extract_sweeps`** — per cycle: find peak firing, grow the smooth run around it (step <20 cm,
   turn <90°), take nearest-anchor → farthest-anchor; measure length, direction, straightness r².
   Accept on **≥4 samples, r² > 0.5, running > 15 cm/s** (`fig1.m:35`), plus optional origin/forward
   gates.
8. **`alternation`** — sweep direction → head-centred; fraction of adjacent triplets whose turn-sign
   flips, vs a 1000× shuffle. Zero triplets → **undefined, not zero**.
9. **`plot_sweeps`** — Fig. 4a/d style: maze + animal path/position (optional rat marker), each
   accepted sweep as purple blobs, light (start, at the animal) → violet (far end). `plot_all_sweeps`
   tiles many windows into a montage.
