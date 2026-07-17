# Methods

```
load_session → theta_cycles → rate_maps → decode → lowpass_trajectory → extract_sweeps → alternation → plot
```

`Config` fields marked **ours** are additions or data-forced re-scalings.
The steps use the **Bayesian (Poisson)** decoder with ~50 ms smoothing
(`decoder="bayes"`); the released code's **PV correlation** (`decoder="pv"`) is the alternative default.
Both feed the same downstream stages.

### 1. Preprocessing
- Curated units (`good` pyramidal; MUA/interneurons off). 10 ms bins; tracking on the same grid; sweep
  run-gate > 15 cm/s.
- `px_per_cm = 1.2435` (from node spacing). Head direction from travel (DLC unusable); LFP rate inferred
  around corrupt timestamps ([data-issues](data-issues.md)).

### 2. Rate maps
- Arena at 5 cm; movement gate > 5 cm/s; drop bins < 0.25 s occupancy; Gaussian σ = 7.5 cm (paper 2.5 cm).
- State space = visited bins dilated by 22.5 cm (3σ), so a sweep can leave the path. Optional 10-fold CV
  picks σ (`--cv-smoothing`).

### 3. Theta phase & cycles
- **Default (ours):** LFP band-passed 5–10 Hz → Hilbert phase.
- **Paper's:** PCA on filtered spike counts (`--theta pca`); fails on our ~16× smaller populations
  (recovers ~0.9 Hz, out of band, vs 7.9 Hz), so LFP is default.
- Phase zero at the theta trough (population firing minimum); keep cycles 0.08–0.22 s.

### 4. Bayesian decoding (per 10 ms bin)
Inputs: spike counts $\mathbf{n}$ ($N × T$, Gaussian-smoothed ~50 ms; PV uses ~10 ms) and tuning $f$
($N × M$, Hz). Poisson log-posterior (uniform prior, $\tau$ = bin width):

$$\log P(x \mid \mathbf{n}) = \sum_i n_i \log f_i(x) - \tau \sum_i f_i(x) + \log P(x)$$

Decoded position = weighted centroid of the top 1 % posterior within 10 cm (not argmax). Keep a bin only
if its peak beats the 99th-percentile spike-shuffle, has ≥ 1 active cell, and its anchor is < 50 cm from
tracking.

### 5. Sweep extraction (per cycle)
- **Anchor** = low-pass decode over the first 40 ms; sweeps measured as *decoded − anchor*.
- **Candidate** = longest run with jumps < 20 cm, turns < 90°, ≥ 4 samples; fold-back truncated.
- **Straightness** (variance along the sweep axis), accept if $r^2 > 0.5$:

$$r^2 = 1 - \frac{\mathrm{Var}(e)}{\mathrm{Var}(x) + \mathrm{Var}(y)}$$

- **Prevalence** = fraction of cycles with a sweep. Optional gates (off by default): forward-only,
  near-animal, L/R alternation.

### 6. Plotting
`plot_sweeps` / `plot_all_sweeps` — Fig. 4a/d style: maze + path, each sweep as purple blobs (light at the
animal → violet at the far end).

## Parameters and source

| `Config` field | value | source |
|---|---|---|
| `bin_s` | 0.010 s | `SweepsSettings.dt = 10e-3` |
| `speed_sweep_cm_s` | 15 | `minSpeed = 0.15` m/s |
| `jump_max_cm`, `turn_max_rad` | 20 cm, π/2 | `chunkThetaPosSweeps.m: maxDist=.2, maxAngleDiff=pi/2` |
| `straightness_min`, `min_valid_samples` | 0.5, 4 | `fig1.m:35` |
| `centroid_percentile`, `centroid_radius_cm` | 99, 10 cm | `processDec` |
| `max_lowpass_error_cm` | 50 / 15 (hc/mec) | `dec.err > .5 / .15` |
| `min_active_cells` | 1 / 5 (hc/mec) | `dec.nactive < 1 / 5` |
| `anchor_n_bins`, `anchor_smooth_cycles` | 4, 1.7 | `runPvPosDecoding.m` |
| `pv_smooth_bins` | 1 (5 → ~50 ms, Bayes) | spike-count Gaussian width |
| `decoder`, `bayes_prior` | `bayes`, uniform | Methods text / Tang et al.; `pv` = `decodePv.m` |
| `spatial_bin_cm`, `rate_smooth_cm` | 5, 7.5 cm | **ours** — re-scaled from 2.5 cm (1.5 m field → 9×5 m maze) |
| `px_per_cm` | 1.2435 | **ours** — measured from node spacing |
| `cv_smoothing` | off | option: pick `rate_smooth_cm` by 10-fold CV |
| `unvisited_margin_cm` | 22.5 cm | **ours** — dilate visited region so a sweep can leave the path |
| `max_sweep_origin_cm` | off (∞) | **ours** — sweep must start near the animal |
| `max_sweep_head_angle_deg` | off | **ours** — sweep must point forward |
| `clip_negative_weights` | off | matches `processDec` (never clips) |
