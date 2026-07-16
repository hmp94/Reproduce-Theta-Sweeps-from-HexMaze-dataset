# Methods: each stage and its source

```
load_session → theta_cycles → rate_maps → decode
             → lowpass_trajectory → extract_sweeps → alternation → plot_sweeps
```

The authors' MATLAB works in **metres**, so every threshold is checkable exactly. Defaults are
paper-faithful; `Config` fields marked **ours** are additions or data-forced re-scalings
([divergences](divergences.md)).

The steps below describe the configuration the current analysis uses: the **Bayesian (Poisson)**
decoder with ~50 ms spike smoothing (`decoder="bayes"`). The authors' released code ships a
**PV-correlation** decoder instead (`decoder="pv"`), which remains available and is the code default;
both feed the identical downstream stages.

## Processing steps

### 1. Preprocessing
- Spike sorting → curated units (kept: `good` **pyramidal**; MUA and interneurons off by default).
- **10 ms** time bins; tracking interpolated onto the same grid; running gate for sweeps **> 15 cm/s**.
- `px_per_cm = 1.2435`, measured from maze node spacing, so every cm↔px conversion is calibrated to
  this maze. Head direction from travel (DLC unusable, [data-issues §2](data-issues.md)); LFP rate
  inferred to work around corrupt timestamps ([data-issues §1](data-issues.md)).

### 2. Rate maps (spatial tuning)
- Arena binned at **5 cm**; rate-map movement gate **> 5 cm/s**; drop bins with **< 0.25 s** occupancy.
- Smoothed with a Gaussian, **σ = 7.5 cm** (re-scaled from the paper's 2.5 cm — a 9×5 m maze is not a
  1.5 m open field).
- Optional **10-fold cross-validation** picks the σ that maximises held-out decoding (`--cv-smoothing`,
  off by default).
- State space = visited bins **dilated by 22.5 cm (= 3σ)**, so a sweep can leave the sampled path
  (not snapped to the maze graph).

### 3. Theta phase & cycles
- LFP band-passed **5–10 Hz** (2nd-order Butterworth) → Hilbert → instantaneous phase *(LFP is the
  default; PCA-on-spikes matches the paper but is unstable on our ~16× smaller populations)*.
- Phase re-zeroed at the population firing minimum; a cycle is kept only if its period is
  **0.08–0.22 s** (≈ 4.5–12.5 Hz).

### 4. Bayesian decoding (per 10 ms bin)
Two input matrices:
- **Spike counts** $\mathbf{n}$: $N$ units × $T$ bins, Gaussian-smoothed over **~50 ms**.
- **Tuning curves** $f$: $N$ units × $M$ positions, in **Hz**.

Poisson log-posterior over positions $x$ (uniform prior $P(x)$, exposure $\tau$ = bin width):

$$\log P(x \mid \mathbf{n}) \;=\; \sum_i n_i \, \log f_i(x) \;-\; \tau \sum_i f_i(x) \;+\; \log P(x)$$

- Decoded position = **weighted centroid of the top 1 % of posterior mass** within a 10 cm radius
  (continuous, not argmax).
- A bin is kept only if its peak clears the **99th-percentile spike-shuffle null**, has **≥ 1 active
  cell**, and its low-pass anchor lies **< 50 cm** from the true position.

### 5. Sweep extraction (per theta cycle)
- **Anchor** = low-pass decoded trajectory over the **first 40 ms** of the cycle — the sweep's origin;
  sweeps are measured as *decoded − anchor*, so direction is anchor-relative.
- **Candidate** = longest run of consecutive decoded bins with jumps **< 20 cm** and turns **< 90°**,
  **≥ 4** samples; the fold-back tail is truncated.
- **Sweep vector** = anchor → most distal decoded point.
- **Straightness** of the run, accepted if $r^2 > 0.5$:

$$r^2 \;=\; 1 - \frac{\mathrm{Var}(e)}{\mathrm{Var}(x) + \mathrm{Var}(y)}$$

  where $e$ is the residual to the fitted sweep axis and $x, y$ are coordinates along it.
- **Prevalence** = fraction of theta cycles containing an accepted sweep.
- *Optional (off by default):* forward-only and near-animal gates; L/R-of-heading labelling and the
  adjacent-cycle **alternation** test (fraction of adjacent triplets whose turn-sign flips, vs a
  1000× shuffle; zero triplets → undefined, not zero).

### 6. Plotting
- `plot_sweeps` — Fig. 4a/d style: maze + animal path/position (optional rat marker), each accepted
  sweep as purple blobs, light (start, at the animal) → violet (far end). `plot_all_sweeps` tiles many
  windows into a montage.

## Parameters and their source

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
| `pv_smooth_bins` | 1 (5 → ~50 ms, Bayes) | spike-count Gaussian width |
| `decoder`, `bayes_prior` | `bayes`, uniform | Poisson posterior (Methods text / Tang et al.); `pv` = `decodePv.m` |
| `spatial_bin_cm`, `rate_smooth_cm` | 5, 7.5 cm | **ours** — re-scaled from their 2.5 cm (1.5 m field → 9×5 m maze) |
| `px_per_cm` | 1.2435 | **ours** — measured from node spacing |
| `cv_smoothing` | off | option: pick `rate_smooth_cm` by 10-fold CV, as the paper does |
| `unvisited_margin_cm` | 22.5 cm | **ours** — dilate visited region so a sweep can leave the path |
| `max_sweep_origin_cm` | off (∞) | **ours** — sweep must start near the animal |
| `max_sweep_head_angle_deg` | off | **ours** — sweep must point forward (≤ this from heading) |
| `clip_negative_weights` | off | matches `processDec` (never clips) |
