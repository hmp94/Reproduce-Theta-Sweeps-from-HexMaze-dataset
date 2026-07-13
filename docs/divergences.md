# Where the code and the paper disagree

Two kinds of divergence, and it matters which is which.

1. **The authors' MATLAB vs their own Methods text.** Where these disagree, this port follows **the
   code**, because the code is what produced the figures.
2. **This port vs the authors.** Deliberate deviations, forced by the data.

---

## 1. Their MATLAB vs their Methods

**The decoder is a thresholded centroid, not an argmax.** The Methods describe Bayesian
reconstruction taking "the position bin that maximized P(x|y)". The released `processDec`
(`fast=0`) instead takes a **probability-weighted centroid** of positions clearing the per-bin 99th
percentile *or* lying within 10 cm of the peak. Not cosmetic: an argmax quantises every decoded step
onto the spatial grid.

**The rate-map state space keeps unvisited bins.**

```matlab
rmap(isnan(rmap)) = eps;      % unvisited bins kept, set to ~0
rmap = rmap(vbins, vbins);    % crop the 1.6x-extended grid to the arena square
tuning(:, u) = rmap(:);       % the WHOLE rectangle is the state space
```

`vbins` is **not a maze mask** — it only crops the extended grid (built so edge-smoothing works) back
to the arena square. This is safe for them: their arenas are **open fields, 1.4–1.8 m across**, so
the rectangle *is* the environment. In a 9 × 5 m hex maze the rectangle is 92% wall, so this port
keeps only visited bins (plus an optional `unvisited_margin_cm` halo).

**The undocumented `dec.err` gate.** Not in the Methods at all: decoded bins are discarded where the
lowpass decoded position is more than **50 cm (hippocampus) / 15 cm (MEC)** from the tracked
position. This is `max_lowpass_error_cm`, and it is load-bearing — on this dataset it passes only
~16% of bins.

**Sweep growth uses the run containing the peak,** not the longest smooth run in the cycle. Easy to
get wrong, and it was wrong in the first version of this port.

**The anchor's smoothing constants.** `tsm.pos_slow = 15` (sigma in samples, dt = 10 ms) does not
line up cleanly with the Methods text; the port follows the code.

---

## 2. This port vs the authors

Three deliberate deviations. Each is a `Config` flag; each is reversible.

### `theta_source = "lfp"` (paper: population PCA)

The paper reconstructs theta phase from the spiking population. Our populations are ~16× smaller and
that reconstruction fails: its phase advances at **0.89 Hz with only 59% of bins moving forwards**
(LFP theta: 7.88 Hz, 100%), PC1+PC2 explain 21.6% of variance, and agreement with true LFP phase is
MVL 0.19. It fails **silently**, still returning ~30.6k plausible-looking cycles.

`--theta pca` reproduces the paper exactly. It changes no result.

### `head_direction_source = "travel"` (paper: LED-tracked head direction)

Our DLC keypoints are unusable ([data-issues §2](data-issues.md)), so heading comes from direction of
travel — a good proxy only while running, hence the 15 cm/s gate.

`--head-direction dlc` forces DLC anyway; `auto` uses it only where it passes the check. Head
direction never touches sweep detection, so this changes no sweep count.

### `max_sweep_origin_cm = 30.0` (not in the paper)

A sweep departs *from* the animal, so its near end must be near the animal. The authors' decoder is
accurate enough that this never binds, so they never state it.

Here it does bind. Without it, "sweeps" are accepted whose decoded positions drift around the far
side of the maze and never come near the animal. It cuts the 10-session count from **84 to 33**, and
takes two sessions to zero.

Set it very large to switch it off — but look at a figure first: `plot_sweeps` draws rejected cycles,
labelled with why.

---

## Not a divergence, though it looks like one

**`px_per_cm = 1.2435`** is measured from the maze (median nearest-neighbour node spacing ÷ the 40 cm
corridor segment), not assumed. It has no counterpart in the paper.

This is **not** scale-free: `bin_size_px = spatial_bin_cm × px_per_cm`, and the maze is a fixed number
of pixels wide, so the scale sets how many bins tile it (1,200 visited positions at 1.2435 px/cm; 939
at 1.658). Ratios *between* cm parameters are scale-free (the jump limit is always
`jump_max_cm / spatial_bin_cm` = 4 bins); the analysis as a whole is not.
