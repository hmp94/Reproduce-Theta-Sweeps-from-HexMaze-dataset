# Where the code and the paper disagree

Two kinds, and it matters which. Everything here is a reversible `Config` flag.

## 1. Authors' MATLAB vs their own Methods text — we follow the code

| the code does | the Methods say |
|---|---|
| decoded position = weighted **centroid** (`processDec`, `fast=0`) | "the bin that maximized P(x\|y)" (argmax) |
| state space = the **whole rectangle**, unvisited bins kept (`rmap(isnan)=eps`) | not stated |
| `dec.err` gate: drop bins where the anchor is > 50/15 cm from tracking | not in the Methods; load-bearing (passes ~16% of bins here) |
| sweep = the smooth run **containing the firing peak** | ambiguous |
| decoder is **PV correlation** (`decodePv`) | describes **Bayesian** reconstruction |

The last row is why both decoders are implemented (`decoder="pv"` / `"bayes"`); the code wins for the
default, the Methods motivate the option.

## 2. This port vs the authors — forced by the data

| deviation | why | flag |
|---|---|---|
| `theta_source="lfp"` (paper: population PCA) | PCA phase fails on our ~16× smaller populations (0.89 Hz, 59% forward, vs LFP 7.88 Hz, 100%) — silently | `--theta pca` |
| `head_direction="travel"` (paper: LED) | DLC keypoints unusable (data-issues §2); never touches decoding | `--head-direction dlc` |
| `spatial_bin_cm=5`, `rate_smooth_cm=7.5` (paper: 2.5 cm) | re-scaled: a 9 × 5 m maze is not a 1.5 m open field | — |
| `unvisited_margin_cm=22.5` | dilate the visited bins so a sweep can leave the path (their rectangle does this for free) | `--unvisited-margin-cm` |
| `px_per_cm=1.2435` | measured from node spacing; **not** scale-free (sets how many bins tile the maze) | `--px-per-cm` |

## Our added acceptance gates — all off by default

| gate | what it rejects |
|---|---|
| `max_sweep_origin_cm` | "sweeps" whose near end never comes near the animal (drift across the maze) |
| `max_sweep_head_angle_deg` | backward sweeps — ~46% of Bayes + 50 ms sweeps point away from heading (see [findings](findings.md)) |

Nothing is clipped, snapped, or gated beyond the paper by default; these are opt-in, and each is
documented where it changes a result.
