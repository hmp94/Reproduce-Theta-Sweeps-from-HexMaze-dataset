# Divergences from the paper

Two kinds, each a reversible `Config` flag.

## 1. Released MATLAB vs the Methods text — the port follows the code

| MATLAB code | Methods |
|---|---|
| decoded position = weighted **centroid** (`processDec`, `fast=0`) | "the bin that maximized P(x\|y)" (argmax) |
| state space = the **whole rectangle**, unvisited bins kept (`rmap(isnan)=eps`) | not stated |
| `dec.err` gate drops bins where the anchor is > 50/15 cm from tracking | not stated; consequential (~16% of bins pass) |
| sweep = the smooth run **containing the firing peak** | ambiguous |
| decoder is **PV correlation** (`decodePv`) | describes **Bayesian** reconstruction |

Both decoders are implemented: `decoder="pv"` (default, matches the code), `"bayes"` (matches the Methods).

## 2. This port vs the authors — required by the data

| deviation | why | flag |
|---|---|---|
| `theta_source="lfp"` (paper: PCA) | PCA phase fails on our ~16× smaller populations (0.89 Hz vs LFP 7.88 Hz) | `--theta pca` |
| `head_direction="travel"` (paper: LED) | DLC keypoints unusable (data-issues §2); never touches decoding | `--head-direction dlc` |
| `spatial_bin_cm=5`, `rate_smooth_cm=7.5` (paper: 2.5 cm) | re-scaled: a 9 × 5 m maze is not a 1.5 m field | — |
| `unvisited_margin_cm=22.5` | dilate visited bins so a sweep can leave the path | `--unvisited-margin-cm` |
| `px_per_cm=1.2435` | measured from node spacing; sets how many bins tile the maze | `--px-per-cm` |

## Added acceptance gates — off by default

| gate | rejects |
|---|---|
| `max_sweep_origin_cm` | sweeps whose near end never reaches the animal |
| `max_sweep_head_angle_deg` | backward sweeps (~46% of Bayes + 50 ms sweeps; see [findings](findings.md)) |

By default nothing is clipped, snapped, or gated beyond the paper.
