# Theta sweeps on the HexMaze dataset

Reproducing **Vollan, Gardner, Moser & Moser (2025), "Left–right-alternating theta sweeps in
entorhinal–hippocampal maps of space", *Nature* 639:995–1005** using python, run on our hippocampal Neuropixels
recordings from a 9 × 5 m hex maze (10 NWB sessions).

## Result 

| | this dataset | Vollan et al. |
|---|---|---|
| cells / session | 42–83 good pyramidal | 769 mean |
| sweep prevalence | 0.001 (PV) · 0.01–0.05 (Bayes + 50 ms) | 0.48 |
| left–right alternation | fails — **0 / 10** sessions clear the shuffle | 79.8 % vs 61.1 % shuffled |

At **10 ms** (paper's parameter) the timescale a sweep needs — the population carries **no decodable position**: a
cross-validated decoder sits at chance (~252 cm error), and **19–52 % of running bins hold zero spikes
from every cell**. The paper's effect rests on large MEC/parasubiculum ensembles; this is a single
hippocampal probe. The heavier-smoothed **Bayesian** variant (Tang et al. 2026) does surface sweep-like
trajectories, but their L/R alternation still fails in every session. Full evidence and every rescue
tried: [findings](docs/findings.md).

## Install & run

```bash
pip install -e .
hexmaze-sweeps --plot figures -o results.csv        # every *.nwb in the working dir
```

```python
from hexmaze_sweeps import Config, run, plot_sweeps
result = run("Rat6_20260629.nwb", "node_list_new.csv", Config(decoder="bayes"))
plot_sweeps(result, save_path="sweeps.png")
```

Key flags (defaults are paper-faithful):

| flag | effect |
|---|---|
| `--decoder pv\|bayes` | PV correlation (authors' code) or Poisson Bayesian (their Methods text / Tang et al.) |
| `--theta lfp\|pca` | theta phase from LFP, or from spiking as the paper does |
| `--cv-smoothing` | pick rate-map smoothing by 10-fold cross-validation, as the paper does |
| `--head-direction travel\|dlc` | heading from motion, or from DLC (noisy here) |
| `--quality good\|good+mua` | multi-unit clusters off by default |

## Pipeline & layout

```
load_session → theta_cycles → rate_maps → decode
             → lowpass_trajectory → extract_sweeps → alternation
```

| module | owns |
|---|---|
| `config.py` | every parameter |
| `data.py` | reading + validating an NWB (LFP-rate inference) |
| `decoding.py` | theta cycles, rate maps, PV + Bayesian decoders, CV smoothing, the anchor |
| `sweeps.py` | sweep extraction, alternation |
| `plotting.py` | Fig. 4a/d-style figures |
| `pipeline.py` | `run()` and the CLI |

## Data defects — read before trusting numbers

Full write-up: [data-issues](docs/data-issues.md).

| defect | status |
|---|---|
| 6/10 NWBs stamp the 1 kHz LFP at 30 kHz → silently frozen theta phase | **corrected** |
| DLC keypoints unusable (likelihood column dropped on write) | flagged, not fixed |
| `node_list_new.csv` missing the long inter-cluster corridors | worked around |
| Camera 8 *looked* broken — it wasn't; the node list was | resolved |

MIT licence.
