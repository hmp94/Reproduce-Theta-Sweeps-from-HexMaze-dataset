# Reproducing theta sweeps on the HexMaze dataset

A Python port of the analysis in **Vollan, Gardner, Moser & Moser (2025), "Left–right-alternating
theta sweeps in entorhinal–hippocampal maps of space", *Nature* 639:995–1005**, applied to
hippocampal Neuropixels recordings from a 9 × 5 m hex maze (Genzel Lab, Donders).

Ported from the authors' released MATLAB (`decodePv.m`, `runPvPosDecoding.m`,
`chunkThetaPosSweeps.m`, `computeAlternationPercent.m`), not from the Methods text alone — the
two disagree in places, and where they do, the code wins. See [docs/divergences.md](docs/divergences.md).

## The result

**The analysis does not reproduce, and the reason is the data, not the port.**

| | this dataset | Vollan et al. |
|---|---|---|
| cells per session | 42–83 good pyramidal | **769 mean** (384–1,522) |
| sweep prevalence | **0.001** | **0.48** |
| mean sweep length | 37–44 cm | 22.5 cm |
| left–right alternation | **undefined** — zero adjacent-cycle triplets | 79.8% vs 61.1% shuffled |

At 10 ms — the resolution a theta sweep requires — the population carries **no decodable position
information**. A cross-validated Random Forest reaches 253 cm against a 252 cm no-information null,
and 19% of *running* 10 ms bins contain **zero spikes from all 83 cells**. Every rescue we tried is
recorded in [docs/findings.md](docs/findings.md); all of them came back negative.

The recordings target hippocampus. The paper's effect rests on large MEC/parasubiculum ensembles.

## Install

```bash
git clone https://github.com/hmp94/Reproduce-Theta-Sweeps-from-HexMaze-dataset.git
cd Reproduce-Theta-Sweeps-from-HexMaze-dataset
pip install -e .
```

## Run

From the directory holding your `.nwb` files:

```bash
hexmaze-sweeps --theta pca --head-direction dlc --plot figures -o results.csv
```

or from Python:

```python
from hexmaze_sweeps import Config, run, plot_sweeps

result = run("Rat6_20260629.nwb", "node_list_new.csv", Config(theta_source="pca"))
plot_sweeps(result, window_s=3.0, save_path="sweeps.png")
```

Useful flags:

| flag | effect |
|---|---|
| `--theta lfp\|pca` | `pca` reconstructs theta from spiking, as the paper does |
| `--head-direction travel\|dlc\|auto` | `auto` uses DLC only where it passes the quality check |
| `--max-sweep-origin-cm` | how close a sweep's near end must be to the animal |
| `--quality good\|good+mua` | multi-unit clusters are off by default |
| `--plot DIR`, `-o CSV` | figures, and the results table |

## Pipeline

```
load_session → theta_cycles → rate_maps → decode
             → lowpass_trajectory → extract_sweeps → alternation
```

| module | what it owns |
|---|---|
| `config.py` | every tunable number; the live status line |
| `data.py` | reading an NWB, **and deciding whether to trust it** — both preflight checks |
| `decoding.py` | theta cycles, rate maps, the PV-correlation decoder, the anchor |
| `sweeps.py` | sweep extraction, the alternation statistic, visited/never-visited space |
| `plotting.py` | one figure, in the style of the paper's Fig. 4a/d |
| `pipeline.py` | `run()` and the CLI |

Method-by-method mapping to the paper and the authors' MATLAB: [docs/methods.md](docs/methods.md).

## Read this before you trust any number

Four defects in the HexMaze data, all found while porting. Two are corrected here; two are not.

| | status |
|---|---|
| **6 of 10 NWBs stamp the 1 kHz LFP at 30 kHz.** `np.interp` then *clamps* rather than extrapolating, silently freezing theta phase after ~131 s — 1,050 cycles instead of ~32,000. Looks exactly like a day effect. | **corrected** (`check_lfp_rates`) |
| **DLC keypoints are unusable.** `create_nwb.py` drops DeepLabCut's likelihood column, so nothing filters low-confidence frames. Nose→neck length varies 2×; the head, body and tail axes all score an identical ~0.48 against direction of travel. | **detected, not fixed** (`check_head_direction`) |
| **`node_list_new.csv` is missing the long corridors** joining hexagon clusters. | documented; the analysis no longer depends on it |
| Camera 8 *looked* broken — it wasn't; the node list was. | resolved |

Full write-up: [docs/data-issues.md](docs/data-issues.md).

## Licence

MIT.
