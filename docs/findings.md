# Findings: why it does not reproduce

**At 10 ms — the timescale a theta sweep needs — the population carries no decodable position, and no
decoder, smoothing, or parameter choice changes that.** The left–right alternation, the paper's
headline, does not reproduce under any setting tried.

## Headline (paper-faithful PV decoder)

| | this dataset | Vollan et al. |
|---|---|---|
| cells / session | 42–83 good pyramidal | 769 mean (384–1,522) |
| sweep prevalence | ~0.001 | 0.48 |
| alternation | undefined — 0 adjacent-cycle triplets | 79.8% vs 61.1% shuffled |

## Cause: the decoder is at chance at 10 ms

Cross-validated position decoding (blocked CV, `Rat6_20260623`, 83 units). `null` = predict the
training centroid, ignoring spikes.

```
  bin    empty bins   RF error   null     beats null
 10 ms     19.1%       253 cm    252 cm     −0 cm    ← sweep timescale
 50 ms      0.2%       253 cm    253 cm     −1 cm
100 ms      0.0%       244 cm    252 cm      8 cm
```

At 10 ms the decoder is **exactly at chance**. An independent Bayesian decoder agrees, across every
population and state space (2.5 cm grid, 5 cm grid, 98 maze nodes).

There is a hard floor underneath this: 19% of running 10 ms bins (52% with MUA off) hold zero spikes
from all cells — an identical all-zero vector, and no method can assign it a position. The paper
decodes large MEC/parasubiculum ensembles where most cells fire every cycle; these are a single
hippocampal probe in a ~40 m² maze.

## Ruled out (each with its decisive number)

- **Speed gate** — prevalence *falls* as the gate loosens; triplets stay 0 at every gate.
- **Theta source** (LFP vs the paper's PCA) — changes *when* you look, not *what* you see; both null.
- **State-space size** — dilating position bins saturates at ~6,800 bins (3σ of rate-map smoothing);
  sweeps already reach never-visited space; 98-node decoding is still at chance.
- **CV smoothing** — the paper picks smoothing by 10-fold CV; we do too. Every width is at chance and
  *less* smoothing is always better (2.5 cm → 252 cm error; 15 cm → 267 cm), the signature of no field
  to sharpen.
- **Head direction / DLC** — feeds only alternation + figures, never decoding; the DLC run equals the
  travel run.
- **LFP timestamp bug** — real (data-issues §1) but fixing it changed nothing; the 4 never-broken
  files are just as empty.
- Also: bin-indexing (100% self-test), spike/position clock alignment, tracking quality, session choice.

## Bayesian decoder + heavy smoothing (Tang et al. 2026) — same conclusion

Tang et al. 2026 use the *same* sweep code as Vollan with a **Bayesian** decoder and ~50 ms spike
smoothing. On our data (`decoder="bayes"`, 50 ms), across all 10 sessions
([figures/tang_10sessions.png](figures/tang_10sessions.png)):

- sweep prevalence beats a spike-shuffle control **3.3×** → some within-cycle structure survives;
- **but alternation clears the 99.9% shuffle ceiling in 0 of 10 sessions.**

And **46% of these "sweeps" point backward** relative to heading — near the 50% expected from random
directions ([figures/sweep_directions_rose.png](figures/sweep_directions_rose.png)). Requiring forward
sweeps (`max_sweep_head_angle_deg=90`) removes the mild above-chance alternation entirely. So heavier
smoothing surfaces smooth, above-chance sweep-*like* trajectories — but they are near-directionless,
not the forward-going, alternating sweeps the paper reports.

## The one route not yet closed: LMT

The paper switches from PV correlation to a **Latent Manifold Tuning** model exactly when PV runs out
(PV can only decode positions the animal physically sampled). LMT fits latent trajectories to spikes
directly, so it is not bounded the same way. Whether it survives ~67 cells is untested — the only
remaining avenue. A working implementation ships with the Tang et al. repo (`external/LMT`).
