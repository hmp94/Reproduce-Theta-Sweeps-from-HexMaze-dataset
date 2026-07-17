# Findings: why it does not reproduce

At 10 ms — the timescale a sweep needs — the population carries no decodable position, and no decoder,
smoothing, or parameter changes it. The left–right alternation does not reproduce under any setting.

## Result (paper-faithful PV decoder)

| | this dataset | Vollan et al. |
|---|---|---|
| cells / session | 42–83 good pyramidal | 769 mean (384–1,522) |
| sweep prevalence | ~0.001 | 0.48 |
| alternation | undefined — 0 adjacent-cycle triplets | 79.8% vs 61.1% shuffled |

## Cause: chance-level decoding at 10 ms

Cross-validated decoding (blocked CV, `Rat6_20260623`, 83 units; `null` = predict the training centroid):

```
  bin    empty bins   RF error   null    beats null
 10 ms     19.1%       253 cm    252 cm    −0 cm   ← sweep timescale
 50 ms      0.2%       253 cm    253 cm    −1 cm
100 ms      0.0%       244 cm    252 cm     8 cm
```

At 10 ms the decoder is exactly at chance; an independent Bayesian decoder agrees across every population
and state space. Hard floor: 19% of running 10 ms bins (52% with MUA off) hold zero spikes from all cells,
an all-zero vector no method can place. The paper decodes large MEC/parasubiculum ensembles; this is one
hippocampal probe in a ~40 m² maze.

## Ruled out (each with its decisive number)

- **Speed gate** — prevalence falls as the gate loosens; triplets stay 0.
- **Theta source** (LFP vs PCA) — changes when you look, not what you see; both null.
- **State-space size** — dilation saturates at ~6,800 bins; 98-node decoding still at chance.
- **CV smoothing** — every width at chance; less smoothing always better (no field to sharpen).
- **Head direction / DLC** — feeds only alternation and figures; DLC run = travel run.
- **LFP timestamp bug** — real (data-issues §1), but fixing it changed nothing.
- Also ruled out: bin-indexing, spike/position clock alignment, tracking quality, session choice.

## Bayesian + 50 ms smoothing (Tang et al. 2026) — same conclusion

Same sweep code, Bayesian decoder, ~50 ms smoothing, all 10 sessions: prevalence beats a spike-shuffle
**3.3×**, but alternation clears the 99.9% shuffle in **0 of 10**. And **46% of these sweeps point
backward** (≈ chance); requiring forward sweeps removes the mild above-chance alternation. Heavier
smoothing surfaces near-directionless sweep-*like* trajectories, not the forward alternating sweeps.

## One route not closed: LMT

The paper switches to a **Latent Manifold Tuning** model when PV runs out. LMT fits latent trajectories to
spikes directly, so it is not position-bounded. Whether it survives ~67 cells is untested — the only
remaining avenue (`external/LMT` in the Tang repo).
