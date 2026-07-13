# Findings: why the analysis does not reproduce

**At 10 ms the population carries no decodable position information**, and no choice made anywhere in
the pipeline changes that.

This is the audit trail — every hypothesis raised, and how it was closed — so the next person doesn't
re-derive it.

---

## The headline

```
                        this dataset          Vollan et al.
cells per session       42-83 good pyr.       769 mean (384-1,522)
sweep prevalence        0.001                 0.48
mean sweep length       37-44 cm              22.5 cm
alternation             UNDEFINED             79.8% vs 61.1% shuffled
                        (0 adjacent triplets)
```

33–87 sweeps out of ~76,000 running theta cycles across 10 sessions, depending on criteria. **Zero**
runs of three consecutive theta cycles that all contain a sweep — so the paper's alternation
statistic is *undefined*, not zero.

## The cause: the decoder is at chance at 10 ms

Cross-validated position decoding, 5-fold blocked CV, `Rat6_20260623` (83 units, our best session).
`null` = a decoder that ignores the spikes and always predicts the training centroid.

```
    bin   empty     kNN      RF    null   beats null by
   10ms   19.1%    284cm   253cm   252cm         -0cm    <- theta-sweep timescale
   50ms    0.2%    260cm   253cm   253cm         -1cm
  100ms    0.0%    258cm   244cm   252cm          8cm
 1000ms    0.0%    248cm   236cm   244cm          8cm
```

**At 10 ms a Random Forest is exactly at chance.** Not "poor" — at chance. An independent Bayesian
rate-map decoder agrees (236–253 cm vs a 241–249 cm chance level), across every population definition
and every state space (2.5 cm grid, 5 cm grid, 98 maze nodes).

**The information floor:** 19% of *running* 10 ms bins contain zero spikes from all 83 cells. With
MUA off and no speed gate it is **52%** (median **0** active cells/bin). Those bins are an identical
all-zero vector — no method can assign different positions to identical inputs. This is an
information-theoretic floor, not a modelling weakness.

**Why:** the paper decodes from large MEC/parasubiculum grid-cell ensembles, where a large fraction
of cells fires every theta cycle. Hippocampal place cells are sparse and the HexMaze is ~40 m². These
recordings are a single hippocampal probe.

---

## Ruled out

### Speed filtering is not the constraint

```
 gate cm/s  running cyc  sweeps  prevalence  triplets
         0       219214     136      0.0006         0     <- gate OFF
        15        86361      87      0.0010         0     <- paper's gate
        30        30487      44      0.0014         0
```

Prevalence *falls* as you loosen the gate. And the decisive number:

> With **no speed gate at all**, only **147 of 318,222 theta cycles (0.05%)** produce a decoded
> trajectory that is even 4 samples long and straight. The paper needs 48%.

The binding constraint is **straightness**, not speed: the decoded trajectory cannot hold a straight
line for 40 ms regardless of what the animal is doing. Triplets stay at zero at every gate.

### The theta source does not matter

```
theta source    cycles  running  sweeps    prev   len_cm  triplets
lfp              32198     8617      12   0.001     36.5         0
pca              30556     7899       9   0.001     43.8         0
```

Theta determines *when* you look; the decoder determines *what you see*. If the decoder is at chance,
it does not matter when you look.

(The paper's PCA method is itself unusable here — its phase advances at 0.89 Hz with only 59% of bins
moving forwards, against LFP theta's 7.88 Hz and 100%. But it **fails silently**, still returning
~30.6k plausible-looking cycles.)

### UMAP cannot help

The paper *does* use UMAP — but as a decoder of **internal direction** (a 1-D ring manifold), on
**theta-cycle bins, not 10 ms bins**, from **theta-rhythmic direction-tuned cells**, 85.6% of which
sit in the **parasubiculum**. Their example used **533 such cells**.

Applying the paper's own inclusion rule (head-direction MVL > 0.3 **and** theta-phase MVL > 0.3) to
our best session gives **3 cells**. You cannot fit a ring manifold to three. The route is closed not
because UMAP is weak but because **the analysis needs a cell type we didn't record**.

### Binning / state-space size is not the constraint

Dilating the position bins past the visited area, so sweeps can leave the travelled path:

```
  margin  pos bins  sweeps    prev   decoded sweep beyond visited
     0cm      1113      14  0.0014        8.6 cm (max 22)
    20cm      6616      11  0.0011        6.7 cm (max 36)
    30cm      6812      11  0.0011       12.6 cm (max 42)
```

Two things fall out. **Sweeps already reach never-visited space** — even at margin 0, the decoded
endpoint lands a median 8.6 cm outside anywhere the rat has been, because the centroid can fall
outside the bins it averages. And the bin count **saturates at ~6,800**: past ~22 cm (3σ of the
7.5 cm rate-map smoothing) there is no tuning to carry. That is the PV decoder's hard reach limit —
the paper hits it too, and switches to LMT.

Coarsening the state space was ruled out separately: decoding on **98 maze nodes** is still at chance.

### Head direction does not matter

DLC head direction is broken ([data-issues §2](data-issues.md)) but feeds *only*
`head_centred_direction` → alternation and the figures. It never touches cycle detection, rate maps,
decoding, or sweep detection. The 10-session DLC run is identical to the travel run.

### The LFP timestamp bug was real, and fixing it changed nothing

```
                        sessions   cycles   running  sweeps   prev   triplets
stamped 30 kHz (fixed)     6      188,307   49,851     56    0.001      0
stamped  1 kHz (clean)     4      129,915   36,510     31    0.001      0
```

Indistinguishable. This validates the fix — the day effect is gone, the corrupted files now sit right
on top of the clean ones — **and** rules out the metadata bug as an explanation for the null, since
the four files that were never broken are just as empty.

### Also ruled out earlier

Bin-indexing (100% exact-rPV self-test) · spike/position clock misalignment (LFP theta amplitude vs
speed peaks at lag 0) · tracking quality · rate-map over-smoothing (fixing it raises spatial
information 0.32 → 0.52 bits; decoding stays at chance) · heading-conditioned rate maps (worse) ·
session choice (all 10 surveyed).

---

## Open: sweeps are ~2× too long, probably a selection effect

Accepted sweeps average 37–44 cm against the paper's 22.5 cm.

A sweep grows outward from the peak-firing bin while consecutive decoded positions stay within
**20 cm per 10 ms step**. Real decoded positions are strongly autocorrelated, so each step is a few
cm and the excursion is smooth and short. Ours are near-independent draws, so each step runs close to
the 20 cm ceiling. The criteria then ask for **≥4 samples and straightness r² > 0.5** — and among
random walks, the ones that happen to look *straight* are precisely those whose steps happened to
*align*, which is also what makes them travel far.

So we may be selecting long, straight random walks and calling them sweeps. **Untested prediction:**
circularly shuffling each cell's spike train (destroying place coding, preserving rate and rhythm)
should yield a similar number of similarly-long "sweeps".

## The one avenue not yet closed: LMT

The paper switches from PV-correlation to a **Latent Manifold Tuning** model precisely when PV runs
out:

> "The PV-correlation method has two limitations. First, **it can only be used to decode positions and
> directions that the animal has physically sampled**, because it relies on tuning curves with respect
> to the animal's tracked position."

LMT fits latent 1-D and 2-D trajectories to the spikes iteratively rather than reading them off
pre-computed tuning curves, so it is not bounded in the same way. Whether it survives 67 cells is
unknown. It is the only remaining route that isn't already closed.
