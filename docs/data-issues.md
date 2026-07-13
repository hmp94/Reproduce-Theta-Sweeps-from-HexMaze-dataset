# Defects in the HexMaze dataset

Four problems found while porting the analysis. Two are corrected in code; two are not, and
you need to know about them before you believe anything downstream.

---

## 1. The LFP sampling rate is wrong in 6 of 10 files — **corrected**

An NWB `TimeSeries` encodes timing **either** as a scalar `.rate` **or** as an explicit
`.timestamps` array. These files use `.timestamps` (`.rate` is `None`), so the sampling rate is
never written down anywhere — it exists only as the spacing between consecutive stamps.

Six files stamp the **downsampled 1 kHz LFP with the 30 kHz acquisition rate**:

```
session                   n_samples  stamped fs  stamps span     ephys   true fs
Rat5_20260623.nwb           4177446        1000      4177.4s   4177.4s      1000   ok
Rat5_20260625.nwb           3945522       30000       131.5s   3945.5s      1000   BAD
Rat5_20260626.nwb           4083435       30000       136.1s   4083.4s      1000   BAD
Rat5_20260629.nwb           4054652       30000       135.2s   4054.6s      1000   BAD
Rat6_20260625.nwb           3932160       30000       131.1s   3932.1s      1000   BAD
Rat6_20260626.nwb          18754618       30000       625.2s   3655.6s      1000   BAD
Rat6_20260629.nwb           4695828       30000       156.5s   4695.8s      1000   BAD
```

### Why it is dangerous

`np.interp` **does not extrapolate** — past its last known x it repeats the last y. So theta phase
is real for ~131 s and then **frozen flat** for the remaining ~3,800 s. No upward zero-crossings,
no cycle onsets: the session yields **~1,050 theta cycles instead of ~32,000**.

Nothing errors. Nothing warns. And because the four files with *correct* timestamps happen to be
the 23rd and 24th, it presents as a clean **"sweeps only occur on the 23rd" day effect**. It is
pure metadata corruption.

### The fix

`infer_lfp_rate_hz` resolves the rate in four tiers:

1. a rate pinned by the caller;
2. the stamps, if they span ≥90% of the recording (the four good files);
3. `n_samples / last_spike_time`, snapped to a standard rate (five of the six bad files → exactly
   1000.0 Hz);
4. the **spectrum** — try each candidate rate, keep the one whose 4–12 Hz peak lands inside theta.

Tier 4 exists for **Rat6_20260626**, a second and different anomaly: its LFP holds 18,754,618
samples ≈ 5.2 h (apparently the whole day, including sleep) while the sorted units span 61 min, so
`n_samples/duration` gives a nonsense 5130 Hz. Only the spectrum disambiguates — at 1 kHz the peak
sits at 8.50 Hz; at 5130 and 30000 Hz it sits at the band edge (11.75 Hz), a filtering artefact.

Two guards back it up:

* `_theta_phase_from_lfp` **raises** if the LFP covers <95% of the session, rather than clamping.
* Pass the **last spike time**, not the tracking duration — the LFP is electrophysiology, so its
  length reflects how long the *electrical* recording ran, not when the camera stopped.

> **Tell the data provider:** the NWB writer is stamping the downsampled LFP with the raw
> acquisition rate. Better still, write `.rate` as a scalar instead of a timestamps array.

After the fix all 10 sessions give 26k–35k theta cycles, and the "day effect" disappears.

---

## 2. The DLC keypoints are unusable — **detected, not fixed**

DLC body parts (`processing/Behavior/DLC_Position`: nose, neck, mid_brain, mid_back, mid_tail,
left/right_ear_tip, tail_base, tail_end) exist in all 10 files. **All 10 fail the quality check.**

```
session                   tracked  segment CV  vs travel  verdict
Rat5_20260623.nwb           92.6%        0.84       0.43  UNUSABLE
Rat6_20260625.nwb           94.9%        0.74       0.22  UNUSABLE
...                                                       (10 of 10)
```

Two tests, both failed:

* **Rigidity.** The nose→neck segment sits on a rigid body, so its length should barely vary. Its
  coefficient of variation is **0.74–1.00**; a well-tracked segment is < 0.1. Ear-to-ear separation
  varies by 2×.
* **Agreement with travel.** While running, the animal's body cannot point sideways. Circular
  concordance with direction of travel is **MVL 0.22–0.43**; it should exceed 0.8.

### The clinching evidence

The head axis, the **body** axis (tail_base→neck) and the **tail** axis all score the *same* ~0.48:

```
head-direction definition     MVL   offset   <45deg
neck      -> nose           0.477    +9.9d    63.7%
tail_base -> neck           0.496    -1.7d    65.4%     <- BODY axis
mid_tail  -> mid_back       0.489    -1.4d    63.8%     <- TAIL axis
```

These are anatomically different axes and should score differently. Identical scores mean every
keypoint is **noise about a common centre**, so every pairwise difference inherits the same noise.

### What is *not* wrong

* **`mid_brain` being identically (0,0) is by design, not a bug.** `create_nwb.py` states it:
  `reference_frame="(0,0) is mid_brain (head-centered)"`. Every part is stored as an offset from
  `mid_brain`, which is therefore the origin. This also explains the negative coordinates and why
  they don't correlate with `Position/Rat`.
* The frame is only **translated**, not rotated (the raw nose→neck angle has MVL 0.078 and spans all
  compass sectors), so head direction *is* recoverable in principle. Translation also preserves
  distances — which is exactly why the rigidity failure cannot be blamed on the centering.

### Root cause and fix

**`create_nwb.py` copies only the `_x`/`_y` columns and drops DeepLabCut's `likelihood`.** Nothing
ever filters low-confidence predictions. The raw DLC `.h5`/`.csv` beside the videos *does* carry it.

> **Fix:** threshold on likelihood (~0.6–0.9) and NaN the rest before writing `DLC_Position`, or
> carry the likelihood column into the NWB so it can be filtered at analysis time.

Until then, keep `--head-direction travel` (the default), or `auto`.

**It changes no result.** Head direction feeds only `head_centred_direction` → alternation and the
figures; it never touches cycle detection, rate maps, decoding, or sweep detection. The 10-session
DLC run matches the travel run exactly.

---

## 3. `node_list_new.csv` is incomplete — **worked around**

The node list holds the hexagon vertices but **not the long corridors joining one cluster to the
next**. A ~183 cm corridor that the animal demonstrably runs down has no nodes along it at all, so
any corridor graph built from node spacing misses it.

This is a live trap: an early version of this code projected decoded positions onto a corridor graph
built from the node list, which would have snapped the animal's *genuine* positions in that corridor
onto a hexagon 42 cm away.

Worse, inferring the graph by "which straight lines did the animal walk" doesn't work either — in a
hexagonal lattice the vertex-to-next-nearest-vertex **chord is √3 · s = 1.73 segments**, and its
midpoint sits only ~20 cm from the real corridor, so it passes any reasonable coverage test. You end
up "discovering" straight lines through hexagon interiors.

**Resolution:** the analysis no longer depends on the node list. Where the animal has been *cannot*
be wrong — it can only have been somewhere it can go. `maze_corridors()` survives **for drawing
only** and says so in its docstring.

---

## 4. Camera 8 looked broken. It wasn't.

`Behavior/Metrics/region_id` is the camera ID (10 cameras tiling the maze in a 5 × 2 grid; the
coordinates are already stitched into one global frame).

Camera 8's frames sit a median of **52–64 cm off any corridor, in all 10 sessions** — reproducibly,
which smells like a calibration failure. It is not. Camera 8's region contains one of the **long
corridors missing from the node list** (§3). The tracking is clean; the *map* was wrong.

Two things were confirmed in the process, and both are good news:

* **Camera registration is fine.** Inter-tile scale variation is ±2% (worst tile +6%). The ±13%
  spread in apparent corridor length is scatter *within* tiles — noise in the hand-placed node list,
  not camera perspective.
* **The maze geometry is stable across sessions.** Every session's tracked positions overlay
  session 1's to within **0–0.8 cm median** (90th pct ≤ 3.2 cm).

So there is no camera-instability problem to fix.
