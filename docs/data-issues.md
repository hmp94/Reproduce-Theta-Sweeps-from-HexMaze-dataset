# Defects in the HexMaze dataset

Four problems found while porting. Two are corrected in code; two are not.

---

## 1. The LFP sampling rate is wrong in 6 of 10 files — **corrected**

These NWBs store timing as a `.timestamps` array, not a scalar `.rate`, so the sampling rate is never
written down — it exists only as the spacing between stamps. Six files stamp the **downsampled 1 kHz
LFP with the 30 kHz acquisition rate**:

```
session                   n_samples  stamped fs  stamps span     ephys   true fs
Rat5_20260623.nwb           4177446        1000      4177.4s   4177.4s      1000   ok
Rat5_20260625.nwb           3945522       30000       131.5s   3945.5s      1000   BAD
Rat6_20260626.nwb          18754618       30000       625.2s   3655.6s      1000   BAD
                                                                       (6 of 10 BAD)
```

**Why it is dangerous.** `np.interp` does not extrapolate — past its last known x it repeats the last
y. Theta phase is real for ~131 s and then **frozen flat** for the remaining ~3,800 s: **~1,050
cycles instead of ~32,000**. Nothing errors, nothing warns. And because the four files with *correct*
timestamps happen to be the 23rd and 24th, it presents as a clean **"sweeps only occur on the 23rd"
day effect**. It is pure metadata corruption.

**The fix.** `infer_lfp_rate_hz` resolves the rate in four tiers: a caller-pinned rate → the stamps,
if they span ≥90% of the recording → `n_samples / last_spike_time` snapped to a standard rate → the
**spectrum** (keep the candidate rate whose 4–12 Hz peak lands inside theta).

Tier 4 exists for **Rat6_20260626**, a second anomaly: its LFP holds 18.7M samples ≈ 5.2 h (the whole
day, including sleep) while the units span 61 min, so `n_samples/duration` gives a nonsense 5130 Hz.
Only the spectrum disambiguates — at 1 kHz the peak sits at 8.50 Hz; at 5130 and 30000 Hz it sits at
the band edge, a filtering artefact.

Two guards: `_theta_phase_from_lfp` **raises** if the LFP covers <95% of the session rather than
clamping; and the rate is inferred from the **last spike time**, not the tracking duration — the LFP
is electrophysiology, so its length reflects the electrical recording, not when the camera stopped.

> **Tell the data provider:** the NWB writer is stamping the downsampled LFP with the raw acquisition
> rate. Better still, write `.rate` as a scalar.

After the fix all 10 sessions give 26k–35k cycles, and the day effect disappears.

---

## 2. The DLC keypoints are unusable — **flagged, not fixed**

All 10 files carry `DLC_Position` (nose, neck, mid_brain, ears, tail…). **All 10 fail the quality
check** (`check_head_direction`), on both tests:

* **Rigidity** — the nose→neck segment sits on a rigid body, so its length should barely vary. Its
  coefficient of variation is **0.74–1.00**; a well-tracked segment is < 0.1. Ear-to-ear separation
  varies 2×.
* **Agreement with travel** — a running animal's body cannot point sideways. Circular concordance
  with direction of travel is **MVL 0.22–0.43**; it should exceed 0.8.

**The clinching evidence:** the head axis, the **body** axis (tail_base→neck) and the **tail** axis
all score the *same* ~0.48 against travel. These are anatomically different axes and should differ.
Identical scores mean every keypoint is **noise about a common centre**.

**What is *not* wrong.** `mid_brain` being identically (0,0) is **by design** — `create_nwb.py` says
`reference_frame="(0,0) is mid_brain (head-centered)"`, so it is the origin and every part is stored
as an offset from it. That also explains the negative coordinates. The frame is only *translated*,
not rotated (raw nose→neck angle: MVL 0.078, spanning all compass sectors), so head direction is
recoverable in principle — and translation preserves distances, which is exactly why the rigidity
failure cannot be blamed on the centering.

**Root cause.** `create_nwb.py` copies only the `_x`/`_y` columns and **drops DeepLabCut's
`likelihood`**, so nothing filters low-confidence predictions. The raw DLC `.h5` beside the videos
does carry it.

> **Fix:** threshold on likelihood (~0.6–0.9) and NaN the rest before writing `DLC_Position`, or
> carry the likelihood column into the NWB.

**It changes no result.** Head direction feeds only `head_centred_direction` → alternation and the
figures; it never touches cycle detection, rate maps, decoding, or sweep detection. The 10-session
DLC run is identical to the travel run.

---

## 3. `node_list_new.csv` is incomplete — **worked around**

The node list holds the hexagon vertices but **not the long corridors joining one cluster to the
next**. A ~183 cm corridor the animal demonstrably runs down has no nodes along it at all.

This is a live trap. An early version of this code projected decoded positions onto a corridor graph
built from the node list — which would have snapped the animal's *genuine* positions in that corridor
onto a hexagon 42 cm away.

Inferring the graph from "which straight lines did the animal walk" doesn't work either: in a
hexagonal lattice the vertex-to-next-nearest-vertex **chord is √3·s = 1.73 segments**, and its
midpoint sits only ~20 cm from the real corridor, so it passes any reasonable coverage test. You end
up "discovering" straight lines through hexagon interiors.

**Resolution:** the analysis no longer depends on the node list. `maze_corridors()` survives **for
drawing only**, and says so in its docstring.

---

## 4. Camera 8 looked broken. It wasn't.

`Behavior/Metrics/region_id` is the camera ID — 10 cameras tiling the maze in a 5 × 2 grid, already
stitched into one global frame.

Camera 8's frames sit a median of **52–64 cm off any corridor, in all 10 sessions** — reproducibly,
which smells like a calibration failure. It is not: camera 8's region contains one of the **long
corridors missing from the node list** (§3). The tracking is clean; the *map* was wrong.

Two things were confirmed in the process, both good news:

* **Camera registration is fine** — inter-tile scale variation is ±2% (worst tile +6%). The ±13%
  spread in apparent corridor length is scatter *within* tiles: noise in the hand-placed node list,
  not camera perspective.
* **The maze geometry is stable across sessions** — every session's tracked positions overlay
  session 1's to within **0–0.8 cm median**.

There is no camera-instability problem to fix.
