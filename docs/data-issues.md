# Defects in the HexMaze data

Four things I ran into while porting. Two I fixed in code, two I only flagged.

## 1. The LFP sampling rate is wrong in 6 of 10 files — fixed

These NWBs store timing as a `.timestamps` array instead of a scalar `.rate`, and six of them stamp
the downsampled 1 kHz LFP with the 30 kHz acquisition rate. So `1/diff(timestamps)` reads 30× too fast
and the stamps only cover a fraction of the session.

This one bit me. `np.interp` doesn't extrapolate, so the theta phase stays real for the first ~131 s
and then just sits frozen for the remaining ~3,800 s, which quietly turns ~32,000 cycles into ~1,050.
Nothing errors along the way. And because the four files with correct timestamps happen to be the 23rd
and 24th, at first it looked like a real "sweeps only on the 23rd" day effect, when really it was only
the metadata.

I resolved it in `infer_lfp_rate_hz` by trying a few things in order. First a rate the caller pinned,
if there is one. Otherwise the timestamps, as long as they span at least 90% of the recording. Failing
that, `n_samples / last_spike_time` snapped to the nearest standard rate. And if even that is
ambiguous, I fall back to the spectrum and keep whichever candidate rate puts a 4–12 Hz peak inside
theta. That last step is really there for `Rat6_20260626`, whose LFP runs 5.2 h — the whole day, sleep
included — while its units span only 61 min, so the length estimate is nonsense. The exact thresholds
are in the code. I also made `_theta_phase_from_lfp` raise when the LFP covers under 95% of the session
rather than clamping, and I take the rate from the last spike time rather than the tracking duration.
After all that, the 10 sessions give 26k–35k cycles.

One thing worth passing back to whoever writes the NWBs. They're stamping the downsampled LFP at the
raw acquisition rate, and the cleanest fix on their end is just to write `.rate` as a scalar.

## 2. The DLC keypoints are unusable — flagged, not fixed

All 10 files carry `DLC_Position`, but I couldn't trust any of it. The body segment that should stay
rigid has a length CV of 0.74–1.00, where a real one is under 0.1, and the keypoints agree with the
direction of travel at only MVL 0.22–0.43 when it should be over 0.8. What finally convinced me was
that the head, body and tail axes all score the same ~0.48 against travel, so every keypoint is really
just noise about a common centre. The cause is in `create_nwb.py`, which drops DeepLabCut's
`likelihood` column, so nothing filters out the low-confidence frames. None of this changes a result
here, since heading only feeds the alternation and the figures and never the decoding, and the DLC run
comes out identical to the travel run.

To fix it properly you would threshold on likelihood before writing `DLC_Position`, or carry that
column into the NWB.

## 3. `node_list_new.csv` is incomplete — worked around

The node list has the hexagon vertices but not the long corridors between clusters, so a ~183 cm
corridor the animal clearly runs has no nodes on it at all. That matters because snapping decoded
positions to this graph would drag the animal's real positions onto a hexagon 42 cm away. So I stopped
using the node list for anything but drawing, and `maze_corridors()` is now only for the figures.

## 4. Camera 8 looked broken — it wasn't

Camera 8's frames sit 52–64 cm off any corridor in all 10 sessions, which looked like a calibration
fault. It isn't. That camera just happens to cover one of the corridors missing from the node list
(§3), so the tracking is fine and the map was wrong. While chasing it I confirmed two reassuring
things, that camera registration is within ±2% and that the maze geometry is stable across sessions,
each one overlaying the first to within 0–0.8 cm.
