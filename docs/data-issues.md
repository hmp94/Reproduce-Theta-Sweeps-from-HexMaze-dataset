# Data defects

Four issues found while porting. Two fixed in code, two flagged.

## 1. LFP sampling rate wrong in 6/10 files — fixed

Six NWBs stamp the 1 kHz LFP with the 30 kHz acquisition rate (timing as `.timestamps`, not scalar
`.rate`). `np.interp` then clamps, freezing theta phase after ~131 s and turning ~32,000 cycles into
~1,050 — silently, and it first looked like a "sweeps only on the 23rd" day effect.

`infer_lfp_rate_hz` resolves it in order: a pinned rate; else timestamps if they span ≥ 90% of the
recording; else `n_samples / last_spike_time` snapped to a standard rate; else the rate whose spectrum
puts a 4–12 Hz peak in theta (needed for `Rat6_20260626`, LFP 5.2 h vs units 61 min). `_theta_phase_from_lfp`
now raises below 95% coverage instead of clamping. All 10 sessions then give 26k–35k cycles.

*Writer-side fix: store `.rate` as a scalar.*

## 2. DLC keypoints unusable — flagged

All 10 files carry `DLC_Position`, but the rigid body segment has length CV 0.74–1.00 (real < 0.1) and
agrees with travel at only MVL 0.22–0.43 (should be > 0.8); head/body/tail axes all score ~0.48, i.e.
noise about a common centre. Cause: `create_nwb.py` drops DeepLabCut's `likelihood` column, so
low-confidence frames aren't filtered. No effect on results — heading feeds only alternation and figures,
never decoding, and the DLC run equals the travel run.

*Fix: threshold on likelihood before writing, or carry the column into the NWB.*

## 3. `node_list_new.csv` incomplete — worked around

Has the hexagon vertices but not the long inter-cluster corridors (a ~183 cm corridor has no nodes).
Snapping decoded positions to this graph would drag them 42 cm onto a hexagon, so the node list is now
used only for drawing (`maze_corridors()`).

## 4. Camera 8 looked broken — it wasn't

Its frames sit 52–64 cm off any corridor, but only because it covers a corridor missing from the node
list (§3); the tracking is fine. Camera registration is within ±2% and maze geometry is stable across
sessions (0–0.8 cm overlay).
