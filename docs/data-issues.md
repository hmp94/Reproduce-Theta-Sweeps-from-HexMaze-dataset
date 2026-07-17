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

