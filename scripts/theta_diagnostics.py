"""Diagnostics: is there theta, are spikes locked to it, is there phase precession?

Decides whether the flat GLM shift curve reflects the data or the method.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import welch

sys.path.insert(0, "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset")
from hexmaze_sweeps.config import Config
from hexmaze_sweeps.data import load_session
from hexmaze_sweeps.glm_shift import external_lfp_phase

SESSION_DIR = "/Volumes/genzel/Rat/HM/Rat_HM_Neuron/task/rat6_491391/20260716"
NWB = f"{SESSION_DIR}/Rat6_20260716.nwb"
LFP_DIR = f"{SESSION_DIR}/LFP_Output"
NODES = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset/node_list_new.csv"
OUT = "/Users/sachuriga/Desktop/code/Reproduce-Theta-Sweeps-from-HexMaze-dataset/results/glm_shift"
FS = 1500.0
CHANNEL = 23          # the channel the GLM run picked

config = Config(theta_source="lfp")
session, nwb_io = load_session(NWB, NODES, config)

xy_cm = np.column_stack([session.track_x_px, session.track_y_px]) / config.px_per_cm
speed = session.speed_px_s / config.px_per_cm
counts = session.spike_counts
moving_dir = session.head_direction
running = np.isfinite(speed) & (speed >= 15.0)
still = np.isfinite(speed) & (speed < 5.0)
print(f"running bins {running.sum()}, still bins {still.sum()}")

# --- raw LFP phase (not re-zeroed) at bin centres ----------------------------
phase = external_lfp_phase(LFP_DIR, FS, CHANNEL, session, config)

# --- 1. theta power during running vs immobility -----------------------------
import glob
lfp = np.load(glob.glob(f"{LFP_DIR}/*_lfp_data.npy")[0], mmap_mode="r")
bin_of_sample = None

def psd_for(mask, n_seg=40, seg_s=4.0):
    """Average Welch PSD over `n_seg` random LFP segments whose bin-mask is true."""
    idx = np.where(mask)[0]
    rng = np.random.default_rng(1)
    seg = int(seg_s * FS)
    psds = []
    for k in rng.choice(len(idx), size=min(n_seg * 4, len(idx)), replace=False):
        t = session.bin_centers_s[idx[k]]
        s0 = int(t * FS)
        if s0 + seg >= lfp.shape[0]:
            continue
        block_bins = idx[(session.bin_centers_s[idx] >= t) & (session.bin_centers_s[idx] < t + seg_s)]
        if len(block_bins) < seg_s / config.bin_s * 0.9:   # segment must stay in-state
            continue
        sig = np.asarray(lfp[s0:s0 + seg, CHANNEL], float)
        f, p = welch(sig, fs=FS, nperseg=seg)
        psds.append(p)
        if len(psds) >= n_seg:
            break
    return f, np.mean(psds, 0)

f_run, p_run = psd_for(running)
f_still, p_still = psd_for(still)

def band_ratio(f, p):
    th = p[(f >= 6) & (f <= 10)].mean()
    de = p[(f >= 2) & (f <= 4)].mean()
    return th / de

print(f"theta/delta running {band_ratio(f_run, p_run):.2f}  vs immobile {band_ratio(f_still, p_still):.2f}")

# --- 2. spike phase locking (validates spike<->LFP alignment too) ------------
n_ph = 24
ph_edges = np.linspace(-np.pi, np.pi, n_ph + 1)
ph_centers = np.degrees(ph_edges[:-1] + np.pi / n_ph)
pop = counts[running].sum(1)
ph_run = phase[running]
occ_per_bin = np.histogram(ph_run, ph_edges)[0] * config.bin_s
pop_hist = np.histogram(ph_run, ph_edges, weights=pop)[0] / occ_per_bin

# per-cell mean resultant length
mrl = np.empty(counts.shape[1])
pref = np.empty(counts.shape[1])
for i in range(counts.shape[1]):
    w = counts[running, i]
    total = w.sum()
    if total < 50:
        mrl[i] = np.nan; pref[i] = np.nan; continue
    vec = (w * np.exp(1j * ph_run)).sum() / total
    mrl[i] = np.abs(vec); pref[i] = np.degrees(np.angle(vec))
mod_depth = (pop_hist.max() - pop_hist.min()) / pop_hist.mean()
print(f"population phase modulation depth {mod_depth:.2f}, "
      f"median per-cell MRL {np.nanmedian(mrl):.3f} "
      f"(n = {np.sum(~np.isnan(mrl))} cells with >=50 running spikes)")

# --- 3. phase precession, top spatial cells ----------------------------------
npz = np.load(f"{OUT}/Rat6_20260716_extlfp_shift_model.npz", allow_pickle=True)
bits = npz["bits_per_spike"]; unit_ids = npz["unit_ids"]
unit_rows = [int(np.where(session.unit_ids == u)[0][0]) for u in unit_ids]
top = np.argsort(bits)[::-1][:10]

# raw rate map to find each field peak
bs = 2.5
gx = np.arange(xy_cm[running, 0].min(), xy_cm[running, 0].max() + bs, bs)
gy = np.arange(xy_cm[running, 1].min(), xy_cm[running, 1].max() + bs, bs)
occ, _, _ = np.histogram2d(xy_cm[running, 0], xy_cm[running, 1], [gx, gy])
occ_s = gaussian_filter(occ * config.bin_s, 2.0)

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
slopes = []
for ax, t in zip(axes.ravel(), top):
    row = unit_rows[t]
    w = counts[running, row]
    spk, _, _ = np.histogram2d(xy_cm[running, 0], xy_cm[running, 1], [gx, gy], weights=w)
    rate = gaussian_filter(spk, 2.0) / np.maximum(occ_s, 0.25)
    rate[occ_s < 0.25] = 0
    pi, pj = np.unravel_index(np.argmax(rate), rate.shape)
    peak = np.array([gx[pi] + bs / 2, gy[pj] + bs / 2])

    near = running & (np.hypot(*(xy_cm - peak).T) <= 15.0)
    s = ((xy_cm[near] - peak) * np.column_stack([np.cos(moving_dir[near]),
                                                 np.sin(moving_dir[near])])).sum(1)
    wn = counts[near, row]
    keep = wn > 0
    s_sp = np.repeat(s[keep], wn[keep].astype(int))
    ph_sp = np.repeat(phase[near][keep], wn[keep].astype(int))

    # circular-linear fit: maximize resultant over slope
    cand = np.radians(np.linspace(-30, 30, 241))          # deg per cm -> rad per cm
    R = [np.abs(np.exp(1j * (ph_sp - a * s_sp)).mean()) for a in cand]
    a_best = cand[int(np.argmax(R))]
    slopes.append(np.degrees(a_best))

    ax.plot(np.r_[s_sp, s_sp], np.r_[np.degrees(ph_sp), np.degrees(ph_sp) + 360],
            ".", ms=2, alpha=0.4)
    xx = np.array([-15, 15])
    phi0 = np.angle(np.exp(1j * (ph_sp - a_best * s_sp)).mean())
    ax.plot(xx, np.degrees(phi0 + a_best * xx), "r-", lw=2)
    ax.set(title=f"u{unit_ids[t]} {bits[t]:.1f}b/sp  {np.degrees(a_best):+.1f} deg/cm "
                 f"({len(s_sp)} spk)", xlabel="dist along moving dir (cm)",
           ylabel="theta phase (deg)", ylim=(-180, 540))

print("precession slopes (deg/cm):", np.round(slopes, 1))
fig.suptitle("phase precession check, top spatial cells (raw LFP phase, ch 23)")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{OUT}/Rat6_20260716_precession_check.png", dpi=130)

# population phase histogram figure
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.semilogy(f_run, p_run, label="running >=15 cm/s")
a1.semilogy(f_still, p_still, label="immobile <5 cm/s")
a1.set(xlim=(0, 30), xlabel="Hz", ylabel="PSD", title=f"LFP ch {CHANNEL}")
a1.legend(frameon=False)
a2.bar(ph_centers, pop_hist, width=360 / n_ph)
a2.set(xlabel="LFP theta phase (deg)", ylabel="population rate (Hz)",
       title=f"population locking, depth {mod_depth:.2f}")
fig2.tight_layout()
fig2.savefig(f"{OUT}/Rat6_20260716_theta_locking.png", dpi=130)
print("wrote precession_check.png and theta_locking.png")
nwb_io.close()
