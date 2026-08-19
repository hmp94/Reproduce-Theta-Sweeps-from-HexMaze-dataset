"""Single-cell GLM shift model (Vollan et al. 2025, Extended Data Fig. 9g,h),
with MOVING DIRECTION standing in for the LMT internal direction.

Each cell is modelled as  y ~ Poisson(exp(b0 + b.X))  with three covariate groups:

    position          2D Gaussian basis functions on a triangular lattice
                      (10 cm spacing) covering the maze plus a buffer zone,
                      so the fitted map can extrapolate beyond the corridors
    moving direction  Von Mises basis (the paper uses head direction AND the
                      LMT internal direction; this dataset's DLC head axis is
                      unreliable, so direction of travel replaces both)
    theta phase       Von Mises basis, phase re-zeroed at the population
                      firing minimum (the same convention as theta_cycles)

Each group is PCA-reduced to 99 % variance before fitting (the paper's
regularisation), then z-scored.

The sweep readout does not decode single cycles. After fitting, the position
covariate is re-evaluated at the animal's position SHIFTED by d centimetres
along its moving direction, and the Poisson log-likelihood is accumulated per
theta-phase bin. If spatial firing sweeps ahead of the animal within each
cycle, the best d moves from behind (early phase) to ahead (late phase); the
peak-to-peak of that curve estimates sweep length without any per-cycle
detection, which is why it works on sessions where few sweeps pass the gates.

Significance: the phase labels are circularly rolled in time, which keeps the
phase autocorrelation but breaks its alignment with the spikes; the observed
peak-to-peak amplitude is compared against that null.

Run:  python -m hexmaze_sweeps.glm_shift <session.nwb> --out results/glm_shift
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import Config, status, _elapsed
from .data import Session, load_session
from .decoding import _theta_phase_from_lfp, _theta_phase_from_population


# =============================================================================
# Theta phase, on the pipeline's convention (zero = population firing minimum)
# =============================================================================
def external_lfp_phase(lfp_path: str, fs: float, channel: int | None,
                       session: Session, config: Config) -> np.ndarray:
    """Theta phase per time bin from a separately exported (n_samples, n_channels)
    .npy, for sessions whose NWB-internal LFP is broken.

    `lfp_path` may be the *_lfp_data.npy itself or the directory holding it.
    Samples are assumed to start on the ephys clock's zero (Trodes export).
    """
    import glob as _glob
    from scipy.fft import next_fast_len
    from scipy.signal import butter, filtfilt, hilbert, welch

    if os.path.isdir(lfp_path):
        matches = sorted(_glob.glob(os.path.join(lfp_path, "*lfp_data.npy")))
        if not matches:
            raise FileNotFoundError(f"no *lfp_data.npy in {lfp_path}")
        lfp_path = matches[0]

    data = np.load(lfp_path, mmap_mode="r")
    n_samples, n_channels = data.shape

    # --- pick the channel with the most theta relative to delta ---------------
    if channel is None:
        probe = np.asarray(data[: int(min(400 * fs, n_samples))], np.float32)
        freq, power = welch(probe, fs=fs, nperseg=int(8 * fs), axis=0)
        theta = power[(freq >= 6) & (freq <= 10)].mean(0)
        delta = power[(freq >= 2) & (freq <= 4)].mean(0)
        channel = int(np.argmax(theta / delta))
        print(f"external LFP: {n_samples} samples x {n_channels} ch @ {fs:.0f} Hz, "
              f"picked channel {channel} (theta/delta {theta[channel] / delta[channel]:.1f})")

    # One column of a row-major mmap; chunked, so only that column stays in memory.
    signal = np.empty(n_samples, np.float64)
    step = int(fs * 600)
    for start in range(0, n_samples, step):
        signal[start:start + step] = data[start:start + step, channel]

    nyquist = fs / 2
    filter_b, filter_a = butter(2, [config.theta_band_hz[0] / nyquist,
                                    config.theta_band_hz[1] / nyquist], "band")
    filtered = filtfilt(filter_b, filter_a, signal)
    analytic = hilbert(filtered, N=next_fast_len(n_samples))[:n_samples]
    lfp_phase = np.angle(analytic)

    lfp_t_s = np.arange(n_samples) / fs
    coverage = lfp_t_s[-1] / session.bin_centers_s[-1]
    if coverage < 0.95:
        raise ValueError(f"external LFP covers only {coverage * 100:.0f}% of the session")

    phase = np.interp(session.bin_centers_s, lfp_t_s, np.unwrap(lfp_phase))
    return (phase + np.pi) % (2 * np.pi) - np.pi


def rezeroed_theta_phase(session: Session, config: Config,
                         raw_phase: np.ndarray | None = None) -> np.ndarray:
    """Theta phase per time bin in [-pi, pi), re-zeroed exactly as theta_cycles does."""
    if raw_phase is not None:
        phase = raw_phase
    elif config.theta_source == "lfp" and session.lfp_theta_channel is not None:
        phase = _theta_phase_from_lfp(session, config)
    else:
        phase = _theta_phase_from_population(session, config)

    population_spikes = session.spike_counts.sum(1)
    n_phase_bins = 60
    phase_bin = (((phase + np.pi) / (2 * np.pi)) * n_phase_bins).astype(int) % n_phase_bins
    mean_rate = np.array([
        population_spikes[phase_bin == k].mean() if np.any(phase_bin == k) else np.inf
        for k in range(n_phase_bins)])
    quietest = np.argmin(mean_rate)
    min_firing_phase = (quietest + 0.5) / n_phase_bins * 2 * np.pi - np.pi
    return (phase - min_firing_phase + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Basis functions
# =============================================================================
def triangular_lattice_cm(track_xy_cm: np.ndarray, spacing_cm: float,
                          buffer_cm: float) -> np.ndarray:
    """Triangular lattice of basis centres covering the track plus a buffer zone.

    The lattice tiles the track's bounding box (padded by `buffer_cm`), then keeps
    only centres within `buffer_cm` of somewhere the animal actually went: the maze
    is corridors, so most of the bounding box would otherwise be dead weight.
    """
    x_min, y_min = track_xy_cm.min(0) - buffer_cm
    x_max, y_max = track_xy_cm.max(0) + buffer_cm

    row_step = spacing_cm * np.sqrt(3) / 2
    centers = []
    for row, y in enumerate(np.arange(y_min, y_max + row_step, row_step)):
        x_offset = (row % 2) * spacing_cm / 2
        for x in np.arange(x_min + x_offset, x_max + spacing_cm, spacing_cm):
            centers.append((x, y))
    centers = np.asarray(centers)

    # Distance from each centre to the (subsampled) track, in manageable pieces.
    track = track_xy_cm[:: max(1, len(track_xy_cm) // 5000)]
    keep = np.zeros(len(centers), bool)
    for start in range(0, len(centers), 512):
        block = centers[start:start + 512]
        d2 = ((block[:, None, :] - track[None, :, :]) ** 2).sum(-1)
        keep[start:start + 512] = d2.min(1) <= buffer_cm ** 2
    return centers[keep]


def gaussian_position_basis(xy_cm: np.ndarray, centers_cm: np.ndarray,
                            sigma_cm: float, chunk: int = 20000) -> np.ndarray:
    """exp(-|x - c|^2 / 2 sigma^2), (n_samples, n_centers) float32."""
    out = np.empty((len(xy_cm), len(centers_cm)), np.float32)
    inv = -0.5 / sigma_cm ** 2
    for start in range(0, len(xy_cm), chunk):
        block = xy_cm[start:start + chunk]
        d2 = ((block[:, None, :] - centers_cm[None, :, :]) ** 2).sum(-1)
        out[start:start + chunk] = np.exp(inv * d2, dtype=np.float32)
    return out


def von_mises_basis(angles: np.ndarray, n_basis: int, kappa: float) -> np.ndarray:
    """Von Mises bumps at evenly spaced preferred angles, peak-normalised to 1."""
    mu = np.linspace(-np.pi, np.pi, n_basis, endpoint=False)
    return np.exp(kappa * (np.cos(angles[:, None] - mu[None, :]) - 1.0)).astype(np.float32)


class _GroupReducer:
    """PCA to a variance fraction, then z-scoring, with the combined linear map kept.

    `weights_in_basis_space(w)` folds PCA + z-scoring + the GLM weights into one
    vector per cell in RAW basis space, so a shifted design only needs the raw
    basis matrix -- no per-shift PCA transform.
    """

    def __init__(self, raw: np.ndarray, variance: float):
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=variance, svd_solver="full").fit(raw)
        scores = self.pca.transform(raw)
        self.mu = scores.mean(0)
        self.sd = scores.std(0) + 1e-12

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return ((self.pca.transform(raw) - self.mu) / self.sd).astype(np.float32)

    @property
    def n_out(self) -> int:
        return len(self.mu)

    def weights_in_basis_space(self, w: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (v, c) with  raw @ v + c == transform(raw) @ w."""
        scaled = w / self.sd
        v = self.pca.components_.T @ scaled
        c = -float(self.pca.mean_ @ v) - float(self.mu @ scaled)
        return v, c


# =============================================================================
# The model
# =============================================================================
def fit_and_shift(session: Session, config: Config, opt,
                  raw_phase: np.ndarray | None = None) -> dict:
    """Fit one GLM per cell, then trace the best position shift per theta phase."""
    px_per_cm = config.px_per_cm
    xy_cm = np.column_stack([session.track_x_px, session.track_y_px]) / px_per_cm
    speed_cm_s = session.speed_px_s / px_per_cm
    moving_dir = session.head_direction          # direction of travel (see data.py)
    counts = session.spike_counts

    status("  theta phase ...")
    phase = rezeroed_theta_phase(session, config, raw_phase)

    fit_mask = np.isfinite(speed_cm_s) & (speed_cm_s >= opt.fit_speed_cm_s)
    eval_mask = np.isfinite(speed_cm_s) & (speed_cm_s >= opt.eval_speed_cm_s)
    if fit_mask.sum() < 5000:
        raise ValueError(f"only {fit_mask.sum()} running bins; nothing to fit")

    # Everything below works on the running bins only; a full-session design
    # matrix would be several times larger for bins no model ever sees.
    rows = np.where(fit_mask | eval_mask)[0]
    fit_in = fit_mask[rows]
    eval_in = eval_mask[rows]

    # --- design matrix --------------------------------------------------------
    status("  position basis ...")
    centers_cm = triangular_lattice_cm(xy_cm[fit_mask], opt.pos_spacing_cm, opt.buffer_cm)
    basis_pos = gaussian_position_basis(xy_cm[rows], centers_cm, opt.pos_sigma_cm)
    basis_dir = von_mises_basis(moving_dir[rows], opt.n_dir_basis, opt.kappa)
    basis_phase = von_mises_basis(phase[rows], opt.n_phase_basis, opt.kappa)

    status("  PCA ...")
    reduce_pos = _GroupReducer(basis_pos[fit_in], opt.pca_variance)
    reduce_dir = _GroupReducer(basis_dir[fit_in], opt.pca_variance)
    reduce_phase = _GroupReducer(basis_phase[fit_in], opt.pca_variance)

    x_pos = reduce_pos.transform(basis_pos)
    x_dir = reduce_dir.transform(basis_dir)
    x_phase = reduce_phase.transform(basis_phase)
    design = np.concatenate([x_pos, x_dir, x_phase], 1).astype(np.float64)
    slice_pos = slice(0, reduce_pos.n_out)
    slice_dir = slice(reduce_pos.n_out, reduce_pos.n_out + reduce_dir.n_out)
    slice_phase = slice(reduce_pos.n_out + reduce_dir.n_out, design.shape[1])

    print(f"position basis: {basis_pos.shape[1]} centres -> {reduce_pos.n_out} PCs | "
          f"direction: {opt.n_dir_basis} -> {reduce_dir.n_out} | "
          f"phase: {opt.n_phase_basis} -> {reduce_phase.n_out} | "
          f"{fit_mask.sum()} fit bins, {eval_mask.sum()} eval bins")

    # --- one Poisson GLM per cell --------------------------------------------
    from sklearn.linear_model import PoissonRegressor

    n_spikes_fit = counts[fit_mask].sum(0)
    unit_rows = np.where(n_spikes_fit >= opt.min_spikes)[0]
    x_fit = design[fit_in]

    coefs, intercepts, bits_per_spike = [], [], []
    started = time.time()
    for done, unit in enumerate(unit_rows):
        status(f"  GLM fit {done + 1}/{len(unit_rows)} "
               f"(unit {session.unit_ids[unit]})   ({_elapsed(started)})")
        y = counts[fit_mask, unit].astype(np.float64)
        model = PoissonRegressor(alpha=opt.alpha, max_iter=opt.max_iter, tol=1e-4)
        model.fit(x_fit, y)
        coefs.append(model.coef_)
        intercepts.append(model.intercept_)

        # In-sample improvement over a constant-rate model, in bits per spike.
        log_rate = x_fit @ model.coef_ + model.intercept_
        rate = np.exp(log_rate)
        mean_rate = y.mean()
        ll_model = float(y @ log_rate - rate.sum())
        ll_const = float(y.sum() * np.log(mean_rate) - mean_rate * len(y))
        bits_per_spike.append((ll_model - ll_const) / max(y.sum(), 1.0) / np.log(2))
    status()

    coefs = np.asarray(coefs)
    intercepts = np.asarray(intercepts)
    bits_per_spike = np.asarray(bits_per_spike)
    included = bits_per_spike >= opt.min_bits_per_spike
    print(f"{len(unit_rows)} cells fit (>= {opt.min_spikes} spikes), "
          f"{included.sum()} spatially informative (>= {opt.min_bits_per_spike} bits/spike)")

    # --- shift the position covariate along the moving direction --------------
    shifts_cm = np.arange(opt.shift_min_cm, opt.shift_max_cm + 1e-9, opt.shift_step_cm)
    n_phase = opt.n_phase_bins

    # Fold PCA + z-score + GLM weights into raw-basis space, once per cell.
    v_pos = np.empty((len(centers_cm), len(unit_rows)))
    const = np.empty(len(unit_rows))
    for i in range(len(unit_rows)):
        v_pos[:, i], const[i] = reduce_pos.weights_in_basis_space(coefs[i, slice_pos])
    const += intercepts

    # Everything except position is fixed while d varies.
    fixed = (x_dir[eval_in] @ coefs[:, slice_dir].T
             + x_phase[eval_in] @ coefs[:, slice_phase].T + const)     # (n_eval, n_cells)

    xy_eval = xy_cm[eval_mask]
    unit_vec = np.column_stack([np.cos(moving_dir[eval_mask]), np.sin(moving_dir[eval_mask])])
    y_eval = counts[eval_mask][:, unit_rows].astype(np.float64)
    phase_eval = phase[eval_mask]
    phase_bin = ((phase_eval + np.pi) / (2 * np.pi) * n_phase).astype(int) % n_phase

    ll_cell = np.zeros((len(unit_rows), n_phase, len(shifts_cm)))
    ll_pop_t = np.zeros((eval_mask.sum(), len(shifts_cm)), np.float32)
    included_idx = np.where(included)[0]

    # Beyond the data's support the extrapolated map can explode, and a single
    # shifted-into-garbage bin's -lambda then swamps the likelihood. Cap each
    # cell's log-rate at its unshifted maximum plus log 2.
    log_cap = (gaussian_position_basis(xy_eval, centers_cm, opt.pos_sigma_cm) @ v_pos
               + fixed).max(0) + np.log(2.0)

    started = time.time()
    for si, d in enumerate(shifts_cm):
        status(f"  shift {si + 1}/{len(shifts_cm)}  d = {d:+.1f} cm   ({_elapsed(started)})")
        basis_shift = gaussian_position_basis(xy_eval + d * unit_vec, centers_cm,
                                              opt.pos_sigma_cm)
        log_rate = np.minimum(basis_shift @ v_pos + fixed, log_cap)  # (n_eval, n_cells)
        ll_t = y_eval * log_rate - np.exp(log_rate)                  # per-bin Poisson LL
        for p in range(n_phase):
            ll_cell[:, p, si] = ll_t[phase_bin == p].sum(0)
        ll_pop_t[:, si] = ll_t[:, included_idx].sum(1)
    status()

    ll_pop = np.stack([ll_pop_t[phase_bin == p].sum(0) for p in range(n_phase)])

    # --- per-bin decoded displacement -----------------------------------------
    # The summed-LL argmax below is dominated by the bins where the model already
    # fits at d = 0; a per-bin posterior over d, averaged within each phase bin,
    # weights every moment equally and is far more sensitive to a sweep.
    log_post = ll_pop_t.astype(np.float64)
    log_post -= log_post.max(1, keepdims=True)
    posterior = np.exp(log_post)
    posterior /= posterior.sum(1, keepdims=True)
    dhat_t = posterior @ shifts_cm
    has_spike = y_eval[:, included_idx].sum(1) > 0

    def displacement_curve(labels):
        return np.array([dhat_t[has_spike & (labels == p)].mean()
                         for p in range(n_phase)])

    disp_curve = displacement_curve(phase_bin)

    # --- optimal shift per phase, with sub-grid (quadratic) refinement --------
    def optimal_shift(ll_row):
        k = int(np.argmax(ll_row))
        if 0 < k < len(ll_row) - 1:
            denom = ll_row[k - 1] - 2 * ll_row[k] + ll_row[k + 1]
            if denom < 0:
                return shifts_cm[k] + 0.5 * (ll_row[k - 1] - ll_row[k + 1]) / denom \
                    * opt.shift_step_cm
        return shifts_cm[k]

    opt_shift_pop = np.array([optimal_shift(ll_pop[p]) for p in range(n_phase)])
    opt_shift_cell = np.array([[optimal_shift(ll_cell[i, p]) for p in range(n_phase)]
                               for i in range(len(unit_rows))])

    # --- null: roll the phase labels in time ----------------------------------
    rng = np.random.default_rng(0)
    null_amplitude = np.empty(opt.n_permutations)
    null_curves = np.empty((opt.n_permutations, n_phase))
    null_disp_curves = np.empty((opt.n_permutations, n_phase))
    for k in range(opt.n_permutations):
        rolled = np.roll(phase_bin, rng.integers(500, len(phase_bin) - 500))
        ll_null = np.stack([ll_pop_t[rolled == p].sum(0) for p in range(n_phase)])
        curve = np.array([optimal_shift(ll_null[p].astype(np.float64))
                          for p in range(n_phase)])
        null_curves[k] = curve
        null_amplitude[k] = curve.max() - curve.min()
        null_disp_curves[k] = displacement_curve(rolled)

    amplitude = float(opt_shift_pop.max() - opt_shift_pop.min())
    p_value = float((np.sum(null_amplitude >= amplitude) + 1) / (opt.n_permutations + 1))

    disp_amplitude = float(disp_curve.max() - disp_curve.min())
    null_disp_amplitude = null_disp_curves.max(1) - null_disp_curves.min(1)
    disp_p_value = float((np.sum(null_disp_amplitude >= disp_amplitude) + 1)
                         / (opt.n_permutations + 1))

    return dict(
        session=session, config=config, opt=opt,
        centers_cm=centers_cm, unit_rows=unit_rows,
        unit_ids=session.unit_ids[unit_rows],
        coefs=coefs, intercepts=intercepts, bits_per_spike=bits_per_spike,
        included=included, reduce_pos=reduce_pos,
        slice_pos=slice_pos, slice_dir=slice_dir, slice_phase=slice_phase,
        shifts_cm=shifts_cm,
        phase_bin_centers=(np.arange(n_phase) + 0.5) / n_phase * 360.0,
        ll_pop=ll_pop, ll_cell=ll_cell,
        opt_shift_pop=opt_shift_pop, opt_shift_cell=opt_shift_cell,
        null_curves=null_curves, null_amplitude=null_amplitude,
        amplitude_cm=amplitude, p_value=p_value,
        disp_curve=disp_curve, null_disp_curves=null_disp_curves,
        disp_amplitude_cm=disp_amplitude, disp_p_value=disp_p_value,
        n_running_eval=int(eval_mask.sum()),
        xy_cm=xy_cm, eval_mask=eval_mask, fit_mask=fit_mask,
        moving_dir=moving_dir, phase=phase,
    )


# =============================================================================
# Figures
# =============================================================================
def plot_shift_summary(res: dict, save_path: str) -> None:
    import matplotlib.pyplot as plt

    opt = res["opt"]
    shifts = res["shifts_cm"]
    phase_deg = res["phase_bin_centers"]
    included = res["included"]

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.4))
    fig.suptitle(
        f"GLM shift model, moving direction | summed-LL amplitude "
        f"{res['amplitude_cm']:.1f} cm (p = {res['p_value']:.4f}) | decoded "
        f"displacement amplitude {res['disp_amplitude_cm']:.1f} cm "
        f"(p = {res['disp_p_value']:.4f})")

    # -- A: population log-likelihood, phase x shift ---------------------------
    ax = axes[0]
    ll = res["ll_pop"] - res["ll_pop"].max(1, keepdims=True)
    im = ax.imshow(ll.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, 360, shifts[0], shifts[-1]],
                   vmin=np.percentile(ll, 5))
    ax.plot(phase_deg, res["opt_shift_pop"], "w.-", lw=2, ms=8)
    ax.axhline(0, color="w", lw=0.5, ls="--")
    ax.set(xlabel="theta phase (deg, 0 = cycle start)", ylabel="shift along moving dir (cm)",
           title="population log-likelihood")
    fig.colorbar(im, ax=ax, label="LL - max per phase")

    # -- B: the optimal-shift curve against its null ---------------------------
    ax = axes[1]
    lo, hi = np.percentile(res["null_curves"], [2.5, 97.5], axis=0)
    ax.fill_between(phase_deg, lo, hi, color="0.85", label="null 95%")
    ax.plot(phase_deg, res["opt_shift_pop"], "k.-", lw=2, ms=8, label="observed")
    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.set(xlabel="theta phase (deg)", ylabel="best shift (cm)",
           title="population best shift vs phase")
    ax.legend(frameon=False)

    # -- C: per-bin decoded displacement, averaged per phase -------------------
    ax = axes[2]
    lo, hi = np.percentile(res["null_disp_curves"], [2.5, 97.5], axis=0)
    ax.fill_between(phase_deg, lo, hi, color="0.85", label="null 95%")
    ax.plot(phase_deg, res["disp_curve"], "C3.-", lw=2, ms=8, label="observed")
    ax.axhline(res["disp_curve"].mean(), color="0.5", lw=0.5, ls="--")
    ax.set(xlabel="theta phase (deg)", ylabel="decoded displacement (cm)",
           title="per-bin posterior-mean displacement")
    ax.legend(frameon=False)

    # -- D: per-cell curves ----------------------------------------------------
    ax = axes[3]
    for i in np.where(included)[0]:
        ax.plot(phase_deg, res["opt_shift_cell"][i], color="C0", alpha=0.25, lw=1)
    ax.plot(phase_deg, res["opt_shift_cell"][included].mean(0), "C3.-", lw=2,
            label="cell mean")
    ax.plot(phase_deg, res["opt_shift_pop"], "k.-", lw=2, label="population")
    ax.axhline(0, color="0.5", lw=0.5, ls="--")
    ax.set(xlabel="theta phase (deg)", ylabel="best shift (cm)",
           title=f"per-cell curves (n = {included.sum()})")
    ax.legend(frameon=False)

    # -- E: per-cell amplitude -------------------------------------------------
    ax = axes[4]
    per_cell = (res["opt_shift_cell"][included].max(1)
                - res["opt_shift_cell"][included].min(1))
    ax.hist(per_cell, bins=15, color="C0")
    ax.axvline(res["amplitude_cm"], color="C3", lw=2, label="population")
    ax.set(xlabel="peak-to-peak shift (cm)", ylabel="cells",
           title="per-cell sweep amplitude")
    ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_example_cells(res: dict, save_path: str, n_cells: int = 8) -> None:
    """Fitted position maps on a grid that extends into the buffer zone, with the
    raw occupancy rate map beside each, so the extrapolation is visible."""
    import matplotlib.pyplot as plt

    opt, config = res["opt"], res["config"]
    xy_cm = res["xy_cm"]
    order = np.argsort(res["bits_per_spike"])[::-1][:n_cells]

    # Grid covering track + buffer.
    pad = opt.buffer_cm
    x_min, y_min = xy_cm[res["fit_mask"]].min(0) - pad
    x_max, y_max = xy_cm[res["fit_mask"]].max(0) + pad
    step = 2.0
    gx, gy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    basis_grid = gaussian_position_basis(grid, res["centers_cm"], opt.pos_sigma_cm)

    # Raw rate map for comparison (2.5 cm bins, light smoothing).
    bins_x = np.arange(x_min, x_max, 2.5)
    bins_y = np.arange(y_min, y_max, 2.5)
    run = res["fit_mask"]
    occupancy, _, _ = np.histogram2d(xy_cm[run, 0], xy_cm[run, 1], [bins_x, bins_y])
    occupancy_s = gaussian_filter(occupancy * config.bin_s, 2.0)

    fig, axes = plt.subplots(2, n_cells, figsize=(2.6 * n_cells, 5.6))
    track = xy_cm[res["fit_mask"]][::200]

    for col, i in enumerate(order):
        v, c = res["reduce_pos"].weights_in_basis_space(res["coefs"][i, res["slice_pos"]])
        log_rate = basis_grid @ v + c + res["intercepts"][i]
        rate_map = np.exp(log_rate).reshape(gx.shape) / config.bin_s   # Hz at mean of other terms

        ax = axes[0, col]
        ax.imshow(rate_map, origin="lower", extent=[x_min, x_max, y_min, y_max],
                  cmap="inferno", vmax=np.percentile(rate_map, 99.5))
        ax.plot(track[:, 0], track[:, 1], ".", color="w", ms=0.3, alpha=0.35)
        ax.set(title=f"unit {res['unit_ids'][i]}  "
                     f"{res['bits_per_spike'][i]:.2f} b/sp", xticks=[], yticks=[])

        counts_unit = res["session"].spike_counts[run, res["unit_rows"][i]]
        spikes, _, _ = np.histogram2d(xy_cm[run, 0], xy_cm[run, 1], [bins_x, bins_y],
                                      weights=counts_unit)
        raw = gaussian_filter(spikes, 2.0) / np.maximum(occupancy_s, 0.5)
        raw[occupancy_s < 0.25] = np.nan
        ax = axes[1, col]
        ax.imshow(raw.T, origin="lower", extent=[x_min, x_max, y_min, y_max],
                  cmap="inferno", vmax=np.nanpercentile(raw, 99.5))
        ax.set(xticks=[], yticks=[])
        if col == 0:
            axes[0, 0].set_ylabel("GLM map (extrapolated)")
            axes[1, 0].set_ylabel("raw rate map")

    fig.suptitle("GLM position maps extend into the buffer zone; raw maps cannot")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Command line
# =============================================================================
def _build_parser():
    parser = argparse.ArgumentParser(
        description="Single-cell GLM shift model with moving direction "
                    "(Vollan et al. ED Fig. 9g,h).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("nwb", help="NWB file")
    parser.add_argument("--nodes", default="node_list_new.csv")
    parser.add_argument("--out", default="results/glm_shift")
    parser.add_argument("--theta", default="lfp", choices=["lfp", "pca"])
    parser.add_argument("--lfp-npy", default=None, metavar="PATH",
                        help="externally exported LFP: a (n_samples, n_channels) .npy "
                             "or the directory holding *_lfp_data.npy; overrides the "
                             "NWB's own LFP and --theta")
    parser.add_argument("--lfp-npy-fs", type=float, default=1500.0)
    parser.add_argument("--lfp-channel", type=int, default=None,
                        help="channel index into the external LFP; default picks the "
                             "highest theta/delta power ratio")

    parser.add_argument("--pos-spacing-cm", type=float, default=10.0)
    parser.add_argument("--pos-sigma-cm", type=float, default=5.0,
                        help="paper: 2 cm; with 10 cm spacing that leaves near-zero "
                             "basis coverage between centres, so the default here is "
                             "half the spacing")
    parser.add_argument("--buffer-cm", type=float, default=30.0)
    parser.add_argument("--n-dir-basis", type=int, default=50)
    parser.add_argument("--n-phase-basis", type=int, default=50)
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--pca-variance", type=float, default=0.99)

    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 penalty")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--min-spikes", type=int, default=100,
                        help="skip cells with fewer running-period spikes")
    parser.add_argument("--min-bits-per-spike", type=float, default=0.05,
                        help="cells below this carry no spatial message and are "
                             "left out of the population curve")

    parser.add_argument("--fit-speed-cm-s", type=float, default=10.0)
    parser.add_argument("--eval-speed-cm-s", type=float, default=15.0)
    parser.add_argument("--n-phase-bins", type=int, default=12)
    parser.add_argument("--shift-min-cm", type=float, default=-30.0)
    parser.add_argument("--shift-max-cm", type=float, default=45.0)
    parser.add_argument("--shift-step-cm", type=float, default=2.5)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--no-cache", action="store_true",
                        help="reload from the NWB even if a local cache exists")
    return parser


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")

    opt = _build_parser().parse_args(argv)
    os.makedirs(opt.out, exist_ok=True)
    name = os.path.basename(opt.nwb).replace(".nwb", "")
    name += "_extlfp" if opt.lfp_npy else f"_{opt.theta}"

    config = Config(theta_source=opt.theta)

    # The session lives on a network volume that comes and goes; once the binned
    # arrays and theta phase are cached locally, reruns need no volume at all.
    cache_file = os.path.join(opt.out, "cache", f"{name}_session.npz")
    if os.path.exists(cache_file) and not opt.no_cache:
        z = np.load(cache_file)
        session = Session(
            bin_centers_s=z["bin_centers_s"], n_bins=len(z["bin_centers_s"]),
            track_x_px=z["track_x_px"], track_y_px=z["track_y_px"],
            speed_px_s=z["speed_px_s"], head_direction=z["head_direction"],
            spike_counts=z["spike_counts"], unit_ids=z["unit_ids"],
            lfp_theta_channel=None, lfp_rate_hz=0.0,
            node_xy_px=z["node_xy_px"], config=config)
        raw_phase = z["raw_phase"] if z["raw_phase"].size else None
        print(f"{name}: from cache ({cache_file})")
        res = fit_and_shift(session, config, opt, raw_phase)
    else:
        status("loading session ...")
        session, nwb_io = load_session(opt.nwb, opt.nodes, config)
        try:
            print(f"{name}: {session.n_units} units, {session.n_bins} bins "
                  f"({session.n_bins * config.bin_s / 60:.1f} min), "
                  f"LFP {session.lfp_rate_hz:.0f} Hz")
            raw_phase = None
            if opt.lfp_npy:
                status("external LFP ...")
                raw_phase = external_lfp_phase(opt.lfp_npy, opt.lfp_npy_fs,
                                               opt.lfp_channel, session, config)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez_compressed(
                cache_file, bin_centers_s=session.bin_centers_s,
                track_x_px=session.track_x_px, track_y_px=session.track_y_px,
                speed_px_s=session.speed_px_s, head_direction=session.head_direction,
                spike_counts=session.spike_counts, unit_ids=session.unit_ids,
                node_xy_px=session.node_xy_px,
                raw_phase=raw_phase if raw_phase is not None else np.empty(0))
            res = fit_and_shift(session, config, opt, raw_phase)
        finally:
            nwb_io.close()

    # --- report ---------------------------------------------------------------
    included = res["included"]
    per_cell = (res["opt_shift_cell"][included].max(1)
                - res["opt_shift_cell"][included].min(1))
    print(f"\npopulation best-shift curve: peak-to-peak {res['amplitude_cm']:.1f} cm, "
          f"p = {res['p_value']:.4f} vs {opt.n_permutations} phase-rolled nulls "
          f"(null 97.5th pct {np.percentile(res['null_amplitude'], 97.5):.1f} cm)")
    print(f"decoded displacement curve: peak-to-peak {res['disp_amplitude_cm']:.1f} cm, "
          f"p = {res['disp_p_value']:.4f} "
          f"(null 97.5th pct {np.percentile(res['null_disp_curves'].max(1) - res['null_disp_curves'].min(1), 97.5):.1f} cm)")
    print(f"phase of farthest-ahead shift: "
          f"{res['phase_bin_centers'][np.argmax(res['opt_shift_pop'])]:.0f} deg "
          f"(displacement readout: "
          f"{res['phase_bin_centers'][np.argmax(res['disp_curve'])]:.0f} deg)")
    print(f"per-cell amplitude (n = {included.sum()}): "
          f"median {np.median(per_cell):.1f} cm  (paper sweep length ~22.5 cm)")

    summary_png = os.path.join(opt.out, f"{name}_shift_model.png")
    cells_png = os.path.join(opt.out, f"{name}_example_cells.png")
    plot_shift_summary(res, summary_png)
    plot_example_cells(res, cells_png)

    np.savez_compressed(
        os.path.join(opt.out, f"{name}_shift_model.npz"),
        shifts_cm=res["shifts_cm"], phase_bin_centers=res["phase_bin_centers"],
        ll_pop=res["ll_pop"], ll_cell=res["ll_cell"],
        opt_shift_pop=res["opt_shift_pop"], opt_shift_cell=res["opt_shift_cell"],
        null_amplitude=res["null_amplitude"], null_curves=res["null_curves"],
        disp_curve=res["disp_curve"], null_disp_curves=res["null_disp_curves"],
        disp_amplitude_cm=res["disp_amplitude_cm"], disp_p_value=res["disp_p_value"],
        bits_per_spike=res["bits_per_spike"], included=res["included"],
        unit_ids=res["unit_ids"], amplitude_cm=res["amplitude_cm"],
        p_value=res["p_value"],
        options=json.dumps({k: v for k, v in vars(opt).items()}))
    print(f"wrote {summary_png}, {cells_png}, and the .npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
