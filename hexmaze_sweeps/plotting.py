"""One figure, in the style of the paper's Fig. 4a/d.

Layers, from the back: the maze; the whole session's trajectory, faint; the animal's path
through the window, as a black arrow; then, for each cycle that PASSES the sweep criteria,
its decoded positions as soft purple blobs -- light where the sweep begins, deepening to
violet at its far end, so the outward progression reads from colour alone. The animal's own
position is a dark dot, so every sweep is seen relative to where the animal actually was.

Only accepted sweeps are drawn. Pass show_rejected=True to overlay the failed cycles in
faint grey for diagnostics.
"""
from __future__ import annotations

import numpy as np

from .config import Config, SEGMENT_CM
from .data import Result
from .sweeps import maze_corridors


# =============================================================================
# Plotting -- one figure, in the style of the paper's Fig. 4a
# =============================================================================

# Fig. 4d's look: each decoded position is a soft purple blob, light near the animal (where
# the sweep begins) and deepening to violet at the far end, so the sweep's direction reads
# from colour without an arrow. The animal's own position is a dark dot; the trajectory is
# grey/black.
PURPLE_NEAR = (0.76, 0.62, 0.88)                # decoded position at the sweep's start
PURPLE_FAR = (0.32, 0.06, 0.50)                 # decoded position at the sweep's far end
COLOUR_ANIMAL = (0.10, 0.10, 0.10)              # the animal's actual position
COLOUR_REJECTED = (0.62, 0.62, 0.62)            # only drawn when show_rejected=True


def _load_rat_rgba(path):
    """An outline PNG (dark lines on white) -> RGBA with the white made transparent."""
    from matplotlib import image as mpimg
    img = np.asarray(mpimg.imread(path), float)
    if img.max() > 1.0:
        img = img / 255.0
    alpha = img[..., 3] if img.shape[2] == 4 else (1.0 - img[..., :3].mean(2))
    out = np.zeros((*img.shape[:2], 4))
    out[..., :3] = 0.10                             # draw the outline near-black
    out[..., 3] = alpha
    return out


def _draw_rat(ax, x, y, heading_rad, rat_rgba, rat_cm, config):
    """Place the rat image at (x, y), sized in maze cm and rotated to its heading.

    The source image has the head pointing up; `angle` rotates that to the travel/head
    direction. Coordinates are image pixels (y grows down), so the rotation is applied in
    that same frame and the axis inversion carries it to the screen.
    """
    from matplotlib.transforms import Affine2D
    h, w = rat_rgba.shape[:2]
    half_w = config.px(rat_cm) / 2
    half_h = half_w * h / w
    image = ax.imshow(rat_rgba, extent=[x - half_w, x + half_w, y - half_h, y + half_h],
                      origin="upper", interpolation="bilinear", zorder=10)
    angle_deg = np.degrees(heading_rad) - 90        # head-up image -> heading
    image.set_transform(Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData)
    return image


def _path_with_gaps(track_x_px, track_y_px, max_step_px):
    """The tracked path, cut wherever it jumps, so a line plot cannot bridge the gap."""
    x, y = track_x_px.copy(), track_y_px.copy()
    jumped = np.r_[False, np.hypot(np.diff(x), np.diff(y)) > max_step_px]
    x[jumped] = np.nan
    y[jumped] = np.nan
    return x, y


def _busiest_window(sweeps, n_bins, window_bins):
    """The stretch of time containing the most sweeps."""
    starts = sweeps["start_bin"][sweeps["is_sweep"]]
    starts = starts[starts >= 0]

    if len(starts) == 0:
        return 0, min(window_bins, n_bins)
    if len(starts) == 1:
        centre = int(starts[0])
    else:
        _, best = max((np.sum((starts >= s) & (starts < s + window_bins)), s) for s in starts)
        centre = int(best + window_bins // 2)

    lo = max(0, centre - window_bins // 2)
    return lo, min(lo + window_bins, n_bins)


def _rejection_reason(sweeps, cycle, config) -> str:
    """Why this theta cycle's sweep was not accepted. Empty string if it was."""
    if sweeps["is_sweep"][cycle]:
        return ""
    if not sweeps["is_running"][cycle]:
        return "too slow"
    if not sweeps["starts_at_animal"][cycle]:
        return "not at the animal"
    if sweeps["n_valid_samples"][cycle] < config.min_valid_samples:
        return "too short"
    if not (sweeps["straightness"][cycle] > config.straightness_min):
        return "not straight"
    return "no sweep"


def _purple_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "sweep_purple", [PURPLE_NEAR, (0.55, 0.30, 0.75), PURPLE_FAR])


def _draw_window(ax, result, lo, hi, purple, show_rejected, rat_rgba, rat_cm, scalebar=True):
    """Draw one time-window of sweeps into `ax`. Returns (accepted, rejected, t0, t1)."""
    session, config, sweeps = result.session, result.config, result.sweeps

    # maze layout + the whole session's path, faint
    for corridor in maze_corridors(session.node_xy_px, config.px(SEGMENT_CM)):
        ax.plot(corridor[:, 0], corridor[:, 1], c="0.90", lw=2.0, zorder=0,
                solid_capstyle="round")
    ax.scatter(session.node_xy_px[:, 0], session.node_xy_px[:, 1], s=8, c="0.82", zorder=1)
    maze_x, maze_y = _path_with_gaps(session.track_x_px, session.track_y_px, config.px(SEGMENT_CM))
    ax.plot(maze_x, maze_y, c="0.80", lw=0.5, alpha=0.6, zorder=1, solid_capstyle="round")

    # the animal's path through the window, as a black arrow
    segment_x, segment_y = _path_with_gaps(session.track_x_px[lo:hi],
                                           session.track_y_px[lo:hi], config.px(SEGMENT_CM))
    ax.plot(segment_x, segment_y, c="0.20", lw=2.2, zorder=3, solid_capstyle="round")
    finite = np.where(np.isfinite(segment_x))[0]
    if len(finite) >= 2:
        a, b = finite[-2], finite[-1]
        ax.annotate("", xytext=(segment_x[a], segment_y[a]), xy=(segment_x[b], segment_y[b]),
                    arrowprops=dict(arrowstyle="-|>", color="0.20", lw=2.2,
                                    shrinkA=0, shrinkB=0, mutation_scale=18), zorder=3)

    # accepted sweeps as purple-blob trails
    cycle_onsets = sweeps["cycle_onsets"]
    in_window = np.where((cycle_onsets >= lo) & (cycle_onsets < hi))[0]
    accepted = rejected = 0
    animal_pts: list = []
    animal_headings: list = []

    for cycle in in_window:
        path = sweeps["path_xy_px"][cycle]
        if path is None or len(path) < 2 or not np.isfinite(sweeps["direction"][cycle]):
            continue
        if not bool(sweeps["is_sweep"][cycle]):
            rejected += 1
            if show_rejected:
                ax.scatter(path[:, 0], path[:, 1], s=24, color=COLOUR_REJECTED, alpha=0.35,
                           linewidths=0, zorder=4)
            continue

        accepted += 1
        shade = np.linspace(0.0, 1.0, len(path))
        ax.plot(path[:, 0], path[:, 1], color=(0.42, 0.24, 0.62), lw=1.4, alpha=0.5,
                zorder=6, solid_capstyle="round")
        ax.scatter(path[:, 0], path[:, 1], s=170, color=PURPLE_FAR, alpha=0.06,
                   linewidths=0, zorder=6)
        ax.scatter(path[:, 0], path[:, 1], s=70, c=shade, cmap=purple, vmin=0, vmax=1,
                   alpha=0.92, edgecolors="white", linewidths=0.4, zorder=7)
        true_xy = sweeps["true_xy_px"][cycle]
        if np.isfinite(true_xy).all():
            animal_pts.append(true_xy)
            animal_headings.append(sweeps["head_direction"][cycle])

    animal_pts = np.array(animal_pts) if animal_pts else np.empty((0, 2))
    if len(animal_pts) and rat_rgba is not None:
        for (x, y), heading in zip(animal_pts, animal_headings):
            _draw_rat(ax, x, y, heading, rat_rgba, rat_cm, config)
    elif len(animal_pts):
        ax.scatter(animal_pts[:, 0], animal_pts[:, 1], s=55, color=COLOUR_ANIMAL,
                   edgecolors="white", linewidths=1.0, zorder=9)

    # zoom to the window and its sweeps
    shown = [p for c in in_window
             if sweeps["is_sweep"][c] and (p := sweeps["path_xy_px"][c]) is not None and len(p)]
    stack = [np.c_[segment_x, segment_y]] + shown + ([animal_pts] if len(animal_pts) else [])
    points = np.vstack(stack)
    points = points[np.isfinite(points).all(1)]
    centre = (np.nanmin(points, 0) + np.nanmax(points, 0)) / 2
    half_px = np.nanmax(np.nanmax(points, 0) - np.nanmin(points, 0)) / 2 + config.px(25.0)
    half_px = max(half_px, config.px(60.0))
    ax.set_xlim(centre[0] - half_px, centre[0] + half_px)
    ax.set_ylim(centre[1] + half_px, centre[1] - half_px)   # inverted: pixel y grows down

    if scalebar:
        bar_x, bar_y = centre[0] - 0.92 * half_px, centre[1] + 0.9 * half_px
        ax.plot([bar_x, bar_x + config.px(50.0)], [bar_y, bar_y], c="k", lw=3.0, zorder=10)
        ax.text(bar_x + config.px(25.0), bar_y - 0.02 * half_px, "0.5 m",
                ha="center", va="bottom", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    t0, t1 = session.bin_centers_s[lo], session.bin_centers_s[hi - 1]
    return accepted, rejected, t0, t1


def _sweep_legend(ax, show_rejected):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="0.20", lw=2.4, marker=">", ms=8, label="animal path & position"),
        Line2D([], [], color=PURPLE_NEAR, lw=0, marker="o", ms=12, mec="0.7",
               label="decoded position — sweep start"),
        Line2D([], [], color=PURPLE_FAR, lw=0, marker="o", ms=12,
               label="decoded position — sweep far end"),
    ]
    if show_rejected:
        handles.append(Line2D([], [], color=COLOUR_REJECTED, lw=0, marker="o", ms=7,
                              label="rejected cycle"))
    ax.legend(handles=handles, loc="upper left", fontsize=10, frameon=False)


def plot_sweeps(result: Result, window_s: float = 3.0, start_s: float | None = None,
                show_rejected: bool = False, rat_png: str | None = None, rat_cm: float = 18.0,
                save_path: str | None = None, verbose: bool = True):
    """Accepted sweeps on the maze, in the style of the paper's Fig. 4a/d.

    Each accepted sweep is a trail of soft purple blobs, one per decoded position, shading
    from light (near the animal, where the sweep begins) to deep violet (its far end) -- so
    the sweep's outward progression reads from colour alone, no direction arrow. The animal is
    a dark dot (or, if `rat_png` is given, a rat image rotated to its heading), so every sweep
    is seen relative to where the animal actually was.

    Only cycles that PASS the sweep criteria are drawn. Pass show_rejected=True to overlay the
    failed cycles in faint grey.

    Args:
        window_s: length of the time window to show.
        start_s: when it starts. None picks the stretch holding the most sweeps.
        show_rejected: also draw the cycles that failed the criteria (default off).
        rat_png: path to a top-view outline PNG to mark the animal with, instead of a dot.
    """
    import matplotlib.pyplot as plt

    session, config, sweeps = result.session, result.config, result.sweeps
    window_bins = int(window_s / config.bin_s)
    if start_s is None:
        lo, hi = _busiest_window(sweeps, session.n_bins, window_bins)
    else:
        lo = int(np.clip(start_s / config.bin_s, 0, session.n_bins - 1))
        hi = min(lo + window_bins, session.n_bins)

    rat_rgba = _load_rat_rgba(rat_png) if rat_png else None
    figure, ax = plt.subplots(figsize=(12, 10))
    accepted, rejected, t0, t1 = _draw_window(ax, result, lo, hi, _purple_cmap(),
                                              show_rejected, rat_rgba, rat_cm)
    _sweep_legend(ax, show_rejected)

    title = f"{t0:.1f}–{t1:.1f} s     {accepted} sweep{'' if accepted == 1 else 's'}"
    if show_rejected and rejected:
        title += f"    ({rejected} rejected, faint grey)"
    ax.set_title(title, fontsize=13)

    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, dpi=150)
        if verbose:
            print(f"wrote {save_path}")
    return figure


def plot_all_sweeps(result: Result, window_s: float = 4.0, ncols: int = 5,
                    gate_origin_cm: float | None = None, sort: str = "count",
                    max_panels: int | None = 20, rat_png: str | None = None,
                    rat_cm: float = 13.0, save_path: str | None = None, verbose: bool = True):
    """Every accepted sweep, tiled across a grid of time-windows (several sweeps per frame).

    The accepted sweeps are grouped into `window_s`-second windows; each window becomes one
    panel. `sort="count"` shows the fullest windows first, `sort="time"` shows them in order.

    Args:
        gate_origin_cm: if set, only show sweeps whose near end is within this many cm of the
            animal -- the ones that genuinely depart FROM the animal (see max_sweep_origin_cm).
        max_panels: cap the grid; None shows every window.
        rat_png: mark the animal with this top-view image instead of a dot.
    """
    import matplotlib.pyplot as plt

    session, config, sweeps = result.session, result.config, result.sweeps

    # optionally restrict the display to sweeps that start at the animal
    keep = sweeps["is_sweep"].copy()
    if gate_origin_cm is not None:
        keep &= (sweeps["origin_error_px"] / config.px_per_cm) <= gate_origin_cm

    window_bins = int(window_s / config.bin_s)
    starts = np.sort(sweeps["start_bin"][keep])
    starts = starts[starts >= 0]

    # greedily group sweeps into windows: each window holds every sweep within window_bins
    windows = []
    i = 0
    while i < len(starts):
        lo = max(0, int(starts[i]) - window_bins // 6)
        hi = int(starts[i]) + window_bins
        j = i
        while j < len(starts) and starts[j] < hi:
            j += 1
        windows.append((lo, min(session.n_bins, hi), j - i))     # (lo, hi, n_sweeps)
        i = j

    total_sweeps = int(keep.sum())
    total_windows = len(windows)
    if sort == "count":
        windows.sort(key=lambda w: -w[2])
    if max_panels:
        windows = windows[:max_panels]
    n = len(windows)
    ncols = min(ncols, n) if n else 1
    nrows = int(np.ceil(n / ncols))

    saved_is_sweep = sweeps["is_sweep"]         # temporarily display only the gated sweeps
    sweeps["is_sweep"] = keep
    try:
        rat_rgba = _load_rat_rgba(rat_png) if rat_png else None
        purple = _purple_cmap()
        figure, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.0),
                                    squeeze=False)
        axes = axes.ravel()
        shown_sweeps = 0
        for k, ax in enumerate(axes):
            if k >= n:
                ax.axis("off")
                continue
            lo, hi, _ = windows[k]
            acc, _, t0, _ = _draw_window(ax, result, lo, hi, purple, False, rat_rgba, rat_cm,
                                         scalebar=(k == 0))
            shown_sweeps += acc
            ax.set_title(f"{t0:.0f} s · {acc} sweep{'' if acc == 1 else 's'}", fontsize=8.5)
    finally:
        sweeps["is_sweep"] = saved_is_sweep

    gate = f", within {gate_origin_cm:.0f} cm of the animal" if gate_origin_cm else ""
    figure.suptitle(f"{shown_sweeps} of {total_sweeps} sweeps{gate} — "
                    f"{n} of {total_windows} windows shown", fontsize=13, y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    if save_path:
        figure.savefig(save_path, dpi=140)
        if verbose:
            print(f"wrote {save_path}")
    return figure
