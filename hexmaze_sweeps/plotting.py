"""One figure, in the style of the paper's Fig. 4a/d.

Layers, from the back: the maze; the whole session's trajectory, faint; the segment the
animal ran, as a black arrow; then, per theta cycle, the decoded positions making up its
sweep as circles running cyan to magenta with time, and a green arrow for the direction --
drawn from the ANCHOR, because that is what the direction is measured against.

Cycles that failed the acceptance criteria are drawn faint grey and labelled with why,
rather than hidden. A figure that only shows what survived cannot tell you whether the
criteria chose sensibly.
"""
from __future__ import annotations

import numpy as np

from .config import Config, SEGMENT_CM
from .data import Result
from .sweeps import maze_corridors


# =============================================================================
# Plotting -- one figure, in the style of the paper's Fig. 4a
# =============================================================================

# Fig. 4d's palette: the decoded positions of a sweep run cyan -> magenta with sweep time,
# the decoded direction is a green arrow, the trajectory is grey/black.
SWEEP_COLOURMAP = "cool"                        # cyan -> magenta
COLOUR_DIRECTION = (0.180, 0.545, 0.341)        # green, the decoded direction
COLOUR_REJECTED = (0.62, 0.62, 0.62)            # a cycle whose sweep failed the criteria
COLOUR_START = (0.800, 0.150, 0.150)            # the red dot where a sweep begins


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


def plot_sweeps(result: Result, window_s: float = 3.0, start_s: float | None = None,
                show_rejected: bool = True, save_path: str | None = None,
                verbose: bool = True):
    """Sweeps on the maze, in the style of the paper's Fig. 4a/d.

    Layers, from the back: the maze itself, drawn faint; the whole session's trajectory,
    fainter still; then the segment the animal ran, as a black arrow; then, for each theta
    cycle, the decoded positions that make up its sweep, as circles running cyan to magenta
    with time through the sweep, a red dot where the sweep begins, and a green arrow for the
    direction it ended up pointing.

    Cycles whose sweep failed the acceptance criteria are drawn faint grey rather than
    hidden, so the figure shows what was kept AND what was thrown away.

    Args:
        window_s: length of the segment to show.
        start_s: when it starts. None picks the stretch holding the most sweeps.
        show_rejected: draw the rejected cycles too.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    session, config, sweeps = result.session, result.config, result.sweeps

    # --- pick the segment -------------------------------------------------------
    window_bins = int(window_s / config.bin_s)
    if start_s is None:
        lo, hi = _busiest_window(sweeps, session.n_bins, window_bins)
    else:
        lo = int(np.clip(start_s / config.bin_s, 0, session.n_bins - 1))
        hi = min(lo + window_bins, session.n_bins)

    figure, ax = plt.subplots(figsize=(12, 10))

    # --- the maze layout, and the whole session's path through it ----------------
    for corridor in maze_corridors(session.node_xy_px, config.px(SEGMENT_CM)):
        ax.plot(corridor[:, 0], corridor[:, 1], c="0.90", lw=2.5, zorder=0,
                solid_capstyle="round")
    ax.scatter(session.node_xy_px[:, 0], session.node_xy_px[:, 1], s=14, c="0.82", zorder=1)

    maze_x, maze_y = _path_with_gaps(session.track_x_px, session.track_y_px,
                                     max_step_px=config.px(SEGMENT_CM))
    ax.plot(maze_x, maze_y, c="0.80", lw=0.5, alpha=0.7, zorder=1, solid_capstyle="round")

    # --- the segment the animal ran, as a black arrow ----------------------------
    segment_x, segment_y = _path_with_gaps(session.track_x_px[lo:hi],
                                           session.track_y_px[lo:hi],
                                           max_step_px=config.px(SEGMENT_CM))
    ax.plot(segment_x, segment_y, c="0.20", lw=2.4, zorder=3, solid_capstyle="round")

    finite = np.where(np.isfinite(segment_x))[0]
    if len(finite) >= 2:                    # an arrowhead on the last step, showing which way
        a, b = finite[-2], finite[-1]
        ax.annotate("", xytext=(segment_x[a], segment_y[a]), xy=(segment_x[b], segment_y[b]),
                    arrowprops=dict(arrowstyle="-|>", color="0.20", lw=2.4,
                                    shrinkA=0, shrinkB=0, mutation_scale=22), zorder=3)

    # --- one sweep per theta cycle ------------------------------------------------
    cycle_onsets = sweeps["cycle_onsets"]
    in_window = np.where((cycle_onsets >= lo) & (cycle_onsets < hi))[0]

    accepted = rejected = undecoded = 0
    reasons: dict[str, int] = {}

    for cycle in in_window:
        path = sweeps["path_xy_px"][cycle]
        if path is None or len(path) < 2 or not np.isfinite(sweeps["direction"][cycle]):
            undecoded += 1                  # the decoder produced nothing for this cycle
            continue

        is_sweep = bool(sweeps["is_sweep"][cycle])
        if not is_sweep:
            rejected += 1
            reason = _rejection_reason(sweeps, cycle, config)
            reasons[reason] = reasons.get(reason, 0) + 1
            if not show_rejected:
                continue

            ax.plot(path[:, 0], path[:, 1], c=COLOUR_REJECTED, lw=1.0, alpha=0.5, zorder=4)
            ax.scatter(path[:, 0], path[:, 1], s=26, color=COLOUR_REJECTED, alpha=0.45,
                       linewidths=0, zorder=4)
            ax.annotate(reason, xy=path[-1], xytext=(7, 7), textcoords="offset points",
                        fontsize=8, color="0.45", zorder=4)
            continue

        accepted += 1

        # The sweep itself: one circle per decoded position, cyan at its start and magenta
        # at its far end, so the order the positions were visited in is readable. No joining
        # line -- a grey one would read as a rejected cycle.
        ax.scatter(path[:, 0], path[:, 1], s=95, c=np.linspace(0, 1, len(path)),
                   cmap=SWEEP_COLOURMAP, vmin=0, vmax=1, edgecolors="0.35", linewidths=0.6,
                   alpha=0.95, zorder=6)
        ax.plot(path[0, 0], path[0, 1], "o", ms=5, c=COLOUR_START, zorder=8)

        # Sweep direction is measured from the ANCHOR -- the lowpass decoded trajectory --
        # not from the first decoded position. The arrow has to start there, or it points
        # somewhere it does not mean. Drawing the anchor too shows when it has drifted off
        # the animal, which is exactly when the direction stops meaning anything.
        anchor = result.lowpass_xy_px[sweeps["stop_bin"][cycle]]
        if np.isfinite(anchor).all():
            ax.plot(anchor[0], anchor[1], "x", ms=10, mew=2.2, c=COLOUR_DIRECTION, zorder=8)
            ax.annotate("", xytext=anchor, xy=path[-1],
                        arrowprops=dict(arrowstyle="-|>", color=COLOUR_DIRECTION, lw=2.2,
                                        shrinkA=0, shrinkB=0, mutation_scale=17), zorder=7)

    # --- zoom to the segment, its sweeps, and their anchors ------------------------
    shown = [p for c in in_window if (p := sweeps["path_xy_px"][c]) is not None and len(p)]
    anchors = result.lowpass_xy_px[[sweeps["stop_bin"][c] for c in in_window
                                    if sweeps["is_sweep"][c] and sweeps["stop_bin"][c] >= 0]]
    points = np.vstack([np.c_[segment_x, segment_y], anchors] + shown)
    points = points[np.isfinite(points).all(1)]

    centre = (np.nanmin(points, 0) + np.nanmax(points, 0)) / 2
    half_px = np.nanmax(np.nanmax(points, 0) - np.nanmin(points, 0)) / 2 + config.px(35.0)
    half_px = max(half_px, config.px(90.0))     # a segment with no sweeps must not collapse
    ax.set_xlim(centre[0] - half_px, centre[0] + half_px)
    ax.set_ylim(centre[1] + half_px, centre[1] - half_px)   # inverted: pixel y grows down

    bar_cm = 50.0
    bar_x, bar_y = centre[0] - 0.92 * half_px, centre[1] + 0.88 * half_px
    ax.plot([bar_x, bar_x + config.px(bar_cm)], [bar_y, bar_y], c="k", lw=3.5, zorder=9)
    ax.text(bar_x + config.px(bar_cm) / 2, bar_y - 0.02 * half_px, f"{bar_cm / 100:.1f} m",
            ha="center", va="bottom", fontsize=11)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    colours = plt.get_cmap(SWEEP_COLOURMAP)
    handles = [
        Line2D([], [], color="0.20", lw=2.4, marker=">", ms=8, label="trajectory"),
        Line2D([], [], color=colours(0.0), lw=0, marker="o", ms=9, mec="0.35",
               label="sweep: start"),
        Line2D([], [], color=colours(1.0), lw=0, marker="o", ms=9, mec="0.35",
               label="sweep: end"),
        Line2D([], [], color=COLOUR_DIRECTION, lw=2.2, marker=">", ms=8,
               label="sweep direction (from the anchor)"),
        Line2D([], [], color=COLOUR_DIRECTION, lw=0, marker="x", ms=9, mew=2.2,
               label="anchor (lowpass decoded position)"),
    ]
    if show_rejected:
        handles.append(Line2D([], [], color=COLOUR_REJECTED, lw=1.0, marker="o", ms=6,
                              label="rejected cycle (labelled with why)"))
    ax.legend(handles=handles, loc="upper left", fontsize=10, frameon=False)

    why = ", ".join(f"{n} {r}" for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]))
    ax.set_title(
        f"{session.bin_centers_s[lo]:.1f}-{session.bin_centers_s[hi - 1]:.1f} s     "
        f"{len(in_window)} theta cycles:  {accepted} accepted,  {rejected} rejected,  "
        f"{undecoded} with no decoded sweep\n"
        f"{'rejected because: ' + why if why else ''}", fontsize=11)

    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, dpi=150)
        if verbose:
            print(f"wrote {save_path}")
    return figure
