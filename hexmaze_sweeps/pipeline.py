"""Running one session, and the command-line interface.

    load_session -> theta_cycles -> rate_maps -> decode
    -> lowpass_trajectory -> extract_sweeps -> alternation

`main` is a loop over `run` plus table formatting. Both preflights (LFP sampling rate, DLC
head direction) run first and print their own tables, so their warnings never interleave
with the results.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .config import (Config, PAPER_ALTERNATION_PCT, PAPER_ALTERNATION_SHUFFLE_PCT,
                     PAPER_PREVALENCE, PAPER_SWEEP_LENGTH_CM, SEGMENT_CM, _elapsed, status)
from .data import (HeadDirectionCheck, LfpRate, Result, check_head_direction,
                   check_lfp_rates, load_session, print_head_direction_check,
                   print_lfp_check)
from .decoding import decode, lowpass_trajectory, rate_maps, shuffle_threshold, theta_cycles
from .plotting import plot_sweeps
from .sweeps import alternation, extract_sweeps


# =============================================================================
# Running one session
# =============================================================================
def run(nwb_path, node_csv_path, config: Config | None = None, verbose=True,
        on_progress=None) -> Result:
    """Load one session, extract its sweeps, and test them for alternation.

    Args:
        on_progress: called with a short stage name as each step begins, e.g. "decoding
            60%". Use it to show a live status; see `status`.
    """
    config = config or Config()
    stage = on_progress or (lambda message: None)

    stage("loading")
    session, nwb_io = load_session(nwb_path, node_csv_path, config)

    try:
        stage("theta cycles")
        cycle_onsets = theta_cycles(session, config)

        stage("rate maps")
        tuning_curves, bin_center_x_px, bin_center_y_px = rate_maps(session, config)

        stage("shuffle threshold")
        shuffle_99 = shuffle_threshold(session, config, tuning_curves)

        decoded_xy_px, peak_correlation = decode(session, config, tuning_curves,
                                                 bin_center_x_px, bin_center_y_px,
                                                 on_progress=on_progress)

        stage("anchor")
        lowpass_xy_px = lowpass_trajectory(session, config, tuning_curves,
                                           bin_center_x_px, bin_center_y_px, cycle_onsets)

        stage("sweeps")
        sweeps = extract_sweeps(session, config, decoded_xy_px, peak_correlation,
                                lowpass_xy_px, cycle_onsets, shuffle_99)

        stage("alternation")
        observed, null_mean, null_high, n_triplets = alternation(sweeps)

        is_sweep = sweeps["is_sweep"]
        mean_length_cm = (np.nanmean(sweeps["length_px"][is_sweep]) / config.px_per_cm
                          if is_sweep.any() else np.nan)

        stats = dict(shuffle_99=shuffle_99,
                     n_cycles=len(cycle_onsets),
                     n_units=session.n_units,
                     n_sweeps=int(is_sweep.sum()),
                     n_running=int(sweeps["is_running"].sum()),
                     prevalence=sweeps["prevalence"],
                     mean_length_cm=mean_length_cm,
                     alternation=observed,
                     alternation_null=null_mean,
                     alternation_null_high=null_high,
                     n_triplets=n_triplets)

        if verbose:
            _print_session_summary(config, session, tuning_curves, stats)

        return Result(session, config, sweeps, stats, decoded_xy_px, lowpass_xy_px, cycle_onsets)
    finally:
        nwb_io.close()


def _print_session_summary(config, session, tuning_curves, stats):
    """The human-readable version of one session's `stats`."""
    print(f"px/cm {config.px_per_cm:.4f} | segment {SEGMENT_CM:.0f} cm | "
          f"{session.n_units} units | {stats['n_cycles']} theta cycles")
    print(f"shuffle threshold {stats['shuffle_99']:.4f} | "
          f"{tuning_curves.shape[0]} visited position bins")
    print(f"sweeps {stats['n_sweeps']} / {stats['n_running']} running cycles "
          f"= prevalence {stats['prevalence']:.3f}   (paper: {PAPER_PREVALENCE})")

    if stats["n_sweeps"]:
        print(f"mean sweep length {stats['mean_length_cm']:.1f} cm   "
              f"(paper: {PAPER_SWEEP_LENGTH_CM})")

    if stats["n_triplets"] == 0:
        print("alternation UNDEFINED: no three consecutive theta cycles all contain a "
              f"sweep (paper: {PAPER_ALTERNATION_PCT}% vs {PAPER_ALTERNATION_SHUFFLE_PCT}% shuffle)")
    else:
        print(f"alternation {stats['alternation'] * 100:.1f}% vs shuffle "
              f"{stats['alternation_null'] * 100:.1f}% "
              f"(99.9th pct {stats['alternation_null_high'] * 100:.1f}%), "
              f"{stats['n_triplets']} triplets")


# =============================================================================
# Command-line interface
# =============================================================================

# Decoded bins are gated differently per brain area, because far more entorhinal cells
# than hippocampal ones are active in a given 10 ms bin (chunkThetaPosSweeps.m).
BRANCHES = {
    "hc":  dict(min_active_cells=1, max_lowpass_error_cm=50.0,
                min_peak_correlation=0.0, decoded_smooth_bins=1.0),
    "mec": dict(min_active_cells=5, max_lowpass_error_cm=15.0,
                min_peak_correlation=None, decoded_smooth_bins=0.8),
}

RESULTS_COLUMNS = (f"{'session':<24}{'units':>6}{'cycles':>8}{'running':>8}{'sweeps':>7}"
                   f"{'prev':>7}{'len_cm':>8}{'trip':>6}{'alt%':>7}{'null%':>7}  {'hd':<7}")


def _build_parser(here: str):
    import argparse
    parser = argparse.ArgumentParser(
        description="Theta-sweep extraction (Vollan et al. 2025 port).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("nwb", nargs="*",
                        help="NWB file(s). Default: every *.nwb in the working directory.")
    parser.add_argument("--nodes", default=os.path.join(here, "node_list_new.csv"),
                        help="CSV of maze node coordinates")
    parser.add_argument("--exclude", nargs="*", default=[], metavar="NAME",
                        help="skip sessions whose filename contains any of these")

    parser.add_argument("--branch", choices=sorted(BRANCHES), default="hc",
                        help="which brain area's reliability gates to use")
    parser.add_argument("--px-per-cm", type=float, default=Config.px_per_cm,
                        help="pixel scale; default is measured from the 40 cm corridor segment")
    parser.add_argument("--quality", default="good", choices=["good", "good+mua"],
                        help="multi-unit clusters are OFF by default")
    parser.add_argument("--cell-types", default="pyramidal", choices=["pyramidal", "all"])

    parser.add_argument("--theta", default="lfp", choices=["lfp", "pca"],
                        help="where theta comes from; 'pca' reproduces the paper exactly")
    parser.add_argument("--head-direction", default="travel",
                        choices=["travel", "dlc", "auto"],
                        help="'travel' uses direction of motion; 'dlc' forces two DLC body "
                             "parts even if they fail the check; 'auto' uses DLC only where "
                             "it passes the check and travel elsewhere")
    parser.add_argument("--dlc-parts", nargs=2, default=list(Config.dlc_head_parts),
                        metavar=("FRONT", "BACK"),
                        help="which DLC body parts define the head axis")
    parser.add_argument("--max-sweep-origin-cm", type=float, default=Config.max_sweep_origin_cm,
                        help="a sweep's near end must be this close to the animal; use a "
                             "huge value to switch the criterion off")
    parser.add_argument("--unvisited-margin-cm", type=float, default=Config.unvisited_margin_cm,
                        help="how far past the visited area the decoder may place the "
                             "animal, so a sweep can leave the travelled path")
    parser.add_argument("--spatial-bin-cm", type=float, default=Config.spatial_bin_cm)
    parser.add_argument("--rate-smooth-cm", type=float, default=Config.rate_smooth_cm)
    parser.add_argument("--speed-smooth-s", type=float, default=Config.speed_smooth_s,
                        help="Gaussian width used to smooth the speed estimate")
    parser.add_argument("--speed-despike-cm-s", type=float, default=None,
                        help="replace tracking jumps faster than this before smoothing")
    parser.add_argument("--no-err-gate", action="store_true",
                        help="keep decoded bins even where the decoder lost the animal")

    parser.add_argument("--plot", metavar="DIR",
                        help="save one figure per session into this directory")
    parser.add_argument("--skip-lfp-check", action="store_true",
                        help="do not run the LFP sampling-rate pre-flight")
    parser.add_argument("-o", "--out", metavar="CSV", help="write the results table here")
    return parser


def _config_kwargs_from_args(args) -> dict:
    """Translate command-line arguments into `Config` fields."""
    kwargs = dict(BRANCHES[args.branch])
    kwargs.update(
        px_per_cm=args.px_per_cm,
        theta_source=args.theta,
        max_sweep_origin_cm=args.max_sweep_origin_cm,
        unvisited_margin_cm=args.unvisited_margin_cm,
        dlc_head_parts=tuple(args.dlc_parts),
        spatial_bin_cm=args.spatial_bin_cm,
        rate_smooth_cm=args.rate_smooth_cm,
        speed_smooth_s=args.speed_smooth_s,
        speed_despike_cm_s=args.speed_despike_cm_s,
        quality=("good", "mua") if args.quality == "good+mua" else ("good",),
        cell_types=None if args.cell_types == "all" else ("pyramidal",))

    if args.no_err_gate:
        kwargs["max_lowpass_error_cm"] = 1e9        # effectively no limit
    return kwargs


def _format_results_row(name: str, stats: dict, head_direction: str) -> str:
    """One line of the results table. Undefined numbers print blank, not 'nan'."""
    def cell(value, width, places):
        return f"{'':>{width}}" if value != value else f"{value:>{width}.{places}f}"

    # Alternation is meaningless without at least one triplet of consecutive sweeps.
    has_triplets = stats["n_triplets"] > 0
    alternation_pct = stats["alternation"] * 100 if has_triplets else np.nan
    null_pct = stats["alternation_null"] * 100 if has_triplets else np.nan

    return (f"{name:<24}"
            f"{stats['n_units']:>6}"
            f"{stats['n_cycles']:>8}"
            f"{stats['n_running']:>8}"
            f"{stats['n_sweeps']:>7}"
            f"{stats['prevalence']:>7.3f}"
            f"{cell(stats['mean_length_cm'], 8, 1)}"
            f"{stats['n_triplets']:>6}"
            f"{cell(alternation_pct, 7, 1)}"
            f"{cell(null_pct, 7, 1)}"
            f"  {head_direction:<7}")


def main(argv=None):
    import glob
    import traceback

    # Defaults come from the working directory, not from where this file happens to live.
    here = os.getcwd()
    args = _build_parser(here).parse_args(argv)

    nwb_files = args.nwb or sorted(glob.glob(os.path.join(here, "*.nwb")))
    nwb_files = [f for f in nwb_files
                 if not any(skip in os.path.basename(f) for skip in args.exclude)]
    if not nwb_files:
        _build_parser(here).error("no NWB files found")

    config_kwargs = _config_kwargs_from_args(args)

    if args.plot:
        os.makedirs(args.plot, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")               # write files without needing a display

    # --- pre-flight: resolve every LFP rate first, so warnings stay out of the table
    lfp_records: dict[str, LfpRate] = {}
    if not args.skip_lfp_check and args.theta == "lfp":
        lfp_records = check_lfp_rates(nwb_files, Config(**config_kwargs))
        print_lfp_check(lfp_records)
        print()

    # --- pre-flight: is the DLC head direction believable? ---------------------
    head_direction_records: dict[str, HeadDirectionCheck] = {}
    if args.head_direction in ("dlc", "auto"):
        head_direction_records = check_head_direction(nwb_files, Config(**config_kwargs))
        n_unusable = print_head_direction_check(head_direction_records)

        # Say what the run will actually do about it, so the check cannot look like a
        # decision it did not make.
        if n_unusable and args.head_direction == "dlc":
            print("--head-direction dlc: using DLC anyway. Pass --head-direction auto to "
                  "use the direction of travel wherever the check fails.")
        elif n_unusable:
            print(f"--head-direction auto: {n_unusable} session(s) will use the direction "
                  f"of travel instead.")
        print()

    # --- the results table -----------------------------------------------------
    print(f"branch={args.branch}  px/cm={args.px_per_cm:.4f}  theta={args.theta}  "
          f"head_dir={args.head_direction}  "
          f"units={args.quality}/{args.cell_types}  speed_sigma={args.speed_smooth_s}s  "
          f"({len(nwb_files)} session(s))\n")
    print(RESULTS_COLUMNS)
    print("-" * len(RESULTS_COLUMNS))

    rows, fell_back = [], []
    started_all = time.time()

    for index, nwb_file in enumerate(nwb_files, 1):
        name = os.path.basename(nwb_file)
        started = time.time()

        session_kwargs = dict(config_kwargs)
        if name in lfp_records:
            session_kwargs["lfp_rate_hz"] = lfp_records[name].rate_hz

        # Decide this session's head direction. "dlc" forces DLC; "auto" uses it only where
        # the preflight passed it. Either way a file with no DLC at all uses travel rather
        # than failing. The `hd` column records what each row actually used.
        record = head_direction_records.get(name)
        head_direction = args.head_direction

        if record is None or not record.has_dlc:
            head_direction = "travel"
        elif args.head_direction == "auto":
            head_direction = "dlc" if record.usable else "travel"

        if args.head_direction != "travel" and head_direction == "travel":
            fell_back.append(name)
        session_kwargs["head_direction_source"] = head_direction

        def show(stage, name=name, index=index, started=started):
            status(f"  [{index}/{len(nwb_files)}] {name}  {stage}   ({_elapsed(started)})")

        try:
            result = run(nwb_file, args.nodes, Config(**session_kwargs), verbose=False,
                         on_progress=show)
        except Exception as error:
            status()
            print(f"{name:<24}  FAILED: {type(error).__name__}: {error}")
            traceback.print_exc()
            continue

        if args.plot:
            show("plotting")
            import matplotlib.pyplot as plt
            figure = plot_sweeps(result, verbose=False,
                                 save_path=os.path.join(args.plot, name.replace(".nwb", "_sweeps.png")))
            plt.close(figure)

        # Erase the status line, so the finished row lands on a clean line of its own.
        status()
        print(_format_results_row(name, result.stats, head_direction), flush=True)

        rows.append(dict(session=name, head_direction=head_direction,
                         **{k: v for k, v in result.stats.items() if k != "shuffle_99"}))

    if not rows:
        return 1

    # --- totals ----------------------------------------------------------------
    table = pd.DataFrame(rows)
    print("-" * len(RESULTS_COLUMNS))
    print(f"{'TOTAL/MEAN':<24}{'':>6}"
          f"{table.n_cycles.sum():>8}"
          f"{table.n_running.sum():>8}"
          f"{table.n_sweeps.sum():>7}"
          f"{table.prevalence.mean():>7.3f}"
          f"{table.mean_length_cm.mean():>8.1f}"
          f"{table.n_triplets.sum():>6}")

    print(f"\n{len(rows)} session(s) in {_elapsed(started_all)}")

    if fell_back:
        print(f"\n{len(fell_back)} session(s) used the direction of travel instead of DLC: "
              f"{', '.join(fell_back)}")

    if args.plot:
        print(f"\nwrote {len(rows)} figures to {args.plot}/")

    print(f"\npaper: prevalence {PAPER_PREVALENCE}, mean length {PAPER_SWEEP_LENGTH_CM} cm, "
          f"alternation {PAPER_ALTERNATION_PCT}% vs {PAPER_ALTERNATION_SHUFFLE_PCT}% shuffle")

    if table.n_triplets.sum() == 0:
        print("NOTE: zero adjacent-cycle sweep triplets -> the paper's alternation "
              "statistic is undefined on this data, not zero.")

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0
