"""Theta-sweep extraction from hippocampal recordings in a hex maze.

Python port of the code released with Vollan, Gardner, Moser & Moser (2025),
"Left-right-alternating theta sweeps in entorhinal-hippocampal maps of space",
Nature 639:995-1005.

Within each ~8 Hz theta cycle the encoded position sweeps outwards ahead of the animal,
alternating left and right on successive cycles. The pipeline is:

    load_session -> theta_cycles -> rate_maps -> decode
    -> lowpass_trajectory -> extract_sweeps -> alternation

    >>> from hexmaze_sweeps import Config, run, plot_sweeps
    >>> result = run("Rat6_20260629.nwb", "node_list_new.csv")
    >>> plot_sweeps(result, save_path="sweeps.png")

`decode` runs either of two decoders, because the paper's Methods text and the paper's
released code disagree about which one it used: `Config(decoder="pv")` is the authors'
population-vector correlation, `Config(decoder="bayes")` the Poisson reconstruction their
Methods describe. See `decoding.py`.

A preflight is worth running before you believe any of it -- `check_lfp_rates`. See
`data.py` for what it is protecting you from.
"""
from .config import (Config, PAPER_ALTERNATION_PCT, PAPER_ALTERNATION_SHUFFLE_PCT,
                     PAPER_PREVALENCE, PAPER_SWEEP_LENGTH_CM, SEGMENT_CM, status)
from .data import (LfpRate, Result, Session, calc_head_direction, check_lfp_rates,
                   infer_lfp_rate_hz, load_session, measure_px_per_cm, print_lfp_check)
from .decoding import (cross_validate_smoothing, decode, decoder_scores, lowpass_trajectory,
                       prepare_log_prior, prepare_tuning, rate_maps, shuffle_threshold,
                       theta_cycles)
from .pipeline import main, run
from .plotting import plot_all_sweeps, plot_sweeps
from .sweeps import (alternation, distance_outside_visited_cm, extract_sweeps,
                     head_centred_direction, maze_corridors, visited_mask)

__all__ = [
    "Config", "Session", "Result", "LfpRate",
    "run", "main", "load_session", "measure_px_per_cm",
    "check_lfp_rates", "print_lfp_check", "infer_lfp_rate_hz",
    "calc_head_direction",
    "theta_cycles", "rate_maps", "decode", "decoder_scores", "shuffle_threshold",
    "prepare_tuning", "prepare_log_prior", "cross_validate_smoothing", "lowpass_trajectory",
    "extract_sweeps", "alternation", "head_centred_direction",
    "maze_corridors", "visited_mask", "distance_outside_visited_cm",
    "plot_sweeps", "plot_all_sweeps",
    "SEGMENT_CM", "PAPER_PREVALENCE", "PAPER_SWEEP_LENGTH_CM",
    "PAPER_ALTERNATION_PCT", "PAPER_ALTERNATION_SHUFFLE_PCT", "status",
]
