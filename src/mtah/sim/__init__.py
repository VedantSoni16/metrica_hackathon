"""Simulation and synthetic evaluation harness."""

from .harness import (
    corrupt_posteriors,
    evaluate_decoder_vs_argmax,
    generate_synthetic_ground_truth,
    run_harness_sweep,
)
from .noise import apply_atd_noise, simulate_ball_visibility

__all__ = [
    "apply_atd_noise",
    "corrupt_posteriors",
    "evaluate_decoder_vs_argmax",
    "generate_synthetic_ground_truth",
    "run_harness_sweep",
    "simulate_ball_visibility",
]
