"""Simulation and synthetic evaluation harness."""

from .harness import (
    corrupt_posteriors,
    evaluate_decoder_vs_argmax,
    generate_synthetic_ground_truth,
    run_harness_sweep,
)

__all__ = [
    "corrupt_posteriors",
    "evaluate_decoder_vs_argmax",
    "generate_synthetic_ground_truth",
    "run_harness_sweep",
]
