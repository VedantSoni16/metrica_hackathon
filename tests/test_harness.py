"""Tests for synthetic posterior decoder harness and edge cases."""

import numpy as np

from mtah.sim.harness import (
    corrupt_posteriors,
    evaluate_decoder_vs_argmax,
    generate_synthetic_ground_truth,
    run_harness_sweep,
)


def test_harness_spell_f1_gain():
    """Verify that Viterbi decoder yields a large, stable gain over raw argmax on spellF1."""
    sweep = run_harness_sweep()
    assert len(sweep) == 3

    for noise, res in sweep:
        # Argmax typically creates hundreds of fragmented spells
        assert res["n_spells_argmax"] > res["n_spells_viterbi"] * 2
        # Viterbi decoder should significantly beat argmax on spellF1
        assert res["delta_spell_f1"] > 0.15
        assert res["spell_f1_viterbi"] > res["spell_f1_argmax"]


def test_harness_edge_cases():
    # 1. Empty sequence
    empty_gt = np.empty(0, dtype=np.int8)
    empty_post, empty_w = corrupt_posteriors(empty_gt)
    res_empty = evaluate_decoder_vs_argmax(empty_gt, empty_post, empty_w)
    assert res_empty["delta_spell_f1"] == 0.0

    # 2. Single-class: all H (0)
    single_gt = np.zeros(200, dtype=np.int8)
    post_single, w_single = corrupt_posteriors(single_gt, noise_level=0.1, burst_prob=0.0, seed=1)
    res_single = evaluate_decoder_vs_argmax(single_gt, post_single, w_single)
    # Should not crash and should correctly identify H
    assert res_single["spell_f1_viterbi"] >= res_single["spell_f1_argmax"]

    # 3. Short half (10 frames)
    short_gt = np.array([0]*5 + [1]*5, dtype=np.int8)
    post_short, w_short = corrupt_posteriors(short_gt, noise_level=0.1, seed=2)
    res_short = evaluate_decoder_vs_argmax(short_gt, post_short, w_short)
    assert "spell_f1_viterbi" in res_short

    # 4. Extreme noise (noise_level = 0.9)
    noisy_gt = generate_synthetic_ground_truth(n_possessions=10, seed=3)
    post_extreme, w_extreme = corrupt_posteriors(noisy_gt, noise_level=0.9, seed=4)
    res_extreme = evaluate_decoder_vs_argmax(noisy_gt, post_extreme, w_extreme)
    assert "delta_spell_f1" in res_extreme
