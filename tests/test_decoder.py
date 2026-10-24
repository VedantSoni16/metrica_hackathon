"""Unit tests for Viterbi decoding and post-processing."""

import numpy as np
import pytest

from mtah.decode.postprocess import min_spell_merge, runs
from mtah.decode.viterbi import (
    build_log_transmat,
    scaled_log_emissions,
    viterbi_log,
)


def test_build_log_transmat():
    mean_dur = {"H": 350.0, "A": 350.0, "D": 200.0}
    leave_share = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])
    log_trans = build_log_transmat(mean_dur, leave_share)
    assert log_trans.shape == (3, 3)
    # Check probabilities sum to 1
    P = np.exp(log_trans)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(3), rtol=1e-5)
    # Diagonal should be close to 1 - 1/dur
    assert P[0, 0] == pytest.approx(1.0 - 1.0 / 350.0, rel=1e-4)
    assert P[1, 1] == pytest.approx(1.0 - 1.0 / 350.0, rel=1e-4)
    assert P[2, 2] == pytest.approx(1.0 - 1.0 / 200.0, rel=1e-4)


def test_viterbi_log_basic_and_edge_cases():
    mean_dur = {"H": 100.0, "A": 100.0, "D": 50.0}
    leave_share = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    log_trans = build_log_transmat(mean_dur, leave_share)

    # Empty
    path_empty = viterbi_log(np.empty((0, 3)), log_trans)
    assert len(path_empty) == 0

    # Single frame
    emit_single = np.array([[10.0, 0.0, 0.0]])
    path_single = viterbi_log(emit_single, log_trans)
    assert len(path_single) == 1
    assert path_single[0] == 0

    # Strong constant evidence for A (1)
    T = 50
    emit_a = np.zeros((T, 3))
    emit_a[:, 1] = 5.0
    path_a = viterbi_log(emit_a, log_trans)
    assert np.all(path_a == 1)

    # Clear two-phase transition: 30 frames H then 30 frames A
    emit_seq = np.zeros((60, 3))
    emit_seq[:30, 0] = 5.0
    emit_seq[30:, 1] = 5.0
    path_seq = viterbi_log(emit_seq, log_trans)
    assert np.all(path_seq[:30] == 0)
    assert np.all(path_seq[30:] == 1)


def test_scaled_log_emissions():
    post = np.array([
        [0.8, 0.1, 0.1],
        [0.33, 0.33, 0.34],
        [0.1, 0.8, 0.1],
    ])
    prior = np.array([0.4, 0.4, 0.2])
    w = np.array([1.0, 0.0, 0.5])

    log_emit = scaled_log_emissions(post, prior, w)
    assert log_emit.shape == (3, 3)
    # Frame 1 has w=0, so log_emit should be exactly 0
    np.testing.assert_allclose(log_emit[1], np.zeros(3))
    # Frame 0 has w=1, state 0 has positive log odds
    assert log_emit[0, 0] > 0
    assert log_emit[0, 1] < 0


def test_runs():
    arr = np.array([0, 0, 1, 1, 1, 2, 0])
    r = runs(arr)
    assert r == [
        (0, 2, 0),
        (2, 5, 1),
        (5, 6, 2),
        (6, 7, 0),
    ]
    assert runs(np.array([])) == []
    assert runs(np.array([1])) == [(0, 1, 1)]


def test_min_spell_merge():
    # Spurious 4-frame flicker of H (0) inside A (1)
    labels = np.array([1]*20 + [0]*4 + [1]*20, dtype=np.int8)
    post = np.zeros((len(labels), 3))
    post[:, 1] = 0.8
    post[:, 0] = 0.2
    merged = min_spell_merge(labels, post, min_len=10)
    assert np.all(merged == 1)

    # Short D (2 frames) between same team H (0)
    labels2 = np.array([0]*30 + [2]*2 + [0]*30, dtype=np.int8)
    post2 = np.zeros((len(labels2), 3))
    post2[:, 0] = 0.9
    merged2 = min_spell_merge(labels2, post2, min_len=10, min_d_len=3)
    assert np.all(merged2 == 0)

    # Valid long spells should NOT be merged
    labels3 = np.array([0]*25 + [1]*25, dtype=np.int8)
    post3 = np.zeros((len(labels3), 3))
    post3[:25, 0] = 0.9
    post3[25:, 1] = 0.9
    merged3 = min_spell_merge(labels3, post3, min_len=10)
    np.testing.assert_array_equal(merged3, labels3)
