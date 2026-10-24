"""Unit tests for Stage 3a model and nearest-player baseline."""

import numpy as np

from mtah.models.baseline_model import PossessionClassifier, run_nearest_player_baseline


def test_possession_classifier_fit_predict():
    rng = np.random.default_rng(42)
    N = 200
    K = 10
    X = rng.normal(0, 1, size=(N, K))
    # Synthetic target: feature 0 strongly predicts class
    y = np.zeros(N, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = 2

    clf = PossessionClassifier(n_estimators=20, max_depth=3)
    clf.fit(X, y)

    prob = clf.predict_proba(X)
    assert prob.shape == (N, 3)
    np.testing.assert_allclose(prob.sum(axis=1), np.ones(N), rtol=1e-5)

    preds = clf.predict(X)
    assert len(preds) == N
    # Accuracy should be high on simple synthetic problem
    acc = np.mean(preds == y)
    assert acc > 0.70


def test_nearest_player_baseline():
    T = 50
    # Ball is at (20, 30) for frames 0..25, then missing (NaN) for 25..50
    ball = np.zeros((T, 2))
    ball[:25] = [20.0, 30.0]
    ball[25:] = np.nan

    # Home player at (21, 30) (distance 1m)
    # Away player at (40, 30) (distance 20m)
    h_players = np.ones((T, 5, 2)) * 21.0
    a_players = np.ones((T, 5, 2)) * 40.0

    preds = run_nearest_player_baseline(ball, h_players, a_players, window=5)
    assert len(preds) == T
    # First 20 frames should be Home (0)
    assert np.all(preds[:20] == 0)
    # Final frames should be Dead (2)
    assert np.all(preds[30:] == 2)
