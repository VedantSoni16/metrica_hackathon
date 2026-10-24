"""Unit tests for ball healing, noise simulation, and feature extraction."""

import numpy as np
import pytest

from mtah.features.extractor import extract_frame_features
from mtah.healing.ball import heal_ball_trajectory
from mtah.sim.noise import apply_atd_noise, simulate_ball_visibility


def test_heal_ball_trajectory():
    # Sequence of 50 frames with a 5-frame gap and a 25-frame gap
    xy = np.zeros((60, 2))
    xy[:, 0] = np.arange(60) * 1.0
    xy[:, 1] = 30.0

    # Short gap of 5 frames (should be interpolated)
    xy[10:15] = np.nan
    # Long gap of 25 frames (should remain NaN)
    xy[25:50] = np.nan

    res = heal_ball_trajectory(xy, max_gap=15)
    healed_xy = res["ball_xy"]

    # Short gap check
    assert not np.isnan(healed_xy[12, 0])
    assert healed_xy[12, 0] == pytest.approx(12.0)
    assert res["ball_imputed"][12] == True
    assert res["ball_reliability"][12] == pytest.approx(0.6)

    # Long gap check
    assert np.isnan(healed_xy[30, 0])
    assert res["ball_imputed"][30] == False
    assert res["ball_missing"][30] == True
    assert res["ball_reliability"][30] == pytest.approx(0.1)



def test_noise_simulation():
    T = 1000
    vis = simulate_ball_visibility(T, miss_frac=0.40, mean_miss_run=25, rng=np.random.default_rng(42))
    assert len(vis) == T
    # Missing fraction should be roughly near 0.40
    miss_rate = 1.0 - np.mean(vis)
    assert 0.30 <= miss_rate <= 0.50

    ball = np.ones((T, 2)) * 50.0
    players_h = np.ones((T, 11, 2)) * 40.0
    players_a = np.ones((T, 11, 2)) * 60.0

    n_ball, n_h, n_a = apply_atd_noise(ball, players_h, players_a, rng=np.random.default_rng(123))
    # Check shape preserved
    assert n_ball.shape == ball.shape
    assert n_h.shape == players_h.shape
    assert n_a.shape == players_a.shape
    # Check that NaNs were introduced into ball
    assert np.isnan(n_ball[:, 0]).sum() > 0


def test_extract_frame_features():
    T = 100
    # Create constant velocity ball moving from (10, 34) towards (60, 34)
    ball = np.zeros((T, 2))
    ball[:, 0] = np.linspace(10, 60, T)
    ball[:, 1] = 34.0

    # Home players near (20, 34), Away players near (80, 34)
    player_h = np.ones((T, 11, 2)) * 20.0
    player_a = np.ones((T, 11, 2)) * 80.0

    feats = extract_frame_features(ball, player_h, player_a, pitch_size=(105.3, 68.0))
    assert len(feats) == T
    assert "ball_speed" in feats.columns
    assert "min_dist_h" in feats.columns
    assert "delta_dist" in feats.columns
    assert "influence_h_3.0" in feats.columns
    assert "out_of_pitch" in feats.columns

    # Initial frames ball is close to Home (x=10-20), far from Away (x=80)
    # delta_dist = min_dist_a - min_dist_h > 0
    assert feats["delta_dist"].iloc[5] > 0
    # Soft influence for Home should exceed Away initially
    assert feats["diff_influence_3.0"].iloc[5] > 0
