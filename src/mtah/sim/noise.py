"""ATD-like noise simulator for realistic corruptions of clean tracking data.

Unit convention
---------------
- All coordinate inputs (ball_xy, player_xy_*) must be in METRES.
- ball_jitter_m and player_jitter_m are Gaussian std devs in METRES.
- mean_miss_run is in FRAMES (multiply by 1/fps to get seconds).
  Use mean_miss_run_inplay (runs <=200 frames) to avoid inflating the
  simulation with long dead-ball gaps that are not tracking dropouts.
"""

import numpy as np


def simulate_ball_visibility(T, miss_frac=0.43, mean_miss_run=25, rng=None):
    """Simulate realistic two-state Markov chain of ball visibility.

    Parameters
    ----------
    T : int
        Number of frames.
    miss_frac : float
        Target stationary missing fraction (default 0.43 for ATD).
    mean_miss_run : float
        Mean missing run length in frames (default 25 = 1.0 s).
    rng : np.random.Generator, optional
        RNG instance.

    Returns
    -------
    vis : np.ndarray
        (T,) boolean mask where True = visible, False = missing.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if T <= 0:
        return np.zeros(0, dtype=bool)

    b = 1.0 / float(mean_miss_run)
    a = b * miss_frac / (1.0 - miss_frac)

    vis = np.ones(T, dtype=bool)
    for t in range(1, T):
        if vis[t - 1]:
            vis[t] = rng.random() >= a
        else:
            vis[t] = rng.random() < b
    return vis


import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_calibrated_noise_params(config_path=None):
    """Load calibrated noise parameters from configs/noise_params.json if available."""
    if config_path is None:
        config_path = ROOT / "configs/noise_params.json"
    p = Path(config_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def apply_atd_noise(
    ball_xy,
    player_xy_h,
    player_xy_a,
    rng=None,
    ball_jitter_m=None,
    player_jitter_m=None,
    miss_frac=None,
    mean_miss_run=None,
    config_path=None,
):
    """Apply complete ATD noise simulation to clean match tracking coordinates.

    Parameters
    ----------
    ball_xy : np.ndarray
        (T, 2) clean ball coordinates (in metres).
    player_xy_h : np.ndarray
        (T, N_H, 2) clean Home player coordinates.
    player_xy_a : np.ndarray
        (T, N_A, 2) clean Away player coordinates.
    rng : np.random.Generator, optional
        RNG generator.
    ball_jitter_m : float, optional
        Positional noise standard deviation for ball in metres.
    player_jitter_m : float, optional
        Positional noise standard deviation for players in metres.
    miss_frac : float, optional
        Target stationary missing fraction.
    mean_miss_run : float, optional
        Mean missing run length in frames.
    config_path : Path or str, optional
        Path to calibrated noise params JSON.

    Returns
    -------
    noisy_ball : np.ndarray
        (T, 2) corrupted ball trajectory with dropouts and jitter.
    noisy_h : np.ndarray
        (T, N_H, 2) corrupted Home players with slot dropouts and jitter.
    noisy_a : np.ndarray
        (T, N_A, 2) corrupted Away players with slot dropouts and jitter.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    calib = load_calibrated_noise_params(config_path)
    if calib is not None:
        if miss_frac is None:
            miss_frac = calib.get("ball_missing_fraction", 0.407)
        if mean_miss_run is None:
            # Prefer in-play run length (<=200 frames); fall back to bounded overall mean.
            # The overall mean_missing_run_len (430.9) mixes dead-ball gaps with in-play
            # dropouts and inflates the Markov chain beyond realistic in-play behaviour.
            raw_inplay = calib.get("mean_missing_run_len_inplay", None)
            raw_overall = calib.get("mean_missing_run_len", 100.0)
            if raw_inplay is not None:
                mean_miss_run = float(raw_inplay)
            else:
                mean_miss_run = min(float(raw_overall), 150.0)
        if ball_jitter_m is None:
            # Units: metres (calibrate_noise.py computes on metre-space coords)
            ball_jitter_m = max(float(calib.get("ball_jitter_m", 0.1)), 0.05)
        if player_jitter_m is None:
            player_jitter_m = max(float(calib.get("player_jitter_m", 0.05)), 0.03)
        player_miss_frac = (22.0 - float(calib.get("mean_active_players", 20.49))) / 22.0
    else:
        if miss_frac is None:
            miss_frac = 0.41
        if mean_miss_run is None:
            mean_miss_run = 50.0
        if ball_jitter_m is None:
            ball_jitter_m = 0.15
        if player_jitter_m is None:
            player_jitter_m = 0.10
        player_miss_frac = 0.07

    T = len(ball_xy)
    noisy_ball = ball_xy.copy().astype(float)
    noisy_h = player_xy_h.copy().astype(float)
    noisy_a = player_xy_a.copy().astype(float)

    # 1. Ball dropouts
    vis = simulate_ball_visibility(T, miss_frac=miss_frac, mean_miss_run=mean_miss_run, rng=rng)
    noisy_ball[~vis] = np.nan

    # 2. Positional jitter
    ball_noise = rng.normal(0, ball_jitter_m, size=noisy_ball.shape)
    noisy_ball += ball_noise  # NaNs remain NaN

    if noisy_h.size > 0:
        h_noise = rng.normal(0, player_jitter_m, size=noisy_h.shape)
        noisy_h += h_noise

    if noisy_a.size > 0:
        a_noise = rng.normal(0, player_jitter_m, size=noisy_a.shape)
        noisy_a += a_noise

    # 3. Occasional player dropout (calibrated to mean active player count ~20.5)
    for p in range(noisy_h.shape[1]):
        p_vis = simulate_ball_visibility(T, miss_frac=player_miss_frac, mean_miss_run=30, rng=rng)
        noisy_h[~p_vis, p, :] = np.nan

    for p in range(noisy_a.shape[1]):
        p_vis = simulate_ball_visibility(T, miss_frac=player_miss_frac, mean_miss_run=30, rng=rng)
        noisy_a[~p_vis, p, :] = np.nan

    return noisy_ball, noisy_h, noisy_a
