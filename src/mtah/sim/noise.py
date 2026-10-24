"""ATD-like noise simulator for realistic corruptions of clean tracking data."""

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


def apply_atd_noise(ball_xy, player_xy_h, player_xy_a, rng=None, ball_jitter_m=0.3, player_jitter_m=0.2):
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
    ball_jitter_m : float
        Positional noise standard deviation for ball in metres.
    player_jitter_m : float
        Positional noise standard deviation for players in metres.

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

    T = len(ball_xy)
    noisy_ball = ball_xy.copy().astype(float)
    noisy_h = player_xy_h.copy().astype(float)
    noisy_a = player_xy_a.copy().astype(float)

    # 1. Ball dropouts
    vis = simulate_ball_visibility(T, miss_frac=0.43, mean_miss_run=25, rng=rng)
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

    # 3. Occasional player dropout (target ~20.5 visible players from 22)
    # Drop ~1-2 players per frame randomly
    for p in range(noisy_h.shape[1]):
        p_vis = simulate_ball_visibility(T, miss_frac=0.07, mean_miss_run=30, rng=rng)
        noisy_h[~p_vis, p, :] = np.nan

    for p in range(noisy_a.shape[1]):
        p_vis = simulate_ball_visibility(T, miss_frac=0.07, mean_miss_run=30, rng=rng)
        noisy_a[~p_vis, p, :] = np.nan

    return noisy_ball, noisy_h, noisy_a
