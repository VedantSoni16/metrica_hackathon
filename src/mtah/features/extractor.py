"""Vectorized feature extraction for ball kinematics, player proximity, and dead-ball cues."""

import numpy as np
import pandas as pd


def compute_derivatives_1d(arr, fps=25.0):
    """Compute 1st, 2nd, and 3rd central difference derivatives for (T, 2) coords in m/s, m/s^2."""
    T = len(arr)
    vel = np.full((T, 2), np.nan, dtype=np.float32)
    acc = np.full((T, 2), np.nan, dtype=np.float32)

    if T < 3:
        return vel, acc

    dt = 1.0 / fps
    # Central difference for velocity: (x[t+1] - x[t-1]) / (2 * dt)
    vel[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
    vel[0] = (arr[1] - arr[0]) / dt
    vel[-1] = (arr[-1] - arr[-2]) / dt

    # Central difference for acceleration
    acc[1:-1] = (vel[2:] - vel[:-2]) / (2.0 * dt)
    acc[0] = (vel[1] - vel[0]) / dt
    acc[-1] = (vel[-1] - vel[-2]) / dt

    return vel, acc


def extract_frame_features(
    ball_xy,
    player_h,
    player_a,
    ball_imputed=None,
    ball_missing=None,
    ball_missing_run_len=None,
    pitch_size=(105.3, 68.0),
    fps=25.0,
):
    """Extract complete feature matrix for frame-level classification.

    Parameters
    ----------
    ball_xy : np.ndarray
        (T, 2) ball coordinates in metres.
    player_h : np.ndarray
        (T, N_H, 2) Home player coordinates in metres.
    player_a : np.ndarray
        (T, N_A, 2) Away player coordinates in metres.
    ball_imputed : np.ndarray, optional
        (T,) boolean mask.
    ball_missing : np.ndarray, optional
        (T,) boolean mask.
    ball_missing_run_len : np.ndarray, optional
        (T,) int array.
    pitch_size : tuple of (float, float)
        (width_m, height_m).
    fps : float
        Frame rate (default 25.0).

    Returns
    -------
    pd.DataFrame
        (T, num_features) tabular feature matrix.
    """
    T = len(ball_xy)
    features = {}
    pitch_w, pitch_h = pitch_size

    # --- Group A: Ball Kinematics ---
    ball_vel, ball_acc = compute_derivatives_1d(ball_xy, fps=fps)
    ball_speed = np.hypot(ball_vel[:, 0], ball_vel[:, 1])
    ball_accel = np.hypot(ball_acc[:, 0], ball_acc[:, 1])

    features["ball_x"] = ball_xy[:, 0]
    features["ball_y"] = ball_xy[:, 1]
    features["ball_speed"] = ball_speed
    features["ball_accel"] = ball_accel

    # Rolling ball speeds (approx 0.2s, 0.5s, 1.0s)
    s_series = pd.Series(ball_speed)
    features["ball_speed_roll5"] = s_series.rolling(5, min_periods=1, center=True).mean().to_numpy()
    features["ball_speed_roll12"] = s_series.rolling(12, min_periods=1, center=True).mean().to_numpy()
    features["ball_speed_roll25"] = s_series.rolling(25, min_periods=1, center=True).mean().to_numpy()

    # --- Group B: Proximity and Soft Influence ---
    # Ball distances to Home players: shape (T, N_H)
    diff_h = player_h - ball_xy[:, None, :]
    dist_h = np.hypot(diff_h[:, :, 0], diff_h[:, :, 1])

    # Ball distances to Away players: shape (T, N_A)
    diff_a = player_a - ball_xy[:, None, :]
    dist_a = np.hypot(diff_a[:, :, 0], diff_a[:, :, 1])

    valid_ball = ~(np.isnan(ball_xy[:, 0]) | np.isnan(ball_xy[:, 1]))

    # Safe min distances per team
    min_dist_h = np.full(T, np.nan, dtype=np.float32)
    min_dist_a = np.full(T, np.nan, dtype=np.float32)
    second_dist_h = np.full(T, np.nan, dtype=np.float32)
    second_dist_a = np.full(T, np.nan, dtype=np.float32)

    if np.any(valid_ball):
        dist_h_v = dist_h[valid_ball]
        dist_a_v = dist_a[valid_ball]
        min_dist_h[valid_ball] = np.nanmin(dist_h_v, axis=1)
        min_dist_a[valid_ball] = np.nanmin(dist_a_v, axis=1)

        sorted_h = np.sort(dist_h_v, axis=1)
        sorted_a = np.sort(dist_a_v, axis=1)
        second_dist_h[valid_ball] = sorted_h[:, 1] if sorted_h.shape[1] > 1 else sorted_h[:, 0]
        second_dist_a[valid_ball] = sorted_a[:, 1] if sorted_a.shape[1] > 1 else sorted_a[:, 0]

    features["min_dist_h"] = min_dist_h
    features["min_dist_a"] = min_dist_a
    features["delta_dist"] = min_dist_a - min_dist_h
    features["second_dist_h"] = second_dist_h
    features["second_dist_a"] = second_dist_a

    # Player counts within radii (2m, 4m, 8m)
    for r in (2.0, 4.0, 8.0):
        c_h = np.zeros(T, dtype=np.float32)
        c_a = np.zeros(T, dtype=np.float32)
        if np.any(valid_ball):
            c_h[valid_ball] = np.nansum(dist_h[valid_ball] < r, axis=1)
            c_a[valid_ball] = np.nansum(dist_a[valid_ball] < r, axis=1)
        features[f"count_h_{int(r)}m"] = c_h
        features[f"count_a_{int(r)}m"] = c_a
        features[f"diff_count_{int(r)}m"] = c_h - c_a

    # Soft influence: sum exp(-d / tau)
    for tau in (1.5, 3.0, 6.0):
        inf_h = np.zeros(T, dtype=np.float32)
        inf_a = np.zeros(T, dtype=np.float32)
        if np.any(valid_ball):
            inf_h[valid_ball] = np.nansum(np.exp(-dist_h[valid_ball] / tau), axis=1)
            inf_a[valid_ball] = np.nansum(np.exp(-dist_a[valid_ball] / tau), axis=1)
        features[f"influence_h_{tau}"] = inf_h
        features[f"influence_a_{tau}"] = inf_a
        features[f"diff_influence_{tau}"] = inf_h - inf_a

    # --- Group C: Control Proxy ---
    # Velocity of Home player slot 0 as baseline proxy
    vel_h, _ = compute_derivatives_1d(player_h[:, 0, :], fps=fps)
    p_speed = np.hypot(vel_h[:, 0], vel_h[:, 1])
    features["player_speed_proxy"] = p_speed
    features["rel_speed"] = np.abs(ball_speed - p_speed)


    # --- Group D: Ball Missing Context ---
    if ball_missing is None:
        ball_missing = np.isnan(ball_xy[:, 0]) | np.isnan(ball_xy[:, 1])
    if ball_imputed is None:
        ball_imputed = np.zeros(T, dtype=bool)
    if ball_missing_run_len is None:
        ball_missing_run_len = np.zeros(T, dtype=int)

    features["ball_missing"] = ball_missing.astype(float)
    features["ball_imputed"] = ball_imputed.astype(float)
    features["ball_missing_run_len"] = np.log1p(ball_missing_run_len)

    # --- Group E: Team Centroids and Spread ---
    cent_h_x = np.nanmean(player_h[:, :, 0], axis=1)
    cent_h_y = np.nanmean(player_h[:, :, 1], axis=1)
    cent_a_x = np.nanmean(player_a[:, :, 0], axis=1)
    cent_a_y = np.nanmean(player_a[:, :, 1], axis=1)

    features["cent_h_x"] = cent_h_x
    features["cent_h_y"] = cent_h_y
    features["cent_a_x"] = cent_a_x
    features["cent_a_y"] = cent_a_y
    features["diff_cent_x"] = cent_h_x - cent_a_x

    features["spread_h"] = np.nanstd(player_h[:, :, 0], axis=1) + np.nanstd(player_h[:, :, 1], axis=1)
    features["spread_a"] = np.nanstd(player_a[:, :, 0], axis=1) + np.nanstd(player_a[:, :, 1], axis=1)

    # --- Group F: Dead Ball Cues ---
    out_of_pitch = (
        (ball_xy[:, 0] < 0.0)
        | (ball_xy[:, 0] > pitch_w)
        | (ball_xy[:, 1] < 0.0)
        | (ball_xy[:, 1] > pitch_h)
    )
    dist_to_boundary = np.minimum.reduce([
        ball_xy[:, 0],
        pitch_w - ball_xy[:, 0],
        ball_xy[:, 1],
        pitch_h - ball_xy[:, 1],
    ])
    features["out_of_pitch"] = out_of_pitch.astype(float)
    features["dist_to_boundary"] = dist_to_boundary

    # --- Group H: Goal Distances ---
    goal_h = np.array([0.0, pitch_h / 2.0])
    goal_a = np.array([pitch_w, pitch_h / 2.0])
    features["dist_goal_h"] = np.hypot(ball_xy[:, 0] - goal_h[0], ball_xy[:, 1] - goal_h[1])
    features["dist_goal_a"] = np.hypot(ball_xy[:, 0] - goal_a[0], ball_xy[:, 1] - goal_a[1])

    return pd.DataFrame(features)
