"""Loader for Metrica Game 1 and Game 2 CSV tracking data."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_metrica_csv_tracking(home_csv_path, away_csv_path, pitch_size=(105.0, 68.0), max_frames=None):
    """Load Metrica CSV tracking format into standardized arrays in metres.

    Parameters
    ----------
    home_csv_path : str or Path
    away_csv_path : str or Path
    pitch_size : tuple of (float, float)
    max_frames : int, optional

    Returns
    -------
    dict
        {
            "frame_ids": (T,) int,
            "ball_xy": (T, 2) in metres,
            "player_h": (T, N_H, 2) in metres,
            "player_a": (T, N_A, 2) in metres,
        }
    """
    home_csv_path = Path(home_csv_path)
    away_csv_path = Path(away_csv_path)
    pitch_w, pitch_h = pitch_size

    # Read Home CSV (rows start at line index 3, 0-indexed line 2 is header)
    df_h = pd.read_csv(home_csv_path, skiprows=2, nrows=max_frames)
    df_a = pd.read_csv(away_csv_path, skiprows=2, nrows=max_frames)

    frame_ids = df_h.iloc[:, 1].to_numpy(dtype=int)

    # Ball columns are the last 2 columns in df_h (or df_a)
    ball_raw = df_h.iloc[:, -2:].to_numpy(dtype=np.float32)
    ball_xy = np.column_stack([ball_raw[:, 0] * pitch_w, ball_raw[:, 1] * pitch_h])

    # Player columns: from column 3 up to -2
    # Each player has 2 columns (x, y)
    h_coords_raw = df_h.iloc[:, 3:-2].to_numpy(dtype=np.float32)
    n_h_players = h_coords_raw.shape[1] // 2
    h_reshaped = h_coords_raw.reshape(-1, n_h_players, 2)
    h_reshaped[:, :, 0] *= pitch_w
    h_reshaped[:, :, 1] *= pitch_h

    a_coords_raw = df_a.iloc[:, 3:-2].to_numpy(dtype=np.float32)
    n_a_players = a_coords_raw.shape[1] // 2
    a_reshaped = a_coords_raw.reshape(-1, n_a_players, 2)
    a_reshaped[:, :, 0] *= pitch_w
    a_reshaped[:, :, 1] *= pitch_h

    return {
        "frame_ids": frame_ids,
        "ball_xy": ball_xy,
        "player_h": h_reshaped,
        "player_a": a_reshaped,
    }
