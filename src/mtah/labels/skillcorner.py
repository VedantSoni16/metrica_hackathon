"""Derive H/A/D silver possession labels from SkillCorner Open Data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtah.labels.metrica import A, D, H, X


def load_skillcorner_labels(match_dir, target_fps=25, source_fps=10):
    """Load and derive 25 fps H/A/D silver labels from a SkillCorner match folder.

    Parameters
    ----------
    match_dir : str or Path
        Directory containing <match_id>_match.json and <match_id>_phases_of_play.csv.
    target_fps : int
        Target frame rate (default 25 fps).
    source_fps : int
        SkillCorner tracking frame rate (default 10 fps).

    Returns
    -------
    labels : np.ndarray
        (N_25,) int8 array of labels in {0=H, 1=A, 2=D, 3=X}.
    meta : dict
        Match metadata including home/away teams and periods.
    """
    match_dir = Path(match_dir)
    match_files = list(match_dir.glob("*_match.json"))
    if not match_files:
        raise FileNotFoundError(f"No match.json found in {match_dir}")
    match_json_path = match_files[0]
    match_id = match_json_path.name.split("_")[0]

    phases_path = match_dir / f"{match_id}_phases_of_play.csv"
    if not phases_path.exists():
        raise FileNotFoundError(f"phases_of_play.csv not found at {phases_path}")

    with open(match_json_path, "r", encoding="utf-8") as f:
        match_meta = json.load(f)

    home_id = match_meta["home_team"]["id"]
    away_id = match_meta["away_team"]["id"]

    # Determine match extent at 10 fps
    periods = match_meta.get("match_periods", [])
    if periods:
        max_frame_10 = max(p["end_frame"] for p in periods) + 1
    else:
        max_frame_10 = 60000

    # Initialize 10 fps labels (default Dead = 2)
    labels_10 = np.full(max_frame_10, D, dtype=np.int8)

    # Load phases of play
    phases_df = pd.read_csv(phases_path)
    for _, row in phases_df.iterrows():
        sf = int(row["frame_start"])
        ef = int(row["frame_end"])
        team_id = row.get("team_in_possession_id")

        if pd.isna(team_id):
            label = D
        elif int(team_id) == home_id:
            label = H
        elif int(team_id) == away_id:
            label = A
        else:
            label = D

        s_clamped = max(0, min(max_frame_10 - 1, sf))
        e_clamped = max(0, min(max_frame_10 - 1, ef))
        if e_clamped >= s_clamped:
            labels_10[s_clamped:e_clamped + 1] = label

    # Mark out-of-period frames as X if applicable
    if len(periods) >= 2:
        p1_end = periods[0]["end_frame"]
        p2_start = periods[1]["start_frame"]
        if p2_start > p1_end:
            labels_10[p1_end + 1:p2_start] = X

        p1_start = periods[0]["start_frame"]
        if p1_start > 0:
            labels_10[:p1_start] = X

        p2_end = periods[1]["end_frame"]
        if p2_end < max_frame_10 - 1:
            labels_10[p2_end + 1:] = X

    # Resample from 10 fps to 25 fps
    total_time_s = max_frame_10 / float(source_fps)
    n_frames_25 = round(total_time_s * float(target_fps))

    # For each 25 fps frame, look up corresponding 10 fps frame
    t_25 = np.arange(n_frames_25) / float(target_fps)
    idx_10 = np.clip(np.round(t_25 * float(source_fps)).astype(int), 0, max_frame_10 - 1)
    labels_25 = labels_10[idx_10]

    meta_out = {
        "match_id": match_id,
        "home_id": home_id,
        "home_name": match_meta["home_team"]["name"],
        "away_id": away_id,
        "away_name": match_meta["away_team"]["name"],
        "n_frames_10": max_frame_10,
        "n_frames_25": n_frames_25,
        "periods": periods,
    }

    return labels_25, meta_out
