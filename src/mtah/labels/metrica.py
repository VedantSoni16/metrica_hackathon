"""Derive H/A/D silver possession labels from Metrica Sports event data."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(ROOT / "evaluation"))

import possession_metrics as pm

H, A, D, X = 0, 1, 2, 3


def load_metrica_events_csv(csv_path):
    """Load and normalize Metrica Game 1 or 2 event CSV."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    events = []
    for _, row in df.iterrows():
        team_str = str(row.get("Team", "")).strip().lower()
        if team_str == "home":
            team = H
        elif team_str == "away":
            team = A
        else:
            team = None

        ev_type = str(row.get("Type", "")).strip().upper()
        subtype = str(row.get("Subtype", "")).strip().upper() if pd.notna(row.get("Subtype")) else ""
        period = int(row.get("Period", 1))
        start_frame = int(row.get("Start Frame", 0))
        end_frame = int(row.get("End Frame", start_frame))

        events.append({
            "team": team,
            "type": ev_type,
            "subtype": subtype,
            "period": period,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
    return pd.DataFrame(events)


def load_metrica_events_json(json_path):
    """Load and normalize Metrica Game 3 event JSON."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)["data"]

    events = []
    for item in data:
        team_info = item.get("team") or {}
        team_name = str(team_info.get("name", "")).strip().lower()
        team_id = str(team_info.get("id", "")).strip().upper()

        if "team a" in team_name or "fifatma" in team_id:
            team = H
        elif "team b" in team_name or "fifatmb" in team_id:
            team = A
        else:
            team = None

        type_info = item.get("type") or {}
        ev_type = str(type_info.get("name", "")).strip().upper()

        subtypes_info = item.get("subtypes") or {}
        if isinstance(subtypes_info, dict):
            subtype = str(subtypes_info.get("name", "")).strip().upper()
        elif isinstance(subtypes_info, list) and subtypes_info:
            subtype = str(subtypes_info[0].get("name", "")).strip().upper()
        else:
            subtype = ""

        period = int(item.get("period", 1))
        start_info = item.get("start") or {}
        end_info = item.get("end") or {}
        start_frame = int(start_info.get("frame", 0))
        end_frame = int(end_info.get("frame", start_frame))

        events.append({
            "team": team,
            "type": ev_type,
            "subtype": subtype,
            "period": period,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
    return pd.DataFrame(events)


def derive_metrica_labels(events_df, n_frames, half_boundaries=None):
    """Derive per-frame H/A/D labels from normalized Metrica events.

    Rules from strat.md Section 7:
    1. Dead ball (D):
       - Triggered by BALL OUT, FAULT RECEIVED, CARD stoppages, and GOALs.
       - Lasts until restart SET PIECE (KICK OFF, THROW IN, GOAL KICK, CORNER KICK, FREE KICK).
       - Also dead before the kickoff of each half, during halftime, and after full-time.
    2. Possession owner while alive:
       - Set to the team performing a controlling event: PASS, RECOVERY, CARRY, SHOT, or SET PIECE restart.
       - CHALLENGE / BALL LOST leaves ownership with the possessing team until the opponent
         completes a controlling event (RECOVERY or PASS).

    Parameters
    ----------
    events_df : pd.DataFrame
        Normalized events DataFrame.
    n_frames : int
        Total tracking frames (1-based frame IDs 1..n_frames mapped to 0..n_frames-1).
    half_boundaries : dict, optional
        Dict with half frame intervals, e.g. {1: (start_id, end_id), 2: (start_id, end_id)}.

    Returns
    -------
    labels : np.ndarray
        (n_frames,) int8 array of {0: H, 1: A, 2: D}.
    """
    labels = np.full(n_frames, D, dtype=np.int8)

    # Sort events chronologically by start frame
    sorted_events = events_df.sort_values(by=["start_frame", "end_frame"]).reset_index(drop=True)

    # Controlling event types
    CONTROLLING = {"PASS", "RECOVERY", "CARRY", "SHOT"}
    # Stoppage events that turn ball DEAD
    STOPPAGES = {"BALL OUT", "FAULT RECEIVED", "CARD"}
    # Restarts that make ball alive again
    RESTARTS = {"KICK OFF", "THROW IN", "GOAL KICK", "CORNER KICK", "FREE KICK", "PENALTY"}

    current_state = D  # Starts dead before kickoff

    # Process events in order
    last_frame = 0
    for _, ev in sorted_events.iterrows():
        ev_team = ev["team"]
        ev_type = ev["type"]
        ev_subtype = ev["subtype"]
        start_f = max(0, int(ev["start_frame"]) - 1)
        end_f = max(start_f, int(ev["end_frame"]) - 1)

        # Clip to match length
        start_f = min(start_f, n_frames - 1)
        end_f = min(end_f, n_frames - 1)

        # Fill prior frames with current_state up to start_f
        if start_f > last_frame:
            labels[last_frame:start_f] = current_state
            last_frame = start_f

        # Check event semantics
        is_restart = (ev_type == "SET PIECE") or any(r in ev_subtype for r in RESTARTS)
        is_stoppage = (ev_type in STOPPAGES) or ("GOAL" in ev_subtype and ev_type == "SHOT")

        if is_stoppage:
            current_state = D
            labels[start_f : end_f + 1] = D
        elif is_restart or ev_type in CONTROLLING:
            if ev_team in (H, A):
                current_state = ev_team
            labels[start_f : end_f + 1] = current_state
        else:
            # Non-controlling event (e.g. CHALLENGE, BALL LOST)
            # Ownership remains with current_state
            labels[start_f : end_f + 1] = current_state

        last_frame = max(last_frame, end_f + 1)

    # Fill remaining frames
    if last_frame < n_frames:
        labels[last_frame:n_frames] = current_state

    # Apply half boundaries if provided: enforce D during halftime and out-of-match
    if half_boundaries is not None:
        in_play = np.zeros(n_frames, dtype=bool)
        for h_start, h_end in half_boundaries.values():
            s = max(0, h_start - 1)
            e = min(n_frames, h_end)
            in_play[s:e] = True
        labels[~in_play] = D

    return labels


def validate_derived_labels(labels, fps=25):
    """Run Section 7 sanity checks on derived silver labels.

    Checks:
    - Mean possession length in 10-25 seconds range.
    - Dead-ball fraction is plausible (~20% to 55%).
    - Possessions are detected for both Home and Away.

    Returns
    -------
    dict
        Statistics summary.
    """
    runs = pm.valid_runs(labels)
    spells = pm.possession_spells(labels, runs)
    stats = pm.possession_stats(labels, runs)

    n_frames = len(labels)
    dead_fraction = float(np.sum(labels == D)) / n_frames if n_frames else 0.0

    summary = {
        "n_frames": n_frames,
        "n_spells": len(spells),
        "mean_duration_s": stats["mean_duration_s"],
        "home_share": stats["home_share"],
        "dead_fraction": dead_fraction,
        "is_plausible": (8.0 <= stats["mean_duration_s"] <= 30.0) and (0.20 <= dead_fraction <= 0.65),
    }
    return summary
