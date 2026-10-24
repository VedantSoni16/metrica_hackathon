"""Parser for EPTS / FIFA tracking format (ATD and Metrica Game 3)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

FIELD_SEP = ":"
GROUP_SEP = ";"
COORD_SEP = ","


def parse_epts_metadata(meta_path):
    """Parse EPTS metadata XML file for team mapping and pitch dimensions.

    Returns
    -------
    dict
        {
            "pitch_size": (width_m, height_m),
            "home_id": str,
            "away_id": str,
            "slot_teams": np.ndarray (0=Home, 1=Away),
            "frame_rate": int,
        }
    """
    root = ET.parse(meta_path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    if len(teams) < 2:
        raise ValueError(f"Expected at least 2 teams in {meta_path}, got {len(teams)}")
    home_id, away_id = teams[0], teams[1]

    slots = [p.get("teamId") for p in root.iter("Player")]
    slot_teams = np.array([0 if tid == home_id else 1 for tid in slots], dtype=np.int8)

    # Pitch size
    pitch_size = (105.3, 68.0)
    field_size = root.find(".//FieldSize")
    if field_size is not None:
        w = field_size.find("Width")
        h = field_size.find("Height")
        if w is not None and h is not None and w.text and h.text:
            pitch_size = (float(w.text), float(h.text))

    fps = 25
    fr = root.find(".//FrameRate")
    if fr is not None and fr.text:
        fps = int(fr.text)

    return {
        "pitch_size": pitch_size,
        "home_id": home_id,
        "away_id": away_id,
        "slot_teams": slot_teams,
        "frame_rate": fps,
    }


def parse_epts_tracking_file(
    raw_path,
    slot_teams,
    pitch_size=(105.3, 68.0),
    max_frames=None,
    frame_range=None,
):
    """Parse EPTS raw text tracking file into metric coordinate arrays.

    Parameters
    ----------
    raw_path : str or Path
        Path to tracking.txt.
    slot_teams : np.ndarray
        (n_slots,) int8 array where 0=Home, 1=Away.
    pitch_size : tuple of (float, float)
        (width_m, height_m) to scale [0, 1] normalized coordinates.
    max_frames : int, optional
        Maximum number of frames to read.
    frame_range : tuple of (int, int), optional
        (first_frame_id, last_frame_id) inclusive range to read.

    Returns
    -------
    dict
        {
            "frame_ids": (T,) int array,
            "ball_xy": (T, 2) float array in metres (NaN if missing),
            "player_h": (T, N_H, 2) float array in metres (NaN if missing),
            "player_a": (T, N_A, 2) float array in metres (NaN if missing),
        }
    """
    raw_path = Path(raw_path)
    n_slots = len(slot_teams)
    h_idx = np.where(slot_teams == 0)[0]
    a_idx = np.where(slot_teams == 1)[0]
    n_h, n_a = len(h_idx), len(a_idx)

    pitch_w, pitch_h = pitch_size

    frame_ids = []
    ball_list = []
    h_list = []
    a_list = []

    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(FIELD_SEP)
            if len(parts) < 3:
                continue

            fid = int(parts[0])
            if frame_range is not None:
                if fid < frame_range[0]:
                    continue
                if fid > frame_range[1]:
                    break

            # Parse ball: parts[2] = "x,y" or "x,y,speed" or "NaN,NaN"
            ball_part = parts[2].split(COORD_SEP)
            try:
                bx = float(ball_part[0]) * pitch_w
                by = float(ball_part[1]) * pitch_h
            except (ValueError, IndexError):
                bx, by = np.nan, np.nan

            # Parse player slots: parts[1] = "x,y;x,y;..."
            slot_parts = parts[1].split(GROUP_SEP)
            coords = np.full((n_slots, 2), np.nan, dtype=np.float32)
            for i, p_str in enumerate(slot_parts[:n_slots]):
                c = p_str.split(COORD_SEP)
                try:
                    coords[i, 0] = float(c[0]) * pitch_w
                    coords[i, 1] = float(c[1]) * pitch_h
                except (ValueError, IndexError):
                    pass

            frame_ids.append(fid)
            ball_list.append([bx, by])
            h_list.append(coords[h_idx])
            a_list.append(coords[a_idx])

            if max_frames and len(frame_ids) >= max_frames:
                break

    return {
        "frame_ids": np.array(frame_ids, dtype=int),
        "ball_xy": np.array(ball_list, dtype=np.float32),
        "player_h": np.array(h_list, dtype=np.float32).reshape(len(frame_ids), n_h, 2),
        "player_a": np.array(a_list, dtype=np.float32).reshape(len(frame_ids), n_a, 2),
    }
