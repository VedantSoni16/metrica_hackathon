"""Loaders and feature encoders for homography and Smart Tagging."""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

MATRIX_COLUMNS = [f"h{row}{col}" for row in range(3) for col in range(3)]


def load_homography_csv(path):
    """Load homography CSV into (matrices, reprojection_error, valid).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict
        {
            "matrices": (N, 3, 3) float64 array,
            "reprojection_error": (N,) float64 array,
            "valid": (N,) bool array,
        }
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    valid = df["valid"].to_numpy().astype(bool)
    reproj = df["reprojection_error"].to_numpy().astype(np.float64)

    matrices = np.full((len(df), 3, 3), np.nan, dtype=np.float64)
    mat_cols = [c.lower() for c in MATRIX_COLUMNS]
    if all(col in df.columns for col in mat_cols):
        flat_vals = df[mat_cols].to_numpy(dtype=np.float64)
        matrices[valid] = flat_vals[valid].reshape(-1, 3, 3)

    return {
        "matrices": matrices,
        "reprojection_error": reproj,
        "valid": valid,
    }


def load_smart_tagging_xml(path, fps=25):
    """Parse Smart Tagging XML intervals.

    Parameters
    ----------
    path : str or Path
    fps : int, optional

    Returns
    -------
    list of dict
        Each dict has {code, start_s, end_s, start_frame, end_frame, tags}
    """
    tree = ET.parse(path)
    instances = []
    for inst in tree.iter("instance"):
        start_s = float(inst.findtext("start") or 0.0)
        end_s = float(inst.findtext("end") or start_s)
        code = inst.findtext("code") or ""
        tags = {
            lbl.findtext("group") or "": lbl.findtext("text") or ""
            for lbl in inst.findall("label")
        }
        instances.append({
            "code": code,
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": round(start_s * fps),
            "end_frame": round(end_s * fps),
            "tags": tags,
        })
    return instances


def encode_smart_tagging_features(instances, n_frames):
    """Encode Smart Tagging intervals into frame-level features.

    Features:
    - has_smart_tag: bool (is inside any tagged phase)
    - is_set_piece: bool (is inside set piece interval)
    - tag_team: float (0=Home, 1=Away, nan=None/NA)
    - dist_to_set_piece_start: float (frames to/from nearest set piece start)
    """
    has_tag = np.zeros(n_frames, dtype=np.float32)
    is_set_piece = np.zeros(n_frames, dtype=np.float32)
    tag_team = np.full(n_frames, np.nan, dtype=np.float32)
    min_dist_set_piece = np.full(n_frames, 9999.0, dtype=np.float32)

    set_piece_starts = []
    for inst in instances:
        s = max(0, inst["start_frame"])
        e = min(n_frames, inst["end_frame"])
        if s < e:
            has_tag[s:e] = 1.0
            code = inst["code"].strip().lower()
            if "set piece" in code:
                is_set_piece[s:e] = 1.0
                set_piece_starts.append(s)

            team_tag = inst["tags"].get("Team", "").strip().lower()
            if "barcelona" in team_tag or "home" in team_tag or "team a" in team_tag:
                tag_team[s:e] = 0.0
            elif "athletic" in team_tag or "away" in team_tag or "team b" in team_tag:
                tag_team[s:e] = 1.0

    if set_piece_starts:
        sp_arr = np.array(set_piece_starts, dtype=int)
        all_frames = np.arange(n_frames)
        # Distance to closest set piece start
        diffs = np.abs(all_frames[:, None] - sp_arr[None, :])
        min_dist_set_piece = np.min(diffs, axis=1).astype(np.float32)

    return pd.DataFrame({
        "has_smart_tag": has_tag,
        "is_set_piece": is_set_piece,
        "tag_team": tag_team,
        "dist_to_set_piece": np.log1p(min_dist_set_piece),
    })
