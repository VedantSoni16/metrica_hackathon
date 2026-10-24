"""Submission exporter and validation wrapper."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(ROOT / "evaluation"))

import possession_metrics as pm

from games import GAME, half_video_blocks, source_frames

CODE_TO_LABEL = {0: "H", 1: "A", 2: "D", 3: "X"}
LABEL_TO_CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def get_git_commit():
    """Retrieve current git commit SHA."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def compute_file_sha256(path):
    """Compute SHA-256 hash of a file for determinism checking."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_full_match_submission(half1_labels, half2_labels, game=GAME):
    """Assemble half-level predictions into the official 199,319-row submission.

    Ungraded frames (before kickoff, halftime, after whistle) receive 'X'.
    Graded frames receive 'H', 'A', or 'D'.

    Parameters
    ----------
    half1_labels : np.ndarray or list
        Labels for Half 1 (length must match Half 1 graded span).
    half2_labels : np.ndarray or list
        Labels for Half 2 (length must match Half 2 graded span).
    game : str
        Game identifier in games.GAMES.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['frame', 'label'].
    """
    n_frames = source_frames("tactical", game)
    blocks = half_video_blocks(game)

    full_labels = np.full(n_frames, "X", dtype=object)

    for (half, first, last), half_preds in zip(blocks, [half1_labels, half2_labels]):
        expected_len = last - first + 1
        if len(half_preds) != expected_len:
            raise ValueError(
                f"Half {half} expects {expected_len} frames ({first}..{last}), got {len(half_preds)}"
            )
        # Convert integers if provided
        str_preds = [
            CODE_TO_LABEL.get(p, p) if isinstance(p, (int, np.integer)) else str(p).upper()
            for p in half_preds
        ]
        full_labels[first : last + 1] = str_preds

    df = pd.DataFrame({"frame": np.arange(n_frames, dtype=int), "label": full_labels})
    return df


def save_and_validate_submission(df, out_csv_path, meta_extra=None, game=GAME):
    """Save submission DataFrame to CSV, run official validator, and write provenance JSON.

    Parameters
    ----------
    df : pd.DataFrame
        Submission DataFrame with 'frame' and 'label'.
    out_csv_path : str or Path
        Destination CSV path.
    meta_extra : dict, optional
        Extra metadata (config, seeds, CV scores) to log in JSON provenance.
    game : str
        Game identifier.

    Returns
    -------
    dict
        Provenance summary including sha256, validation status, label counts.
    """
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV without index
    df.to_csv(out_csv_path, index=False)

    # Validate against expected rows
    n_expected = source_frames("tactical", game)
    parsed_table = pm.read_label_csv(str(out_csv_path))
    reason = pm.validate_submission(parsed_table, n_expected)

    if reason is not None:
        raise ValueError(f"Submission validation FAILED for {out_csv_path}: {reason}")

    file_hash = compute_file_sha256(out_csv_path)
    counts = df["label"].value_counts().to_dict()

    meta = {
        "submission_file": out_csv_path.name,
        "sha256": file_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "n_rows": len(df),
        "label_counts": counts,
        "is_valid": True,
        "extra": meta_extra or {},
    }

    meta_path = out_csv_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta
