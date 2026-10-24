#!/usr/bin/env python3
"""Calibrate ATD noise simulator parameters against match tracking data.

Auto-skips gracefully if match data is absent, or runs on Metrica Game 3 as stand-in.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from games import GAME, require
from mtah.io.atd import parse_epts_metadata, parse_epts_tracking_file


def calibrate_from_arrays(ball_xy, player_h, player_a):
    """Compute empirical noise statistics from tracking arrays."""
    T = len(ball_xy)
    ball_missing = np.isnan(ball_xy[:, 0]) | np.isnan(ball_xy[:, 1])
    miss_frac = float(np.mean(ball_missing))

    # Missing run lengths
    diff = np.diff(ball_missing.astype(int))
    starts = np.where(diff == 1)[0] + 1
    if ball_missing[0]:
        starts = np.r_[0, starts]
    ends = np.where(diff == -1)[0] + 1
    if ball_missing[-1]:
        ends = np.r_[ends, T]

    run_lengths = [e - s for s, e in zip(starts, ends)]
    mean_miss_run = float(np.mean(run_lengths)) if run_lengths else 25.0
    median_miss_run = float(np.median(run_lengths)) if run_lengths else 20.0

    # Player visibility
    h_vis = ~np.isnan(player_h[:, :, 0])
    a_vis = ~np.isnan(player_a[:, :, 0])
    total_vis = np.sum(h_vis, axis=1) + np.sum(a_vis, axis=1)
    mean_active_players = float(np.mean(total_vis))

    return {
        "n_frames": T,
        "ball_missing_fraction": miss_frac,
        "mean_missing_run_len": mean_miss_run,
        "median_missing_run_len": median_miss_run,
        "mean_active_players": mean_active_players,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate noise simulator from tracking data")
    parser.add_argument("--standin", action="store_true", help="Run on Metrica Game 3 stand-in")
    parser.add_argument("--data-dir", default=None, help="Path to match data directory")
    args = parser.parse_args()

    out_dir = ROOT / "configs"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "noise_params.json"

    if args.standin:
        standin_txt = ROOT / "external_data/metrica_sample_data/data/Sample_Game_3/Sample_Game_3_tracking.txt"
        standin_xml = ROOT / "external_data/metrica_sample_data/data/Sample_Game_3/Sample_Game_3_metadata.xml"
        if not standin_txt.exists():
            print(f"[SKIP] Stand-in data not found at {standin_txt}.")
            return

        print(f"[INFO] Calibrating noise simulator using stand-in: {standin_txt.name}...")
        meta = parse_epts_metadata(standin_xml)
        data = parse_epts_tracking_file(standin_txt, meta["slot_teams"], pitch_size=meta["pitch_size"], max_frames=20000)
        params = calibrate_from_arrays(data["ball_xy"], data["player_h"], data["player_a"])
        params["source"] = "metrica_game_3_standin"
    else:
        try:
            raw_path = require(["atd_raw"], game=GAME, data_dir=args.data_dir)[0]
            meta_path = require(["atd_meta"], game=GAME, data_dir=args.data_dir)[0]
        except (OSError, KeyError):
            print(f"[SKIP] Real match data not found for {GAME} (gated match assets not yet downloaded).")
            print("       Pre-flight auto-skip enabled. Run with --standin to calibrate on external data.")
            return

        print(f"[INFO] Calibrating noise simulator from real ATD: {raw_path}...")
        meta = parse_epts_metadata(meta_path)
        data = parse_epts_tracking_file(raw_path, meta["slot_teams"], pitch_size=meta["pitch_size"])
        params = calibrate_from_arrays(data["ball_xy"], data["player_h"], data["player_a"])
        params["source"] = f"real_{GAME}_atd"

    print("Calibrated Noise Parameters:")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open(out_json, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved calibrated parameters to {out_json}")


if __name__ == "__main__":
    main()
