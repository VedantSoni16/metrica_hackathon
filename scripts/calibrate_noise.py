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


from games import GAME, half_video_blocks, require
from mtah.io.atd import parse_epts_metadata, parse_epts_tracking_file


def calibrate_from_arrays(ball_xy, player_h, player_a, fps=25.0):
    """Compute empirical noise and kinematic statistics from tracking arrays."""
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

    run_lengths = [int(e - s) for s, e in zip(starts, ends)]
    if run_lengths:
        mean_miss_run = float(np.mean(run_lengths))
        median_miss_run = float(np.median(run_lengths))
        p25_miss_run = float(np.percentile(run_lengths, 25))
        p75_miss_run = float(np.percentile(run_lengths, 75))
        p90_miss_run = float(np.percentile(run_lengths, 90))
        max_miss_run = int(np.max(run_lengths))
        short_gaps_frac = float(np.mean([r <= 20 for r in run_lengths]))
    else:
        mean_miss_run = 25.0
        median_miss_run = 20.0
        p25_miss_run = 10.0
        p75_miss_run = 40.0
        p90_miss_run = 80.0
        max_miss_run = 100
        short_gaps_frac = 0.5

    # Player visibility
    h_vis = ~np.isnan(player_h[:, :, 0])
    a_vis = ~np.isnan(player_a[:, :, 0])
    total_vis = np.sum(h_vis, axis=1) + np.sum(a_vis, axis=1)
    mean_active_players = float(np.mean(total_vis))
    mean_h_players = float(np.mean(np.sum(h_vis, axis=1)))
    mean_a_players = float(np.mean(np.sum(a_vis, axis=1)))

    # Kinematics & jitter estimation
    # Player jitter: standard finite difference estimate on contiguous 3-frame sequences
    # second difference d^2 x = x_{t+1} - 2x_t + x_{t-1} has variance 6 * sigma^2 for i.i.d. noise
    p_jitters = []
    p_speeds = []
    for p_arr in (player_h, player_a):
        for slot in range(p_arr.shape[1]):
            valid = ~np.isnan(p_arr[:, slot, 0])
            if np.sum(valid) > 10:
                coords = p_arr[valid, slot, :]
                d1 = np.diff(coords, axis=0) * fps
                speed = np.hypot(d1[:, 0], d1[:, 1])
                # filter out teleportation/slot re-assignment jumps
                plausible = speed < 12.0
                if np.any(plausible):
                    p_speeds.extend(speed[plausible][::10].tolist())
                # Second difference for jitter
                d2 = np.diff(coords, n=2, axis=0)
                plausible_d2 = np.hypot(d2[:, 0], d2[:, 1]) < 1.0
                if np.any(plausible_d2):
                    std_d2 = np.std(d2[plausible_d2])
                    p_jitters.append(float(std_d2 / np.sqrt(6.0)))

    player_jitter_m = float(np.median(p_jitters)) if p_jitters else 0.15
    player_speed_mean = float(np.mean(p_speeds)) if p_speeds else 2.5
    player_speed_p95 = float(np.percentile(p_speeds, 95)) if p_speeds else 7.0

    # Ball jitter & speeds
    valid_b = ~ball_missing
    ball_valid_coords = ball_xy[valid_b]
    b_speeds = []
    b_jitters = []
    if len(ball_valid_coords) > 10:
        d1_b = np.diff(ball_valid_coords, axis=0) * fps
        sp_b = np.hypot(d1_b[:, 0], d1_b[:, 1])
        plausible_b = sp_b < 35.0
        if np.any(plausible_b):
            b_speeds = sp_b[plausible_b].tolist()
        d2_b = np.diff(ball_valid_coords, n=2, axis=0)
        plausible_d2_b = np.hypot(d2_b[:, 0], d2_b[:, 1]) < 2.0
        if np.any(plausible_d2_b):
            std_d2_b = np.std(d2_b[plausible_d2_b])
            b_jitters.append(float(std_d2_b / np.sqrt(6.0)))

    ball_jitter_m = float(np.median(b_jitters)) if b_jitters else 0.25
    ball_speed_mean = float(np.mean(b_speeds)) if b_speeds else 6.0
    ball_speed_p95 = float(np.percentile(b_speeds, 95)) if b_speeds else 18.0

    return {
        "n_frames": T,
        "ball_missing_fraction": miss_frac,
        "mean_missing_run_len": mean_miss_run,
        "median_missing_run_len": median_miss_run,
        "p25_missing_run_len": p25_miss_run,
        "p75_missing_run_len": p75_miss_run,
        "p90_missing_run_len": p90_miss_run,
        "max_missing_run_len": max_miss_run,
        "short_gaps_frac_le_20": short_gaps_frac,
        "mean_active_players": mean_active_players,
        "mean_home_players": mean_h_players,
        "mean_away_players": mean_a_players,
        "player_jitter_m": player_jitter_m,
        "player_speed_mean_mps": player_speed_mean,
        "player_speed_p95_mps": player_speed_p95,
        "ball_jitter_m": ball_jitter_m,
        "ball_speed_mean_mps": ball_speed_mean,
        "ball_speed_p95_mps": ball_speed_p95,
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
        # Parse tracking on graded frames to isolate in-play match noise distribution
        blocks = half_video_blocks(GAME)
        # video_frame = frame_id - 1 => fid = video_frame + 1
        # Extract frames covering half 1 and half 2
        data_h1 = parse_epts_tracking_file(
            raw_path,
            meta["slot_teams"],
            pitch_size=meta["pitch_size"],
            frame_range=(blocks[0][1] + 1, blocks[0][2] + 1),
        )
        data_h2 = parse_epts_tracking_file(
            raw_path,
            meta["slot_teams"],
            pitch_size=meta["pitch_size"],
            frame_range=(blocks[1][1] + 1, blocks[1][2] + 1),
        )

        b_xy = np.vstack([data_h1["ball_xy"], data_h2["ball_xy"]])
        p_h = np.vstack([data_h1["player_h"], data_h2["player_h"]])
        p_a = np.vstack([data_h1["player_a"], data_h2["player_a"]])

        params = calibrate_from_arrays(b_xy, p_h, p_a)
        params["source"] = f"real_{GAME}_atd_graded"

    print("\nCalibrated Noise Parameters:")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved calibrated parameters to {out_json}")


if __name__ == "__main__":
    main()
