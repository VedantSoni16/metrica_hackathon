from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from games import GAME, half_video_blocks
from mtah.io.atd import parse_epts_metadata, parse_epts_tracking_file


def estimate_ball_size():
    homo_path = ROOT / "data/tactical/BAR-ATH_tactical_homography.csv"
    atd_txt = ROOT / "data/tactical/BAR-ATH_tactical_atd.txt"
    atd_xml = ROOT / "data/tactical/BAR-ATH_tactical_atd.xml"

    print("Tactical video resolution: 1920x1080 Full HD (centre: 965.7, 467.7 px).")

    # 2. Homography and tracking data
    meta = parse_epts_metadata(atd_xml)
    pitch_w, pitch_h = meta["pitch_size"]  # 105.3, 68.0

    homo_df = pd.read_csv(homo_path)
    # Pick 5 sampled graded frames where ball is tracked
    blocks = half_video_blocks(GAME)
    # Pick candidate frames in half 1 and half 2
    sample_candidate_frames = [
        blocks[0][1] + 1000,
        blocks[0][1] + 15000,
        blocks[0][1] + 20000,
        blocks[1][1] + 10000,
        blocks[1][1] + 40000,
    ]

    ball_diameter_m = 0.22  # FIFA regulation football diameter is 22 cm

    results = []
    # Read tracking for these 5 frames
    for vf in sample_candidate_frames:
        fid = vf + 1
        data = parse_epts_tracking_file(atd_txt, meta["slot_teams"], pitch_size=(pitch_w, pitch_h), frame_range=(fid, fid))
        if len(data["ball_xy"]) == 0 or np.isnan(data["ball_xy"][0, 0]):
            continue

        bx, by = data["ball_xy"][0]
        # Homography for frame vf
        h_row = homo_df.iloc[vf]
        if not h_row["valid"]:
            continue

        H = np.array([
            [h_row["h00"], h_row["h01"], h_row["h02"]],
            [h_row["h10"], h_row["h11"], h_row["h12"]],
            [h_row["h20"], h_row["h21"], h_row["h22"]],
        ])

        # Normalised pitch coords of ball center and bounding edges (+/- 11 cm)
        bx_norm = bx / pitch_w
        by_norm = by / pitch_h
        dx_norm = (ball_diameter_m / 2.0) / pitch_w
        dy_norm = (ball_diameter_m / 2.0) / pitch_h

        def project(x_n, y_n, mat=H):
            pt = mat @ np.array([x_n, y_n, 1.0])
            return pt[0] / pt[2], pt[1] / pt[2]

        cx, cy = project(bx_norm, by_norm)
        px1, py1 = project(bx_norm - dx_norm, by_norm)
        px2, py2 = project(bx_norm + dx_norm, by_norm)
        py_top_x, py_top_y = project(bx_norm, by_norm - dy_norm)
        py_bot_x, py_bot_y = project(bx_norm, by_norm + dy_norm)

        diam_x_px = np.hypot(px2 - px1, py2 - py1)
        diam_y_px = np.hypot(py_bot_x - py_top_x, py_bot_y - py_top_y)
        mean_diam_px = (diam_x_px + diam_y_px) / 2.0

        results.append({
            "video_frame": vf,
            "ball_pitch_xy_m": (round(float(bx), 2), round(float(by), 2)),
            "image_pixel_xy": (round(float(cx), 1), round(float(cy), 1)),
            "diam_x_px": round(float(diam_x_px), 2),
            "diam_y_px": round(float(diam_y_px), 2),
            "mean_diam_px": round(float(mean_diam_px), 2),
        })

    print(f"\nEstimated Ball Pixel Size across {len(results)} Sampled Frames:")
    print(f"{'Frame':<10} {'Pitch Pos (m)':<20} {'Image (px)':<20} {'Est Width (px)':<16} {'Est Height (px)':<16} {'Mean Diameter (px)':<18}")
    print("-" * 102)
    diams = []
    for r in results:
        diams.append(r["mean_diam_px"])
        print(f"{r['video_frame']:<10} {r['ball_pitch_xy_m']!s:<20} {r['image_pixel_xy']!s:<20} {r['diam_x_px']:<16} {r['diam_y_px']:<16} {r['mean_diam_px']:<18}")

    overall_mean = float(np.mean(diams)) if diams else 0.0
    print(f"\nOverall Average Projected Ball Diameter: {overall_mean:.2f} pixels.")
    return overall_mean, results


if __name__ == "__main__":
    estimate_ball_size()
