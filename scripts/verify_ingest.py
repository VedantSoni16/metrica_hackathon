#!/usr/bin/env python3
"""Verify ingestion of the competition match data and produce reports/ingest_report.md."""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from games import GAME, half_video_blocks, source_frames


def run_verification():
    report_lines = []
    report_lines.append("# Ingest Verification Report: BAR-ATH Tactical")
    report_lines.append("")

    # 1. ATD Metadata check
    xml_path = ROOT / "data/tactical/BAR-ATH_tactical_atd.xml"
    root = ET.parse(xml_path).getroot()
    teams = {t.get("id"): t.findtext("Name") for t in root.iter("Team")}
    team_ids = list(teams.keys())
    home_id, away_id = team_ids[0], team_ids[1]
    home_name, away_name = teams[home_id], teams[away_id]

    report_lines.append("## 1. Team Mapping & Metadata")
    report_lines.append(f"- **Home Team ID:** `{home_id}` ({home_name})")
    report_lines.append(f"- **Away Team ID:** `{away_id}` ({away_name})")
    home_match = (home_name == "Barcelona")
    report_lines.append(f"- **Home = Barcelona Mapping:** {'MATCH [OK]' if home_match else 'MISMATCH [FAIL]'}")

    # Field dimensions
    field_w = root.findtext(".//FieldSize/Width")
    field_h = root.findtext(".//FieldSize/Height")
    fps_val = root.findtext(".//FrameRate")
    report_lines.append(f"- **Metadata Pitch Dimensions:** {field_w}m x {field_h}m (games.py: 105.3m x 68.0m)")
    report_lines.append(f"- **Metadata FrameRate:** {fps_val} fps")
    report_lines.append("")

    # 2. Tracking File & Graded Blocks Check
    txt_path = ROOT / "data/tactical/BAR-ATH_tactical_atd.txt"
    blocks = half_video_blocks(GAME)
    # Graded set of frame_ids: video_frame = frame_id - 1 => frame_id = video_frame + 1
    graded_fids = set()
    for _, first, last in blocks:
        for vf in range(first, last + 1):
            graded_fids.add(vf + 1)

    total_graded = len(graded_fids)
    expected_frames = source_frames("tactical", GAME)

    report_lines.append("## 2. Tracking Alignment & Graded Blocks")
    report_lines.append(f"- **Expected Total Frames:** {expected_frames}")
    for half, first, last in blocks:
        report_lines.append(f"- **Half {half} Graded Block:** video {first}..{last} ({last - first + 1} frames)")
    report_lines.append(f"- **Total Graded Frames:** {total_graded} (expected: 158,857)")

    # Streaming pass over tracking file
    line_count = 0
    first_fid = None
    last_fid = None
    is_contiguous = True

    graded_ball_missing = 0
    graded_tracked_objects_sum = 0
    graded_frames_seen = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            # Format: fid:players:ball
            idx_colon1 = line.find(":")
            if idx_colon1 == -1:
                continue
            fid = int(line[:idx_colon1])
            if first_fid is None:
                first_fid = fid
            if last_fid is not None and fid != last_fid + 1:
                is_contiguous = False
            last_fid = fid

            if fid in graded_fids:
                graded_frames_seen += 1
                # Parse players and ball
                idx_colon2 = line.find(":", idx_colon1 + 1)
                players_part = line[idx_colon1 + 1:idx_colon2]
                ball_part = line[idx_colon2 + 1:].strip()

                # Ball check
                b_coords = ball_part.split(",")
                b_valid = len(b_coords) >= 2 and b_coords[0] != "NaN" and b_coords[1] != "NaN"
                if not b_valid:
                    graded_ball_missing += 1

                # Tracked objects: players + ball
                p_count = 0
                for p_str in players_part.split(";"):
                    if p_str and not p_str.startswith("NaN,NaN"):
                        p_count += 1
                graded_tracked_objects_sum += (p_count + (1 if b_valid else 0))
                graded_players_sum = getattr(run_verification, "_p_sum", 0) + p_count
                run_verification._p_sum = graded_players_sum

    mean_players = run_verification._p_sum / total_graded if total_graded else 0.0
    ball_missing_pct = (graded_ball_missing / total_graded) * 100.0 if total_graded else 0.0
    mean_objects = (graded_tracked_objects_sum / total_graded) if total_graded else 0.0

    report_lines.append(f"- **Actual Tracking Records:** {line_count}")
    report_lines.append(f"- **Frame ID Range:** {first_fid}..{last_fid} (contiguous: {is_contiguous})")
    report_lines.append(f"- **Alignment (video_frame = frame_id - 1):** {'MATCH [OK]' if (first_fid == 1 and line_count == expected_frames) else 'MISMATCH [FAIL]'}")
    report_lines.append(f"- **Ball Missing on Graded Frames:** {graded_ball_missing} / {total_graded} ({ball_missing_pct:.2f}%, expected ~41%)")
    report_lines.append(f"- **Mean Tracked Players on Graded Frames:** {mean_players:.2f} (expected ~20.5)")
    report_lines.append(f"- **Mean Tracked Objects (Players + Ball) on Graded Frames:** {mean_objects:.2f} (21-22 during play)")
    report_lines.append("")

    # 3. Homography Check
    homo_path = ROOT / "data/tactical/BAR-ATH_tactical_homography.csv"
    homo_rows = 0
    homo_valid_count = 0
    with open(homo_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            homo_rows += 1
            if r["valid"] in ("1", "True", "true"):
                homo_valid_count += 1

    homo_valid_pct = (homo_valid_count / homo_rows * 100.0) if homo_rows else 0.0
    report_lines.append("## 3. Homography Validity")
    report_lines.append(f"- **Homography Rows:** {homo_rows} (expected {expected_frames})")
    report_lines.append(f"- **Valid Homographies:** {homo_valid_count} / {homo_rows} ({homo_valid_pct:.2f}%, expected 100.0%)")
    report_lines.append("")

    # 4. Smart Tagging Check
    st_path = ROOT / "data/tactical/BAR-ATH_tactical_smart_tagging.xml"
    st_root = ET.parse(st_path).getroot()
    instances = list(st_root.iter("instance"))
    st_count = len(instances)

    team_tags = {}
    code_counts = {}
    for inst in instances:
        code = inst.findtext("code") or "Unknown"
        code_counts[code] = code_counts.get(code, 0) + 1
        t_val = None
        for label in inst.iter("label"):
            text = label.findtext("text")
            group = label.findtext("group")
            if group == "Team":
                t_val = text
        t_key = t_val if t_val is not None else "N/A"
        team_tags[t_key] = team_tags.get(t_key, 0) + 1

    report_lines.append("## 4. Smart Tagging Summary")
    report_lines.append(f"- **Instance Count:** {st_count} (expected 638)")
    report_lines.append("- **Team Tag Breakdown:**")
    for t_name, count in sorted(team_tags.items(), key=lambda x: -x[1]):
        report_lines.append(f"  - `{t_name}`: {count}")
    report_lines.append("- **Top Event Codes:**")
    for c_name, count in sorted(code_counts.items(), key=lambda x: -x[1])[:8]:
        report_lines.append(f"  - `{c_name}`: {count}")
    report_lines.append("")

    # 5. Conclusion & Verification Summary
    report_lines.append("## 5. Verification Verdict")
    checks = [
        ("Frame count is 199,319", line_count == 199319),
        ("Frame alignment is frame_id - 1", first_fid == 1 and line_count == expected_frames),
        ("Graded blocks match games.py", total_graded == 158857),
        ("Homography 100% valid on tactical", homo_valid_pct == 100.0),
        ("Home is Barcelona", home_name == "Barcelona"),
        ("Ball missing ~41%", 40.0 <= ball_missing_pct <= 42.0),
        ("Mean tracked players ~20.5 (20.49)", 20.0 <= mean_players <= 21.0),
        ("Smart Tagging instances 638", st_count == 638),
    ]
    all_ok = True
    for desc, ok in checks:
        status = "PASSED" if ok else "FAILED"
        if not ok:
            all_ok = False
        report_lines.append(f"- [{ 'x' if ok else ' ' }] {desc}: **{status}**")

    report_lines.append("")
    report_lines.append(f"**Overall Status:** {'ALL INGEST CHECKS PASSED [OK]' if all_ok else 'CHECKS FAILED'}")

    report_content = "\n".join(report_lines) + "\n"

    out_report = ROOT / "reports/ingest_report.md"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report_content, encoding="utf-8")
    print(f"Ingest verification complete. Wrote report to {out_report}")
    print("\n" + report_content)


if __name__ == "__main__":
    run_verification()
