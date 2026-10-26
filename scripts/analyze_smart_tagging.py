#!/usr/bin/env python3
"""Infer meaning of Smart Tagging Team tag per event code via self-supervised nearest-ball agreement.

Never uses ground truth (strict zero-leakage rule).
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mtah.io.atd import parse_epts_metadata, parse_epts_tracking_file


def analyze_smart_tagging():
    xml_meta = ROOT / "data/tactical/BAR-ATH_tactical_atd.xml"
    txt_raw = ROOT / "data/tactical/BAR-ATH_tactical_atd.txt"
    st_xml = ROOT / "data/tactical/BAR-ATH_tactical_smart_tagging.xml"

    meta = parse_epts_metadata(xml_meta)
    print(f"ATD metadata: Home={meta['home_name']} (id={meta['home_id']}), Away={meta['away_name']} (id={meta['away_id']})")

    # Load tracking
    tracking = parse_epts_tracking_file(txt_raw, meta["slot_teams"], pitch_size=meta["pitch_size"])
    ball_xy = tracking["ball_xy"]
    p_h = tracking["player_h"]
    p_a = tracking["player_a"]
    T = len(ball_xy)

    # Compute frame-by-frame nearest player team (0=Home, 1=Away, -1=NaN)
    valid_ball = ~(np.isnan(ball_xy[:, 0]) | np.isnan(ball_xy[:, 1]))
    diff_h = p_h - ball_xy[:, None, :]
    dist_h = np.nanmin(np.hypot(diff_h[:, :, 0], diff_h[:, :, 1]), axis=1)
    diff_a = p_a - ball_xy[:, None, :]
    dist_a = np.nanmin(np.hypot(diff_a[:, :, 0], diff_a[:, :, 1]), axis=1)

    nearest_team = np.full(T, -1, dtype=np.int8)
    valid_both = valid_ball & ~np.isnan(dist_h) & ~np.isnan(dist_a)
    nearest_team[valid_both] = np.where(dist_h[valid_both] <= dist_a[valid_both], 0, 1)

    # Parse Smart Tagging instances
    st_root = ET.parse(st_xml).getroot()
    events = []
    fps = 25.0

    for inst in st_root.iter("instance"):
        code = inst.findtext("code")
        start_s = float(inst.findtext("start"))
        end_s = float(inst.findtext("end"))
        sf = round(start_s * fps)
        ef = round(end_s * fps)

        team_val = None
        for label in inst.iter("label"):
            text = label.findtext("text")
            group = label.findtext("group")
            if group == "Team":
                team_val = text

        events.append({
            "code": code,
            "start_frame": sf,
            "end_frame": ef,
            "team": team_val,
        })

    print(f"Loaded {len(events)} Smart Tagging events.")

    # Group by code and measure self-supervised agreement
    # Agreement: when event has Team=="Barcelona" (Home), does nearest_team == 0?
    # when event has Team=="Atletic Club" (Away), does nearest_team == 1?
    code_stats = {}
    for ev in events:
        c = ev["code"]
        team_str = ev["team"]
        if c not in code_stats:
            code_stats[c] = {"n_events": 0, "n_frames_valid": 0, "same_team_frames": 0, "opp_team_frames": 0, "team_counts": {}}
        code_stats[c]["n_events"] += 1
        code_stats[c]["team_counts"][team_str] = code_stats[c]["team_counts"].get(team_str, 0) + 1

        if team_str in ("Barcelona", "Atletic Club"):
            tagged_side = 0 if team_str == "Barcelona" else 1
            opp_side = 1 - tagged_side

            # span frames (clipped to T)
            sf = max(0, min(T - 1, ev["start_frame"]))
            ef = max(0, min(T - 1, ev["end_frame"]))
            if ef >= sf:
                sub_nearest = nearest_team[sf:ef + 1]
                v_frames = sub_nearest[sub_nearest >= 0]
                if len(v_frames) > 0:
                    same = int(np.sum(v_frames == tagged_side))
                    opp = int(np.sum(v_frames == opp_side))
                    code_stats[c]["n_frames_valid"] += len(v_frames)
                    code_stats[c]["same_team_frames"] += same
                    code_stats[c]["opp_team_frames"] += opp

    print("\nSmart Tagging Code Agreement with Nearest Player (Self-Supervised):")
    print(f"{'Code':<32} {'Events':>6} {'ValidFrames':>12} {'TaggedTeam%':>12} {'OppTeam%':>10} {'Inferred Meaning':<20}")
    print("-" * 96)

    for c, s in sorted(code_stats.items(), key=lambda x: -x[1]["n_events"]):
        tot = s["n_frames_valid"]
        if tot > 0:
            same_pct = (s["same_team_frames"] / tot) * 100.0
            opp_pct = (s["opp_team_frames"] / tot) * 100.0
            if same_pct >= 65.0:
                meaning = "Attacking/In-Poss"
            elif opp_pct >= 65.0:
                meaning = "Defending/Opp-Poss"
            else:
                meaning = "Contested/Mixed"
        else:
            same_pct = 0.0
            opp_pct = 0.0
            meaning = "No tracked ball"

        print(f"{c:<32} {s['n_events']:>6} {tot:>12} {same_pct:>11.1f}% {opp_pct:>9.1f}% {meaning:<20}")

    return code_stats


if __name__ == "__main__":
    analyze_smart_tagging()
