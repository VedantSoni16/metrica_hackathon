#!/usr/bin/env python3
"""Run entire end-to-end possession inference pipeline in one command.

Usage:
    python scripts/run_all.py              # runs on competition match data/
    python scripts/run_all.py --standin    # develops/tests on Metrica Game 3 stand-in
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames
from mtah.decode.postprocess import min_spell_merge
from mtah.decode.viterbi import build_log_transmat, scaled_log_emissions, viterbi_log
from mtah.eval.export import create_full_match_submission, save_and_validate_submission
from mtah.features.extractor import extract_frame_features
from mtah.healing.ball import heal_ball_trajectory
from mtah.io.atd import parse_epts_metadata, parse_epts_tracking_file
from mtah.io.metrica_csv import load_metrica_csv_tracking
from mtah.labels.metrica import derive_metrica_labels, load_metrica_events_csv
from mtah.models.baseline_model import PossessionClassifier
from mtah.sim.noise import apply_atd_noise


def preflight_check(game=GAME, data_dir=None):
    """Check whether required match assets exist under data/ and list missing ones in plain English."""
    paths = asset_paths(game=game, data_dir=data_dir, source="tactical")
    required_assets = ["atd_raw", "atd_meta", "homography", "smart_tagging"]

    missing = []
    for name in required_assets:
        p = Path(paths[name])
        if not p.is_file():
            missing.append(f"  - {name}: {p.name} (expected at {p.parent})")

    return missing, paths


def train_external_model():
    """Train Stage 3a model strictly on external data (Metrica Games) with simulated ATD noise."""
    print("[1/5] Training Stage 3a model on external Metrica data with simulated ATD noise...")
    g1_h = ROOT / "external_data/metrica_sample_data/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
    g1_a = ROOT / "external_data/metrica_sample_data/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
    g1_ev = ROOT / "external_data/metrica_sample_data/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv"

    if not g1_h.exists():
        raise FileNotFoundError(f"External training data not found at {g1_h}.")

    tr1 = load_metrica_csv_tracking(g1_h, g1_a, max_frames=40000)
    ev1 = load_metrica_events_csv(g1_ev)
    y_train = derive_metrica_labels(ev1, len(tr1["frame_ids"]))

    b_noisy, h_noisy, a_noisy = apply_atd_noise(
        tr1["ball_xy"], tr1["player_h"], tr1["player_a"], rng=np.random.default_rng(42)
    )
    healed = heal_ball_trajectory(b_noisy, max_gap=20)
    X_train = extract_frame_features(
        healed["ball_xy"],
        h_noisy,
        a_noisy,
        ball_imputed=healed["ball_imputed"],
        ball_missing=healed["ball_missing"],
        ball_missing_run_len=healed["ball_missing_run_len"],
    )

    clf = PossessionClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    print(f"      Model trained on {len(y_train)} frames across 46 features.")
    return clf


def run_standin_pipeline(clf, team_name="solution"):
    """Run full pipeline on Metrica Game 3 as a stand-in match."""
    print("[INFO] Running stand-in pipeline on Metrica Game 3 EPTS...")
    meta_path = ROOT / "external_data/metrica_sample_data/data/Sample_Game_3/Sample_Game_3_metadata.xml"
    raw_path = ROOT / "external_data/metrica_sample_data/data/Sample_Game_3/Sample_Game_3_tracking.txt"

    meta = parse_epts_metadata(meta_path)
    data = parse_epts_tracking_file(raw_path, meta["slot_teams"], pitch_size=meta["pitch_size"], max_frames=5000)

    healed = heal_ball_trajectory(data["ball_xy"], max_gap=20)
    X = extract_frame_features(
        healed["ball_xy"],
        data["player_h"],
        data["player_a"],
        ball_imputed=healed["ball_imputed"],
        ball_missing=healed["ball_missing"],
        ball_missing_run_len=healed["ball_missing_run_len"],
        pitch_size=meta["pitch_size"],
    )

    post = clf.predict_proba(X)
    prior = np.array([0.4, 0.4, 0.2])
    mean_dur = {"H": 350.0, "A": 350.0, "D": 200.0}
    leave_share = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    log_trans = build_log_transmat(mean_dur, leave_share)
    log_emit = scaled_log_emissions(post, prior, healed["ball_reliability"])
    pred_viterbi = viterbi_log(log_emit, log_trans)
    pred_final = min_spell_merge(pred_viterbi, post, min_len=12)

    print(f"      Stand-in decoded: {len(pred_final)} frames.")
    print("      Stand-in test successfully completed!")


def run_real_pipeline(clf, paths, team_name="solution"):
    """Execute complete S0-S7 pipeline on the competition match."""
    print("[2/5] Ingesting and aligning match tracking data (S0)...")
    meta = parse_epts_metadata(paths["atd_meta"])
    # Pitch size from games.py: (105.3, 68.0)
    pitch_size = GAMES[GAME]["pitch"]

    data = parse_epts_tracking_file(paths["atd_raw"], meta["slot_teams"], pitch_size=pitch_size)
    n_total = source_frames("tactical", GAME)
    print(f"      Loaded tracking: {len(data['frame_ids'])} frames (total expected: {n_total}).")

    blocks = half_video_blocks(GAME)
    half_predictions = []

    for half, first, last in blocks:
        span_len = last - first + 1
        print(f"[3/5] Processing Half {half} ({first}..{last}, {span_len} frames)...")

        # Extract frames for this half
        # In ATD: video_frame = frame_id - 1 => frame_id = video_frame + 1
        h_slice = slice(first, last + 1)
        b_xy = data["ball_xy"][h_slice]
        p_h = data["player_h"][h_slice]
        p_a = data["player_a"][h_slice]

        # S1: Ball healing
        healed = heal_ball_trajectory(b_xy, max_gap=20)

        # S2: Features
        X_half = extract_frame_features(
            healed["ball_xy"],
            p_h,
            p_a,
            ball_imputed=healed["ball_imputed"],
            ball_missing=healed["ball_missing"],
            ball_missing_run_len=healed["ball_missing_run_len"],
            pitch_size=pitch_size,
        )

        # S3: Frame-level probabilities
        post = clf.predict_proba(X_half)

        # S4: Structured decoder per half
        prior = np.array([0.4, 0.4, 0.2])
        mean_dur = {"H": 350.0, "A": 350.0, "D": 200.0}
        leave_share = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        log_trans = build_log_transmat(mean_dur, leave_share)
        log_emit = scaled_log_emissions(post, prior, healed["ball_reliability"])
        decoded_path = viterbi_log(log_emit, log_trans)

        # S5: Post-processing min-spell merge
        smoothed_path = min_spell_merge(decoded_path, post, min_len=12)
        half_predictions.append(smoothed_path)

    print("[4/5] Assembling full 199,319-frame submission and validating (S7)...")
    sub_df = create_full_match_submission(half_predictions[0], half_predictions[1], game=GAME)

    out_csv = ROOT / f"submissions/{team_name}.csv"
    meta_info = save_and_validate_submission(
        sub_df,
        out_csv,
        meta_extra={"method": "Stage 3a LightGBM + 3-state log-Viterbi + min_spell_merge"},
        game=GAME,
    )
    print(f"      Submission saved: {out_csv}")
    print(f"      SHA-256 Hash:     {meta_info['sha256']}")
    print(f"      Rows:             {meta_info['n_rows']}")
    print(f"      Counts:           {meta_info['label_counts']}")

    # Built-in run-twice check for determinism (LightGBM num_threads=1, deterministic=True, fixed seeds)
    print("[Deterministic Check] Re-running inference pass 2 to verify byte-identical output...")
    pass2_halves = []
    for half, first, last in blocks:
        h_slice = slice(first, last + 1)
        b_xy = data["ball_xy"][h_slice]
        p_h = data["player_h"][h_slice]
        p_a = data["player_a"][h_slice]
        healed_2 = heal_ball_trajectory(b_xy, max_gap=20)
        X_half_2 = extract_frame_features(
            healed_2["ball_xy"],
            p_h,
            p_a,
            ball_imputed=healed_2["ball_imputed"],
            ball_missing=healed_2["ball_missing"],
            ball_missing_run_len=healed_2["ball_missing_run_len"],
            pitch_size=pitch_size,
        )
        post_2 = clf.predict_proba(X_half_2)
        log_emit_2 = scaled_log_emissions(post_2, prior, healed_2["ball_reliability"])
        decoded_path_2 = viterbi_log(log_emit_2, log_trans)
        smoothed_path_2 = min_spell_merge(decoded_path_2, post_2, min_len=12)
        pass2_halves.append(smoothed_path_2)

    sub_df_2 = create_full_match_submission(pass2_halves[0], pass2_halves[1], game=GAME)
    import hashlib
    sha1 = meta_info["sha256"]
    sha2 = hashlib.sha256(sub_df_2.to_csv(index=False).encode("utf-8")).hexdigest()
    if sha1 != sha2:
        raise RuntimeError(f"DETERMINISM CHECK FAILED: Pass 1 ({sha1}) != Pass 2 ({sha2})")
    print(f"      [OK] Deterministic check passed: SHA-256 matches exactly across runs ({sha1[:16]}...).")

    # Run official validator
    print("[5/5] Running official validate_submission.py check...")
    validator = ROOT / "evaluation/validate_submission.py"
    res = subprocess.run([sys.executable, str(validator), str(out_csv)], check=False, capture_output=True, text=True)
    print(res.stdout)


def main():
    parser = argparse.ArgumentParser(description="End-to-end MTAH-2027 solution pipeline")
    parser.add_argument(
        "--team-name", "--team",
        dest="team_name",
        default="Yamal Fanboys",
        help="Team name for submission CSV (default: Yamal Fanboys)",
    )
    parser.add_argument("--standin", action="store_true", help="Run on Metrica Game 3 stand-in")
    parser.add_argument("--data-dir", default=None, help="Match data directory")
    args = parser.parse_args()

    print("=====================================================================")
    print("  More Than A Hack 2027: Ball Possession Solution Pipeline")
    print("=====================================================================")

    # Preflight Check - requires only data/tactical/
    missing_files, paths = preflight_check(game=GAME, data_dir=args.data_dir)

    if missing_files and not args.standin:
        print("\n[PREFLIGHT CHECK FAILED] Missing match data files:")
        for m in missing_files:
            print(m)
        print("\nMatch data is gated and not yet in data/ directory.")
        print("To run the full pipeline on real data:")
        print("  1. Unzip the competition bundle into ./data/tactical/")
        print("  2. Re-run:  python scripts/run_all.py")
        print("\nTo test the end-to-end pipeline right now on external stand-in data:")
        print("  python scripts/run_all.py --standin\n")
        return

    # Train model on external data
    clf = train_external_model()

    if args.standin or missing_files:
        run_standin_pipeline(clf, team_name=args.team_name)
    else:
        run_real_pipeline(clf, paths, team_name=args.team_name)


if __name__ == "__main__":
    main()
