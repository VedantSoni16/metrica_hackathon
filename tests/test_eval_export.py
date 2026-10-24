"""Unit tests for evaluation wrappers, bootstrap, and submission export."""

import numpy as np
import pytest

from games import half_video_blocks, source_frames
from mtah.eval.bootstrap import evaluate_predictions, paired_bootstrap_diff
from mtah.eval.export import (
    compute_file_sha256,
    create_full_match_submission,
    save_and_validate_submission,
)


def test_evaluate_predictions_and_paired_bootstrap():
    # Construct synthetic ground truth: 3 distinct possessions
    # 50 frames H (0), 30 frames D (2), 50 frames A (1), 30 frames D (2), 50 frames H (0)
    gt = np.array([0]*50 + [2]*30 + [1]*50 + [2]*30 + [0]*50, dtype=np.int8)

    # Perfect prediction
    res_perfect = evaluate_predictions(gt, gt, bootstrap_n=50)
    assert res_perfect["spell_f1"] == pytest.approx(1.0)
    assert res_perfect["macro_f1"] == pytest.approx(1.0)
    assert res_perfect["n_gt_possessions"] == 3

    # Degraded prediction: flip team and add heavy chatter
    pred_bad = gt.copy()
    pred_bad[gt == 0] = 1  # wrong team
    pred_bad[gt == 1] = 0

    res_bad = evaluate_predictions(gt, pred_bad, bootstrap_n=0)
    assert res_bad["spell_f1"] < 0.1

    # Paired bootstrap test: Perfect vs Bad should have delta > 0 and be significant
    diff = paired_bootstrap_diff(gt, pred_bad, gt, n_draws=100, seed=42)
    assert diff["delta_mean"] > 0.8
    assert diff["ci_low"] > 0.5
    assert diff["is_significant"] is True


def test_submission_creation_and_validation(tmp_path):
    blocks = half_video_blocks("BAR-ATH")
    len_h1 = blocks[0][2] - blocks[0][1] + 1
    len_h2 = blocks[1][2] - blocks[1][1] + 1

    h1_preds = np.zeros(len_h1, dtype=np.int8)  # all H
    h2_preds = np.ones(len_h2, dtype=np.int8)   # all A

    df = create_full_match_submission(h1_preds, h2_preds, game="BAR-ATH")
    n_expected = source_frames("tactical", "BAR-ATH")
    assert len(df) == n_expected
    assert df["frame"].iloc[0] == 0
    assert df["frame"].iloc[-1] == n_expected - 1

    # Check ungraded frames are X
    assert df["label"].iloc[0] == "X"
    assert df["label"].iloc[blocks[0][1] - 1] == "X"
    assert df["label"].iloc[blocks[0][1]] == "H"
    assert df["label"].iloc[blocks[0][2]] == "H"
    assert df["label"].iloc[blocks[0][2] + 1] == "X"
    assert df["label"].iloc[blocks[1][1]] == "A"
    assert df["label"].iloc[blocks[1][2]] == "A"

    # Test saving and validation
    out_csv = tmp_path / "test_sub.csv"
    meta = save_and_validate_submission(df, out_csv, meta_extra={"test": True})
    assert meta["is_valid"] is True
    assert meta["n_rows"] == n_expected
    assert out_csv.exists()
    assert (tmp_path / "test_sub.json").exists()

    # Hash should be non-empty string of 64 hex characters
    assert len(meta["sha256"]) == 64
    assert meta["sha256"] == compute_file_sha256(out_csv)
