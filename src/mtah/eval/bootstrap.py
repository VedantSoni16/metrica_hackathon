"""Bootstrap evaluation and model comparison utilities wrapping possession_metrics."""

import sys
from pathlib import Path

import numpy as np

# Ensure evaluation module is importable
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(ROOT / "evaluation"))

import possession_metrics as pm


def evaluate_predictions(gt_codes, pred_codes, bootstrap_n=0, seed=0):
    """Run full evaluation using the official possession_metrics scorer.

    Parameters
    ----------
    gt_codes : np.ndarray
        Ground truth integer codes (0=H, 1=A, 2=D, 3=X).
    pred_codes : np.ndarray
        Predicted integer codes (0=H, 1=A, 2=D, 3=X).
    bootstrap_n : int
        Number of bootstrap iterations (e.g. 1000). 0 for point estimates only.
    seed : int
        Random seed for bootstrap.

    Returns
    -------
    dict
        Dictionary containing:
        - spell_f1: float (ranking metric)
        - spell_f1_ci: (low, high) or (None, None)
        - macro_f1: float (tie-break metric)
        - macro_f1_ci: (low, high) or (None, None)
        - n_gt_possessions: int
        - n_pred_possessions: int
        - mean_duration_s: float
        - raw_eval: full dict from pm.evaluate()
    """
    res = pm.evaluate(gt_codes, pred_codes, bootstrap_n=bootstrap_n)
    spell_f1 = float(res["spell_f1_mean"])
    spell_ci = (res["spell_bootstrap_low"], res["spell_bootstrap_high"])
    macro_f1 = float(res["macro_f1"])
    macro_ci = (res["bootstrap_low"], res["bootstrap_high"])

    return {
        "spell_f1": spell_f1,
        "spell_f1_ci": spell_ci,
        "macro_f1": macro_f1,
        "macro_f1_ci": macro_ci,
        "n_gt_possessions": int(res["n_gt_spells"]),
        "n_pred_possessions": int(res["n_pred_spells"]),
        "mean_duration_s": float(res["mean_spell_duration_s"]),
        "raw_eval": res,
    }



def paired_bootstrap_diff(gt_codes, pred_a, pred_b, n_draws=1000, seed=0, percentiles=(5, 95)):
    """Compute paired bootstrap difference in spellF1 between two models: Δ = B - A.

    Decision rule: Adopt change B only if the confidence interval for ΔspellF1
    excludes 0 (i.e. low > 0).

    Parameters
    ----------
    gt_codes : np.ndarray
        Ground truth codes.
    pred_a : np.ndarray
        Baseline model predictions.
    pred_b : np.ndarray
        New model predictions.
    n_draws : int
        Number of bootstrap draws.
    seed : int
        Random seed.
    percentiles : tuple of (float, float)
        Confidence interval percentiles (default 5, 95 for 90% CI).

    Returns
    -------
    dict
        {
            "delta_mean": float,
            "ci_low": float,
            "ci_high": float,
            "is_significant": bool (ci_low > 0)
        }
    """
    runs = pm.valid_runs(gt_codes)
    gt_spells = pm.possession_spells(gt_codes, runs)
    if not gt_spells:
        return {"delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "is_significant": False}

    spells_a = pm.possession_spells(pred_a, runs)
    spells_b = pm.possession_spells(pred_b, runs)

    overlaps_a = pm.spell_overlaps(gt_spells, spells_a)
    overlaps_b = pm.spell_overlaps(gt_spells, spells_b)

    n_gt = len(gt_spells)
    n_a, n_b = len(spells_a), len(spells_b)

    flags_a = np.vstack([pm.matched_gt_flags(overlaps_a, n_gt, t) for t in pm.SPELL_IOUS])
    flags_b = np.vstack([pm.matched_gt_flags(overlaps_b, n_gt, t) for t in pm.SPELL_IOUS])

    rng = np.random.RandomState(seed)
    deltas = np.empty(n_draws, dtype=np.float64)

    for draw in range(n_draws):
        picks = rng.randint(0, n_gt, n_gt)

        # Score A on picks
        f1s_a = []
        for row in flags_a:
            tp = int(row[picks].sum())
            fn = n_gt - tp
            fp = max(0, n_a - tp)
            f1s_a.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
        score_a = np.mean(f1s_a)

        # Score B on picks
        f1s_b = []
        for row in flags_b:
            tp = int(row[picks].sum())
            fn = n_gt - tp
            fp = max(0, n_b - tp)
            f1s_b.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
        score_b = np.mean(f1s_b)

        deltas[draw] = score_b - score_a

    ci_low, ci_high = np.percentile(deltas, percentiles)
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "is_significant": bool(ci_low > 0.0),
    }
