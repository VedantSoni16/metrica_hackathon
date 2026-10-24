"""Synthetic posterior decoder harness.

Validates the Viterbi decoder against raw argmax across a range of noise settings,
burst errors, and blind dropouts without touching any match ground truth.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(ROOT / "evaluation"))

import possession_metrics as pm

from mtah.decode.postprocess import min_spell_merge
from mtah.decode.viterbi import build_log_transmat, scaled_log_emissions, viterbi_log


def generate_synthetic_ground_truth(
    n_possessions=30,
    mean_poss_frames=350,
    mean_dead_frames=200,
    seed=42,
):
    """Generate realistic sequence of H, A, D labels.

    Alternates between possession spells (H or A) and dead-ball spells (D),
    with lognormal spell durations mirroring real football possessions.

    Returns
    -------
    gt : np.ndarray
        1D int8 array of states in {0, 1, 2}.
    """
    rng = np.random.default_rng(seed)
    spells = []
    current_team = 0  # start with H

    for _ in range(n_possessions):
        # Possession spell (lognormal duration)
        poss_len = int(np.clip(rng.lognormal(mean=np.log(mean_poss_frames) - 0.2, sigma=0.6), 25, 1500))
        spells.append(np.full(poss_len, current_team, dtype=np.int8))

        # Dead ball spell (70% probability between possessions)
        if rng.random() < 0.7:
            dead_len = int(np.clip(rng.lognormal(mean=np.log(mean_dead_frames) - 0.2, sigma=0.5), 15, 800))
            spells.append(np.full(dead_len, 2, dtype=np.int8))

        # Next possession: either turnover to other team (70%) or same team keeps (30%)
        if rng.random() < 0.7:
            current_team = 1 - current_team

    return np.concatenate(spells)


def corrupt_posteriors(
    gt,
    noise_level=0.3,
    burst_prob=0.005,
    burst_len_range=(20, 80),
    blind_prob=0.01,
    blind_len_range=(20, 60),
    seed=42,
):
    """Simulate noisy model posteriors and reliability weights from ground truth.

    Parameters
    ----------
    gt : np.ndarray
        (T,) integer ground truth in {0, 1, 2}.
    noise_level : float
        Dirichlet concentration or Gaussian noise magnitude.
    burst_prob : float
        Per-frame probability of triggering a wrong-class burst error (flicker).
    burst_len_range : tuple of (int, int)
        Min and max length of a burst error.
    blind_prob : float
        Per-frame probability of triggering a blind spell (flat posterior, low w).
    blind_len_range : tuple of (int, int)
        Min and max length of a blind spell.
    seed : int
        RNG seed.

    Returns
    -------
    post : np.ndarray
        (T, 3) simulated frame posteriors summing to 1.
    w : np.ndarray
        (T,) reliability weights in [0.1, 1.0].
    """
    rng = np.random.default_rng(seed)
    T = len(gt)
    if T == 0:
        return np.empty((0, 3)), np.empty(0)

    # Base posteriors: biased towards true class
    one_hot = np.zeros((T, 3), dtype=float)
    for c in (0, 1, 2):
        one_hot[gt == c, c] = 1.0

    # Add Dirichlet noise
    alpha = np.maximum(0.1, (1.0 - noise_level) * 10.0 * one_hot + noise_level * 2.0)
    # Vectorized Dirichlet via Gamma
    gamma_samples = rng.gamma(shape=alpha, scale=1.0)
    post = gamma_samples / gamma_samples.sum(axis=1, keepdims=True)

    w = np.ones(T, dtype=float)

    # Inject burst errors (spurious flips)
    t = 0
    while t < T:
        if rng.random() < burst_prob:
            b_len = rng.integers(burst_len_range[0], burst_len_range[1] + 1)
            end_t = min(T, t + b_len)
            true_cls = gt[t]
            wrong_cls = (true_cls + rng.choice([1, 2])) % 3
            post[t:end_t, :] = 0.1
            post[t:end_t, wrong_cls] = 0.8
            t = end_t
        else:
            t += 1

    # Inject blind spells (tracking lost / missing ball)
    t = 0
    while t < T:
        if rng.random() < blind_prob:
            b_len = rng.integers(blind_len_range[0], blind_len_range[1] + 1)
            end_t = min(T, t + b_len)
            # Flatten posteriors and lower reliability weight w
            post[t:end_t, :] = 1.0 / 3.0 + rng.normal(0, 0.05, size=(end_t - t, 3))
            post[t:end_t] = np.clip(post[t:end_t], 0.01, 0.99)
            post[t:end_t] /= post[t:end_t].sum(axis=1, keepdims=True)
            w[t:end_t] = 0.1  # low reliability
            t = end_t
        else:
            t += 1

    return post, w


def evaluate_decoder_vs_argmax(
    gt,
    post,
    w=None,
    mean_dur=None,
    leave_share=None,
    prior=None,
    min_len=12,
):
    """Run both raw argmax and Viterbi decoder on posteriors, scoring on spellF1 and macroF1.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth labels.
    post : np.ndarray
        (T, 3) posteriors.
    w : np.ndarray, optional
        (T,) reliability weights.
    mean_dur : dict, optional
        Expected durations {"H": 350, "A": 350, "D": 200}.
    leave_share : np.ndarray, optional
        (3, 3) leave share matrix.
    prior : np.ndarray, optional
        (3,) class priors.
    min_len : int
        Min spell length for post-processing merge.

    Returns
    -------
    dict
        Evaluation metrics comparing argmax vs Viterbi.
    """
    if len(gt) == 0:
        return {"spell_f1_argmax": 0.0, "spell_f1_viterbi": 0.0, "delta_spell_f1": 0.0}

    mean_dur = mean_dur or {"H": 350.0, "A": 350.0, "D": 200.0}
    if leave_share is None:
        leave_share = np.array([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
    if prior is None:
        prior = np.array([0.4, 0.4, 0.2])

    # 1. Raw argmax baseline
    pred_argmax = np.argmax(post, axis=1).astype(np.int8)

    # 2. Viterbi decode
    log_trans = build_log_transmat(mean_dur, leave_share)
    log_emit = scaled_log_emissions(post, prior, w)
    pred_viterbi = viterbi_log(log_emit, log_trans)
    pred_viterbi_post = min_spell_merge(pred_viterbi, post, min_len=min_len)

    # Scorer evaluation
    res_argmax = pm.evaluate(gt, pred_argmax, bootstrap_n=0)
    res_viterbi = pm.evaluate(gt, pred_viterbi_post, bootstrap_n=0)

    spell_argmax = float(res_argmax["spell_f1_mean"])
    spell_viterbi = float(res_viterbi["spell_f1_mean"])
    macro_argmax = float(res_argmax["macro_f1"])
    macro_viterbi = float(res_viterbi["macro_f1"])

    return {
        "spell_f1_argmax": spell_argmax,
        "spell_f1_viterbi": spell_viterbi,
        "delta_spell_f1": spell_viterbi - spell_argmax,
        "macro_f1_argmax": macro_argmax,
        "macro_f1_viterbi": macro_viterbi,
        "delta_macro_f1": macro_viterbi - macro_argmax,
        "n_gt_spells": int(res_argmax["n_gt_spells"]),
        "n_spells_argmax": int(res_argmax["n_pred_spells"]),
        "n_spells_viterbi": int(res_viterbi["n_pred_spells"]),
    }


def run_harness_sweep():
    """Run harness over multiple noise levels to verify stable, large spellF1 gain."""
    gt = generate_synthetic_ground_truth(n_possessions=40, seed=123)
    results = []

    noise_levels = [0.2, 0.35, 0.5]
    for noise in noise_levels:
        post, w = corrupt_posteriors(gt, noise_level=noise, seed=456)
        res = evaluate_decoder_vs_argmax(gt, post, w)
        results.append((noise, res))

    return results
