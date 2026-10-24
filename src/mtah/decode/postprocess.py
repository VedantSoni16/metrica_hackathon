"""Post-processing for decoded possession sequences."""

import numpy as np


def runs(labels):
    """Return list of (start, end_exclusive, label) for a 1D int array.

    Parameters
    ----------
    labels : np.ndarray
        1D integer array.

    Returns
    -------
    list of (int, int, int)
        (start_idx, end_idx_exclusive, label) tuples.
    """
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    idx = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, idx]
    ends = np.r_[idx, len(labels)]
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


def min_spell_merge(labels, post, min_len=12, min_d_len=3):
    """Merge short runs to prevent spurious flickers that penalize spellF1.

    Runs of H (0) or A (1) shorter than min_len are merged into the adjacent run
    with the larger summed posterior.
    Short D (2) runs between identical team spells (e.g., H - D - H) shorter than
    min_d_len are also bridged back to the team.

    Parameters
    ----------
    labels : np.ndarray
        1D int array of states (0=H, 1=A, 2=D).
    post : np.ndarray
        (T, 3) posterior probabilities.
    min_len : int
        Minimum frame length for an H or A spell.
    min_d_len : int
        Minimum frame length for a D stoppage between identical team runs.

    Returns
    -------
    labels : np.ndarray
        Smoothed 1D int array.
    """
    labels = np.asarray(labels, dtype=np.int8).copy()
    post = np.asarray(post, dtype=float)
    T = len(labels)
    if T <= 1:
        return labels

    max_iter = T
    for _ in range(max_iter):
        changed = False
        R = runs(labels)
        if len(R) <= 1:
            break

        for i, (s, e, lab) in enumerate(R):
            length = e - s

            # Rule 1: Short D between identical team runs (e.g. H - short D - H -> H)
            if lab == 2 and length < min_d_len and 0 < i < len(R) - 1:
                left_lab = R[i - 1][2]
                right_lab = R[i + 1][2]
                if left_lab == right_lab and left_lab in (0, 1):
                    labels[s:e] = left_lab
                    changed = True
                    break

            # Rule 2: Short H or A run (< min_len)
            if lab in (0, 1) and length < min_len:
                if 0 < i < len(R) - 1:
                    left_lab = R[i - 1][2]
                    right_lab = R[i + 1][2]
                    cand = [left_lab, right_lab]
                    score = [float(post[s:e, c].sum()) for c in cand]
                    best_cand = cand[int(np.argmax(score))]
                    labels[s:e] = best_cand
                    changed = True
                    break
                elif i == 0 and len(R) > 1:
                    # Merge initial short run into following run
                    labels[s:e] = R[1][2]
                    changed = True
                    break
                elif i == len(R) - 1 and len(R) > 1:
                    # Merge trailing short run into preceding run
                    labels[s:e] = R[i - 1][2]
                    changed = True
                    break

        if not changed:
            break

    return labels
