"""Viterbi decoding for 3-state HMM on soft frame posteriors."""

import numpy as np

STATES = ("H", "A", "D")  # indices 0, 1, 2


def build_log_transmat(mean_dur_frames, leave_share, eps=1e-12):
    """Build log transition matrix for 3 states (H, A, D).

    Parameters
    ----------
    mean_dur_frames : dict
        {"H": float, "A": float, "D": float}
        Expected duration of each state in frames.
        Self-transition = 1 - 1 / mean_dur (geometric duration).
    leave_share : np.ndarray
        3x3 row-stochastic matrix with zero diagonal representing transition
        probabilities conditional on leaving the current state.
        Row 0 (H): [0, P(H->A|leave), P(H->D|leave)]
        Row 1 (A): [P(A->H|leave), 0, P(A->D|leave)]
        Row 2 (D): [P(D->H|leave), P(D->A|leave), 0]
    eps : float
        Numerical floor to prevent log(0).

    Returns
    -------
    log_trans : np.ndarray
        (3, 3) log transition matrix.
    """
    K = len(STATES)
    leave_share = np.asarray(leave_share, dtype=float)
    if leave_share.shape != (K, K):
        raise ValueError(f"leave_share must have shape ({K}, {K}), got {leave_share.shape}")

    P = np.zeros((K, K), dtype=float)
    for i, s in enumerate(STATES):
        dur = float(mean_dur_frames[s])
        if dur <= 1.0:
            stay = 0.0
        else:
            stay = 1.0 - 1.0 / dur
        P[i, i] = stay
        P[i] += (1.0 - stay) * leave_share[i]

    # Normalize rows to ensure exact stochasticity before log
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums
    return np.log(P + eps)


def viterbi_log(log_emit, log_trans, log_start=None):
    """Log-domain Viterbi algorithm.

    Parameters
    ----------
    log_emit : np.ndarray
        (T, K) log emissions for each frame.
    log_trans : np.ndarray
        (K, K) log transition matrix where [i, j] is log P(state_j | state_i).
    log_start : np.ndarray, optional
        (K,) log prior over initial state. If None, uniform log(1/K).

    Returns
    -------
    path : np.ndarray
        (T,) sequence of state indices in [0..K-1].
    """
    log_emit = np.asarray(log_emit, dtype=float)
    T, K = log_emit.shape
    if T == 0:
        return np.empty(0, dtype=np.int8)

    if log_start is None:
        log_start = np.full(K, -np.log(K), dtype=float)
    else:
        log_start = np.asarray(log_start, dtype=float)

    dp = np.empty((T, K), dtype=float)
    bp = np.empty((T, K), dtype=np.int8)

    dp[0] = log_start + log_emit[0]
    for t in range(1, T):
        # scores[i, j] = dp[t-1][i] + log_trans[i, j]
        scores = dp[t - 1][:, None] + log_trans
        bp[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + log_emit[t]

    path = np.empty(T, dtype=np.int8)
    path[-1] = dp[-1].argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def scaled_log_emissions(post, prior, w=None, beta=1.0, eps=1e-6):
    """Compute reliability-weighted log emission likelihoods.

    Converts calibrated posterior P(y|x) into scaled log Bayes factors:
        log P(x|y) ~ beta * w_t * (log P(y|x) - log P(y))
    Weighted by frame reliability w_t in (0, 1] and emission scale beta.

    Parameters
    ----------
    post : np.ndarray
        (T, K) calibrated posteriors.
    prior : np.ndarray
        (K,) class prior frequencies.
    w : np.ndarray, optional
        (T,) reliability weights in (0, 1]. If None, all ones.
    beta : float
        Emission inverse temperature / scaling factor (default 1.0).
    eps : float
        Numerical floor.

    Returns
    -------
    log_emit : np.ndarray
        (T, K) scaled log emissions.
    """
    post = np.asarray(post, dtype=float)
    prior = np.asarray(prior, dtype=float)
    if post.ndim != 2 or post.shape[1] != len(prior):
        raise ValueError(f"post shape {post.shape} incompatible with prior shape {prior.shape}")

    if w is None:
        w_factor = 1.0
    else:
        w = np.asarray(w, dtype=float)
        if w.ndim == 1:
            w_factor = w[:, None]
        else:
            w_factor = w

    # Scaled log-odds
    log_odds = np.log(post + eps) - np.log(prior + eps)
    return (beta * w_factor) * log_odds
