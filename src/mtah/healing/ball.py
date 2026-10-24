"""Ball trajectory healing: short-gap interpolation and reliability estimation."""

import numpy as np
from scipy.interpolate import interp1d


def heal_ball_trajectory(ball_xy, max_gap=20):
    """Interpolate short missing-ball gaps and compute missingness context.

    Parameters
    ----------
    ball_xy : np.ndarray
        (T, 2) ball coordinates (x, y). Missing frames are NaN.
    max_gap : int
        Maximum gap length (in frames) to interpolate. Gaps > max_gap are left NaN.

    Returns
    -------
    dict
        {
            "ball_xy": (T, 2) np.ndarray (healed trajectory),
            "ball_imputed": (T,) bool array,
            "ball_missing": (T,) bool array (True where still NaN),
            "ball_missing_run_len": (T,) int array,
            "ball_reliability": (T,) float array in [0.1, 1.0],
        }
    """
    ball_xy = np.asarray(ball_xy, dtype=float).copy()
    T = len(ball_xy)
    if T == 0:
        return {
            "ball_xy": ball_xy,
            "ball_imputed": np.zeros(0, dtype=bool),
            "ball_missing": np.zeros(0, dtype=bool),
            "ball_missing_run_len": np.zeros(0, dtype=int),
            "ball_reliability": np.zeros(0, dtype=float),
        }

    is_nan = np.isnan(ball_xy[:, 0]) | np.isnan(ball_xy[:, 1])
    ball_imputed = np.zeros(T, dtype=bool)

    # Find runs of NaNs
    diff = np.diff(is_nan.astype(int))
    starts = np.where(diff == 1)[0] + 1
    if is_nan[0]:
        starts = np.r_[0, starts]
    ends = np.where(diff == -1)[0] + 1
    if is_nan[-1]:
        ends = np.r_[ends, T]

    for s, e in zip(starts, ends):
        gap_len = e - s
        # Only interpolate internal gaps <= max_gap (never extrapolate at borders)
        if gap_len <= max_gap and s > 0 and e < T:
            # Linear / cubic interpolation between s-1 and e
            x_vals = [s - 1, e]
            y_vals = ball_xy[[s - 1, e], :]
            interp = interp1d(x_vals, y_vals, axis=0, kind="linear")
            ball_xy[s:e] = interp(np.arange(s, e))
            ball_imputed[s:e] = True

    # Compute missing run lengths
    ball_missing = np.isnan(ball_xy[:, 0]) | np.isnan(ball_xy[:, 1])
    missing_run_len = np.zeros(T, dtype=int)
    cur_len = 0
    for t in range(T):
        if ball_missing[t]:
            cur_len += 1
            missing_run_len[t] = cur_len
        else:
            cur_len = 0

    # Reliability weights: 1.0 observed, 0.6 imputed, 0.1 missing
    reliability = np.ones(T, dtype=float)
    reliability[ball_imputed] = 0.6
    reliability[ball_missing] = 0.1

    return {
        "ball_xy": ball_xy,
        "ball_imputed": ball_imputed,
        "ball_missing": ball_missing,
        "ball_missing_run_len": missing_run_len,
        "ball_reliability": reliability,
    }
