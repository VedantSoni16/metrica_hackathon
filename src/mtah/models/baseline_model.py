"""Stage 3a baseline LightGBM model and benchmark comparator."""

import lightgbm as lgb
import numpy as np
import pandas as pd


class PossessionClassifier:
    """Stage 3a LightGBM model predicting P(H), P(A), P(D) from frame features."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.08,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ):
        self.params = {
            "objective": "multiclass",
            "num_class": 3,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "deterministic": True,
            "num_threads": 1,
            "verbosity": -1,
            "n_jobs": 1,
        }
        self.n_estimators = n_estimators
        self.booster = None

    def fit(self, X, y):
        """Fit model on features X and labels y in {0, 1, 2}."""
        if isinstance(X, pd.DataFrame):
            X_mat = X.to_numpy(dtype=np.float32)
        else:
            X_mat = np.asarray(X, dtype=np.float32)
        y_vec = np.asarray(y, dtype=int)

        dataset = lgb.Dataset(X_mat, label=y_vec, free_raw_data=False)
        self.booster = lgb.train(self.params, dataset, num_boost_round=self.n_estimators)
        return self

    def predict_proba(self, X):
        """Return calibrated (T, 3) posterior probabilities."""
        if self.booster is None:
            raise RuntimeError("Model has not been fitted yet.")
        if isinstance(X, pd.DataFrame):
            X_mat = X.to_numpy(dtype=np.float32)
        else:
            X_mat = np.asarray(X, dtype=np.float32)
        return self.booster.predict(X_mat)

    def predict(self, X):
        """Return hard predictions in {0, 1, 2}."""
        return np.argmax(self.predict_proba(X), axis=1)



def run_nearest_player_baseline(ball_xy, player_h, player_a, window=50):
    """Implement reference nearest-player-majority-vote baseline from starter/baseline.py."""
    T = len(ball_xy)
    pred_raw = np.full(T, 2, dtype=np.int8)  # default Dead

    valid_ball = ~(np.isnan(ball_xy[:, 0]) | np.isnan(ball_xy[:, 1]))
    if np.any(valid_ball):
        diff_h = player_h[valid_ball] - ball_xy[valid_ball, None, :]
        dist_h = np.nanmin(np.hypot(diff_h[:, :, 0], diff_h[:, :, 1]), axis=1)

        diff_a = player_a[valid_ball] - ball_xy[valid_ball, None, :]
        dist_a = np.nanmin(np.hypot(diff_a[:, :, 0], diff_a[:, :, 1]), axis=1)

        team_choice = np.where(dist_h <= dist_a, 0, 1).astype(np.int8)
        pred_raw[valid_ball] = team_choice

    if window <= 0:
        return pred_raw

    smoothed = np.empty(T, dtype=np.int8)
    one_hot = np.zeros((T, 3), dtype=np.int32)
    for c in (0, 1, 2):
        one_hot[:, c] = (pred_raw == c).astype(np.int32)

    cum = np.vstack([np.zeros((1, 3), dtype=np.int32), np.cumsum(one_hot, axis=0)])
    for t in range(T):
        s = max(0, t - window)
        e = min(T, t + window + 1)
        window_counts = cum[e] - cum[s]
        smoothed[t] = int(np.argmax(window_counts))

    return smoothed


def find_best_nearest_player_baseline(ball_xy, player_h, player_a, y_true, windows=(25, 50, 125)):
    """Evaluate nearest-player baseline across candidate smoothing windows and return the best."""
    import possession_metrics as pm

    best_w = None
    best_score = -1.0
    best_pred = None

    for w in windows:
        pred = run_nearest_player_baseline(ball_xy, player_h, player_a, window=w)
        eval_res = pm.evaluate(y_true, pred, bootstrap_n=0)
        score = float(eval_res["spell_f1_mean"])
        if score > best_score:
            best_score = score
            best_w = w
            best_pred = pred

    return {"best_window": best_w, "best_spell_f1": best_score, "pred": best_pred}

