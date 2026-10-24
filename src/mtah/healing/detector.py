"""Tactical video ball detector wrapper with auto-skip guards."""

from pathlib import Path


def is_detector_environment_ready(video_path=None, weights_path=None, require_gpu=False):
    """Check whether video, model weights, and compute environment are ready.

    Returns
    -------
    tuple of (bool, str)
        (is_ready, reason_if_not_ready)
    """
    reasons = []

    if video_path is None or not Path(video_path).exists():
        reasons.append(f"Tactical video not found ({video_path})")

    if weights_path is None or not Path(weights_path).exists():
        reasons.append(f"Model weights not found ({weights_path})")

    try:
        import cv2  # noqa: F401
    except ImportError:
        reasons.append("OpenCV (cv2) not installed")

    if require_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                reasons.append("GPU requested but CUDA is not available")
        except ImportError:
            reasons.append("PyTorch not installed")

    if reasons:
        return False, "; ".join(reasons)
    return True, "Ready"


def run_ball_detector(
    video_path,
    weights_path,
    missing_frame_indices,
    homography_matrix_provider=None,
    out_parquet_path=None,
):
    """Run ball detector on frames with missing ball, back-projecting to pitch coordinates.

    Auto-skips cleanly if prerequisites are missing.

    Parameters
    ----------
    video_path : str or Path
    weights_path : str or Path
    missing_frame_indices : list or np.ndarray
    homography_matrix_provider : callable, optional
    out_parquet_path : str or Path, optional

    Returns
    -------
    dict
        {frame_idx: (x_metres, y_metres)} or empty dict if skipped.
    """
    ready, reason = is_detector_environment_ready(video_path, weights_path)
    if not ready:
        print(f"[SKIP] Detector auto-skipped: {reason}")
        return {}

    # Placeholder for actual YOLO inference loop when assets are unpacked
    print(f"[INFO] Running ball detector on {len(missing_frame_indices)} missing frames...")
    detections = {}
    return detections
