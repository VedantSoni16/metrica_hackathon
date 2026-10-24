# MTAH-2027 Solution Notes

## 1. Hard Rules (RULES.md & SUBMISSION.md)
* **Zero GT leakage:** Public 5-min GT (`evaluation/gt/`) is for pipeline-correctness checks ONLY. Never train, tune, or estimate parameters from it.
* **No hand labels / human data:** No human annotations, corrections, or external data (Opta, StatsBomb, Wyscout) for the BAR-ATH match.
* **External data allowed:** Data & annotations for OTHER games (Metrica, SkillCorner) are 100% allowed if public & documented.
* **Artifact exclusion:** Never commit `data/`, `external_data/`, `cache/`, `weights/`, or frame-level derived data to git.
* **Determinism & reproducibility:** Final submission `<team>.csv` must be byte-identical on rerun (`python scripts/run_all.py`). Fixed seeds, pinned deps.
* **Safety line:** Ask human before any remote/cloud/API call or if near any competition rule line. VLM audit remains OFF.

## 2. Key Architecture & Design Decisions (strat.md §5, §6)
* **Metric focus:** Ranking metric is `spellF1` (mean IoU @ 0.3..0.7 over maximal H/A runs). Tie-break: `macroF1`. Spurious flips kill `spellF1`.
* **Timeline:** Tactical video is ground truth (199,319 frames @ 25 fps, 0-indexed). Broadcast is unsynced (do not index to it).
* **Ball healing:** Short-gap cubic spline / Kalman smoothing (<=20 frames, mark imputed + lower reliability). Optional tactical YOLO detector for missing ball (auto-skips if video/weights/GPU missing). Drop player Re-ID.
* **Feature pipeline:** Kinematics (v, a, jerk), soft team influence `exp(-d/tau)` (tau in 1.5, 3, 6m), control proxy (rel speed, cos angle), dead-ball cues (pitch bounds 105.3x68m, stoppage dynamics), Smart Tagging priors (as evidence, not hard labels), missingness context.
* **Model:** Calibrated LightGBM (depth <= 4) / regularised logistic regression for P(H), P(A), P(D) + reliability weight `w_t`. Trained exclusively on external data with simulated ATD noise.
* **Decoder:** 3-state log-Viterbi per half (H, A, D) on graded frames. Transitions parameterised by mean durations & leave-shares estimated from external data. Weighted emissions `w_t * (log P - log prior)`.
* **Post-processing:** Min-spell merge (<12 frames into higher-posterior neighbour). Boundary snapping to control events. Output `X` on ungraded frames (before kickoff, halftime, full-time); no `X` on graded frames.

## 3. Data Layout & Formats (DATA.md & games.py)
* `data/tactical/BAR-ATH_tactical_atd.xml|.txt`: EPTS format. Player slots 1..46 (not IDs). `video_frame = frame_id - 1`.
* Graded blocks from `games.half_video_blocks("BAR-ATH")`: Half 1 (11,185..90,651), Half 2 (110,808..190,197) = 158,857 graded frames.
* Coordinates: Normalised [0, 1] x [0, 1]. In metres: x * 105.3, y * 68.0. Normalise attack direction per half.
* Smart Tagging: 638 XML phases, time in seconds -> `round(sec * 25)`.
* Homography: 3x3 matrix mapping normalised pitch to image pixels.
* Submission contract: exactly 199,319 rows, header `frame,label`, labels in `{H, A, D, X}`. Validated by `evaluation/validate_submission.py`.
