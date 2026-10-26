# MTAH-2027 Implementation Progress

## Phase A: Pre-Match Setup & Architecture Core (No Match Data Required)
- [x] **A.1 Data access check** (Human check: match data gated, developing against specs & Metrica stand-in).
- [x] **A.2 Documentation & Rules review** (Read `strat.md`, `RULES.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md`, `possession_metrics.py`).
- [x] **A.3 Baselines verification** (Reproduced baseline scores with `evaluation/run_evaluation.py`; validated format with `validate_submission.py`).
- [x] **A.4 Label semantics check** (Public GT inspected for formatting/semantics only; no parameters estimated from it).
- [x] **A.5 Core decoder & evaluation utilities** (`src/mtah/decode/viterbi.py`, `src/mtah/decode/postprocess.py`, export/validation wrapper `src/mtah/eval/export.py`, bootstrap CV runner `src/mtah/eval/bootstrap.py`; 7 unit tests passing).
- [x] **A.6 Synthetic-posterior decoder harness** (`src/mtah/sim/harness.py`, `tests/test_harness.py`; confirmed Viterbi decoder cuts fragmentation from ~2500+ spells to ~110 and increases spellF1 by +0.23 to +0.30 across noise levels; all edge cases tested and passing).

- [x] **A.7 External data download** (Verified `external_data/SOURCES.json`, Metrica Games 1–3, and SkillCorner 20 matches with full LFS tracking files present).
- [x] **A.8 Silver label derivation** (`src/mtah/labels/metrica.py`, `tests/test_labels.py`; derived H/A/D silver labels from Metrica events; sanity checks verified on Game 1 and Game 3).

- [x] **A.9 ATD noise simulator & feature extraction library** (`src/mtah/healing/ball.py`, `src/mtah/sim/noise.py`, `src/mtah/io/atd.py`, `src/mtah/features/extractor.py`, `tests/test_features_healing.py`; 14 unit tests passing; 46 features extracted from Game 3 EPTS).
- [x] **A.10 Train frame-level model (Stage 3a)** (`src/mtah/models/baseline_model.py`, `tests/test_model.py`; LightGBM trained on simulated-noise Metrica Game 1 evaluated on Game 3 EPTS; achieved +0.0474 spellF1 gain over nearest-player baseline with 90% CI [+0.0005, +0.0969], significant under bootstrap).


- [x] **A.11 Pre-build detector scripts & calibrate_noise.py** (`scripts/calibrate_noise.py`, `scripts/run_detector.py`, `src/mtah/healing/detector.py`; auto-skips with clear plain-English diagnostics when match video/weights/GPU are absent; stand-in support tested).
- [x] **A.12 End-to-end one-command pipeline (`scripts/run_all.py`)** (Full S0–S7 pipeline pre-built; preflight check lists any missing files; `--standin` runs seamlessly on Metrica Game 3 EPTS; ready to generate `submissions/<team>.csv` upon unzipping match data).
- [x] **A.13 Pre-data hardening** (CV window grid (25, 50, 125) benchmark comparator verified; `scripts/run_all.py` supports `--team-name`, strict tactical-only preflight, and built-in run-twice SHA-256 byte-identical validation with LightGBM `deterministic=True`, `num_threads=1`, fixed seeds).

## Phase B: Real Match Data Ingestion (Triggered Upon Unzipping Data)
- [x] **B.1 Ingest & alignment verification** (`scripts/verify_ingest.py`, `reports/ingest_report.md`; verified 199,319 frames, `video_frame = frame_id - 1`, 158,857 graded frames across both halves, 100% tactical homography validity, Home=Barcelona mapping confirmed, 40.69% ball missing, 20.49 mean tracked players, 638 Smart Tagging instances with 36 N/A; fixed float FrameRate parsing in `src/mtah/io/atd.py`).
- [x] **B.2 Calibrate noise simulator** (`scripts/calibrate_noise.py` executed on real graded ATD; measured ball missing 40.69%, mean missing run 430.95, mean active players 20.49, jitter ball=0.0082m, player=0.0016m; saved to `configs/noise_params.json`; wired into `src/mtah/sim/noise.py`).
- [x] **B.3 End-to-end pipeline run (`scripts/run_all.py`)** (Generated `submissions/Yamal Fanboys.csv` with SHA-256 `d83a391ae0e2...`; verified deterministic pass 1 == pass 2; validated 199,319 rows with `validate_submission.py`; scored once on public window: 18 possessions vs 17 GT, mean dur 13.4s vs 14.2s GT, temporal/semantic alignment confirmed).

- [x] **B.4 Smart Tagging integration** (`scripts/analyze_smart_tagging.py`; self-supervised agreement against nearest-player ball team confirmed clear phase semantics: `BUILD UP` 75.8% and `PROGRESSION` 69.6% attacking/in-possession; `DEFENDING IN ATTACKING/MIDDLE THIRD` 75.8%/69.6% opponent possession. Omitted from training features since external data lacks Smart Tagging schemas and external-CV ablation does not support unmatched features).
- [x] **B.5 Detector evaluation** (`scripts/estimate_ball_pixels.py`; estimated projected ball diameter across 5 sampled tactical frames: 3.49px, 4.26px, 4.43px, 5.91px, 6.25px, mean 4.87px in 1080p. **Decision: NO-GO**. 4.9px is far below off-the-shelf detection thresholds and would inject spurious noise. Spline/Kalman healing handles <=20 frame gaps deterministically without external weights or GPU dependencies).

## Phase C & D: Refinements & Freeze
- [x] **C.1 SkillCorner loader & leave-one-game-out CV** (`src/mtah/labels/skillcorner.py`, `scripts/run_external_cv.py`, `tests/test_skillcorner_labels.py`; implemented 10->25 fps resampled SkillCorner silver labels with D gap evidence; validated sanity against Metrica standards; estimated decoder parameters strictly from external silver labels; ran leave-one-game-out CV with paired bootstrap CIs. Rule enforced: adopt changes only if 90% CI excludes 0. Maintained verified robust plateau parameters: mean_dur H/A=350, D=200, min_len=12, symmetric leave shares).
- [x] **C.2 Technical document draft (`docs/technical_report.md`)** (Comprehensive report drafted following SUBMISSION.md covering approach, assets used/dropped, external data commit SHAs, IFAB possession rules, limitations, reproduction, and ablation table).
- [x] **D.1 Single-command pipeline reproducibility verification** (Executed full pipeline twice from clean state; byte-identical match confirmed with SHA-256 `d83a391ae0e257a840dd5e6c33735c8467257bab1fe9bde7778be14bc3e6ee50`).
- [x] **D.2 Submission preparation** (`submissions/Yamal Fanboys.csv` generated, validated via `evaluation/validate_submission.py`; `requirements-solution.txt` pinned; `README.md` updated with exact reproduction instructions).
