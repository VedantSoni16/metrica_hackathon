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
- [ ] **B.1 Ingest & alignment verification** (Verify `load_atd.py`, frame alignment, team assignment, graded mask from `games.py`).
- [ ] **B.2 Calibrate noise simulator** (`calibrate_noise.py` on real ATD distributions).
- [ ] **B.3 End-to-end pipeline run (`scripts/run_all.py`)** (Generate `submissions/<team>.csv`, validate, score on public window for correctness).

- [ ] **B.4 Smart Tagging integration** (Self-supervised proxy validation without public GT).
- [ ] **B.5 Detector evaluation** (Tactical ball detection experiment if video present).

## Phase C & D: Refinements & Freeze
- [ ] **C.1 Boundary snapping & transition refinement**
- [ ] **C.2 Technical document draft (`docs/technical_report.md`)**
- [ ] **D.1 Single-command pipeline reproducibility verification** (SHA-256 match across runs).
- [ ] **D.2 Submission preparation**
