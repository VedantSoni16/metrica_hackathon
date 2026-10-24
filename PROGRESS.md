# MTAH-2027 Implementation Progress

## Phase A: Pre-Match Setup & Architecture Core (No Match Data Required)
- [x] **A.1 Data access check** (Human check: match data gated, developing against specs & Metrica stand-in).
- [x] **A.2 Documentation & Rules review** (Read `strat.md`, `RULES.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md`, `possession_metrics.py`).
- [x] **A.3 Baselines verification** (Reproduced baseline scores with `evaluation/run_evaluation.py`; validated format with `validate_submission.py`).
- [x] **A.4 Label semantics check** (Public GT inspected for formatting/semantics only; no parameters estimated from it).
- [x] **A.5 Core decoder & evaluation utilities** (`src/mtah/decode/viterbi.py`, `src/mtah/decode/postprocess.py`, export/validation wrapper `src/mtah/eval/export.py`, bootstrap CV runner `src/mtah/eval/bootstrap.py`; 7 unit tests passing).
- [x] **A.6 Synthetic-posterior decoder harness** (`src/mtah/sim/harness.py`, `tests/test_harness.py`; confirmed Viterbi decoder cuts fragmentation from ~2500+ spells to ~110 and increases spellF1 by +0.23 to +0.30 across noise levels; all edge cases tested and passing).

- [ ] **A.7 External data download** (`scripts/download_external.py` executed for Metrica & SkillCorner; verify `SOURCES.json` and non-pointer files).
- [ ] **A.8 Silver label derivation** (Parse Metrica events to derive H/A/D silver labels; sanity checks on possession statistics).
- [ ] **A.9 ATD noise simulator & feature extraction library** (Kinematics, soft influence, control proxy, dead-ball cues; tested on Metrica Game 3 EPTS).
- [ ] **A.10 Train frame-level model (Stage 3a)** (LightGBM on simulated noise Metrica; evaluate with leave-one-game-out CV + decoder).
- [ ] **A.11 Pre-build detector scripts & calibrate_noise.py** (Auto-skips if video/weights/GPU missing).

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
