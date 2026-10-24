# STRAT.md — More than a Hack 2027, Phase 1: Ball Possession

> **Audience:** an AI coding agent (Claude Code, Antigravity, Cursor, etc.) and the human owner.
> **Purpose:** single source of truth for *what* we are building, *why*, and *how to build it cleanly within the competition rules*.
> **Status date:** 2026-10-24. Submission deadline: **2026-11-17**. Results before 2026-11-26.
> **Owner's situation:** repo cloned, venv created, `requirements.txt` installed. **Match data not yet downloaded.** Work that does not need the match data comes first (Section 12).
> **v2 (rules-reviewed):** `RULES.md` and `SUBMISSION.md` have now been read and this file was revised to comply. The biggest change: the public 5-minute ground truth is **not** a training or tuning set, so no parameter in this system may be estimated from it (Sections 2, 9, 10).

---

## 0. How the agent must use this file

1. **Read before coding, in this order:** `README.md`, `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md`, `RULES.md`, `FAQ.md`, then `evaluation/possession_metrics.py` (the scoring spec; if prose and code disagree, **the code is right**).
2. `RULES.md` and `SUBMISSION.md` have been incorporated into Section 2 and Section 15. If either file changes or you find a contradiction, **the repo files win** — update this document. If an approach sits near a line, open an issue on the organisers' repo and ask **before** building on it (RULES.md says so explicitly).
3. Items marked **[VERIFY]** are claims from memory or secondary sources. Items marked **[HYPOTHESIS]** are design ideas to be tested, not facts.
4. **Never modify** files under `evaluation/` (organisers' scorer). Put all new code in `src/mtah/` (Section 11).
5. Build in the order of Section 12. Each stage has an acceptance test. Do not start a later stage until the earlier one passes.
6. Every experiment is judged by **bootstrap intervals**, not a single score (Section 10).
7. Keep everything reproducible: fixed seeds, config files, one command per stage, no manual labelling of any kind.

---

## 1. The task in one paragraph

Given one full LaLiga match (FC Barcelona = Home `H` vs Athletic Club = Away `A`, 2024-25, matchday 2), output one label per video frame of the **tactical** video (199,319 frames, 25 fps) from `{H, A, D, X}`:

| Label | Meaning |
|---|---|
| `H` | Home in possession, ball alive |
| `A` | Away in possession, ball alive |
| `D` | Dead ball (out of play / stoppage) |
| `X` | No answer (abstain) |

Submission = `frame,label` CSV with exactly 199,319 rows (+ public code repo + short technical document; see `SUBMISSION.md`).

### Metrics (from `EVALUATION.md`)

* **Ranking metric: `spellF1`.** A *possession* = a maximal run of `H` or of `A`. `D` is never a possession. Predicted possessions are matched one-to-one with true possessions by temporal IoU; F1 is averaged over IoU thresholds {0.3, 0.4, 0.5, 0.6, 0.7}. Constant/random guess = 0. One-frame offset is free; 1 s offset ≈ 0.65.
* **Tie-break: `macroF1`.** Mean of per-class F1 over H, A, D on all graded frames (frame-exact). `X` on a graded frame counts as a miss.
* **`X` handling in `spellF1`:** the scorer forward-fills each `X` with the last non-`X` label *in the same half* (back-fills at head of a half) before matching. So `X` = "repeat my previous answer" for the ranking metric.
* **Noise floors:** differences < ~0.01 macroF1 and < ~0.04 spellF1 (full match) are noise.
* **Published floor to beat:** `baseline_smooth50`, full-match spellF1 **0.2433** (macroF1 0.6724). Other baselines: raw 0.0486 / 0.6532; smooth25 0.1996 / 0.6752; smooth125 0.2242 / 0.6356.

### Key consequence
The metric counts **segments**, not frames. The system must output **few, stable, correctly-timed possession segments**. Every spurious flip is a false positive *and* splits a true possession. Design everything around that.

---

## 2. Rules and constraints (from `RULES.md` and `SUBMISSION.md`, verified)

### What is allowed
* **Any method**, as long as (1) organisers can **access and inspect** every part of it (your repo, a public model, a commercial API, an open dataset) and (2) the technical document **explains** it.
* Pre-trained models (commercial/open), LLMs (hosted/local), **external data about other games**, your own detection/tracking/event recognition, classical methods.
* **Improving or replacing the provided tracking is legitimate**, provided it is done by code.
* **Hand-labelled data from OTHER games is allowed in any amount** — but organisers must be able to access it and the document must say what was labelled, how, by whom, how much.
* Unsupervised / self-supervised learning and **algorithmic pseudo-labels** on the provided game are allowed (no person may decide any of the labels).
* Watching the video / inspecting the data to understand it, debug and design features is allowed.

### What is forbidden
1. **Hand annotation.** No human-written labels, corrections, or manual fixes on the provided game — not for the submission, not for training, not for validation. This includes hand-correcting tracking (ball positions, identity switches), hand-editing the output CSV, fixing only transitions, and **rules/constants tied to specific frames of the game** ("turning what you saw into … rules and constants tied to specific frames").
2. **Private ground truth, or any human-produced data about the provided game from any source** (Opta/StatsBomb/Wyscout/LaLiga events, corrected tracking, possession data, annotations). Data of this kind about *other* games is fine.
   * **The public 5-minute GT is for pipeline-correctness checks ONLY** (frames aligned, file well-formed, labels mean what you think). It is **not training data and not a tuning set.**
   * Consequence for this project: **do not estimate any parameter from it** — not decoder durations, not transition shares, not the meaning of Smart Tagging tags, not thresholds, not smoothing windows. All such parameters come from external data (Section 7) and CV on it. Reading the public GT to understand what the labels *mean* (e.g. how restarts are labelled) is fine; computing numbers from it that enter the model is not.
3. **A submission the repo cannot regenerate.** Running the repo on the provided data must reproduce the **same file** (byte-identical, not approximately). This drives determinism requirements below.

### Data licence (the enrolment terms are the authority — read them)
* Match assets are licensed for **this competition only**. Do not redistribute; **do not commit data, frame-level tracking exports, clipped video, or similar derived data to the public repo.** `data/` is already gitignored; also ignore `external_data/`, `cache/`, model weights.
* **[ACTION — human]** Re-read the enrolment terms before (a) uploading match video/frames to any third-party API (VLM audit), or (b) processing the data on cloud machines. If the terms do not clearly allow it, do not do it, or ask the organisers.

### Reproducibility requirements that follow from rule 3
* Single command regenerates the CSV from the data bundle. Pin dependencies (exact versions). Fixed seeds. Set deterministic flags (e.g. `torch.use_deterministic_algorithms(True)`, single-threaded/stable ops where needed). Prefer CPU or fixed-version GPU inference for anything feeding the final labels; **verify by running the pipeline twice and comparing file hashes (SHA-256).**
* External datasets must be downloaded by a script from pinned commits (see `scripts/download_external.py`, Section 7A) so organisers can obtain exactly what you used.
* Any remote API call (e.g. a vision LLM) is a determinism and data-licence risk: **default OFF.** If ever enabled, its responses must be cached in the repo keyed by a content hash so the regenerated file is identical without calling the API, and the organisers must be able to inspect the prompts/model id. Ask the organisers first whether caching small per-window text responses is acceptable.
* Intermediate frame-level products derived from the match (detections, features, tracking fixes) are **generated locally by the pipeline and never committed**.

### Team and submission mechanics
* Teams of **3–5 members** (fewer than 3 = not eligible for the Innovation Challenge / Final Pitch).
* One Google Form (link emailed to registered participants; **needs a Google account**), one response per team, **last response before Nov 17, 2026 is scored**, you may edit it. **No live leaderboard and no feedback** — only the public 5-minute window and `validate_submission.py`. **Validate the exact file you upload.**
* File names: **`<team_name>.csv`** and **`<team_name>.pdf`** (technical document may be `.pdf` or `.md`; name it after the team too), public repo URL.

---

## 3. Data facts (from `DATA.md`)

Layout after download (unpack into `data/` or set `MTAH_DATA_DIR`):

```
data/tactical/BAR-ATH_tactical.mp4              # 199,319 frames, 25 fps, 1920x1080. THE REFERENCE TIMELINE
data/tactical/BAR-ATH_tactical_atd.xml|.txt     # tracking (EPTS/FIFA format) — players + ball
data/tactical/BAR-ATH_tactical_homography.csv   # 199,319 rows, 100% valid
data/tactical/BAR-ATH_tactical_smart_tagging.xml# 638 automatic tactical phases
data/broadcast/...                              # own timeline (164,488 records), NOT synchronised, 65.8% valid homography
```

Use `games.py` (`asset_paths()`, `half_video_blocks()`) — never hard-code filenames.

| Fact | Value |
|---|---|
| Graded frames | 158,857 (1st half 11,185–90,651; 2nd half 110,808–190,197). Everything else is ignored by the scorer; write `X` there. |
| Frame alignment | ATD `frame_id` 1…199,319 → `video_frame = frame_id − 1`. Smart Tagging seconds → `round(sec·25)`. Homography `frame` 0-based = submission row. |
| Coordinates | ATD & homography use **normalised 0–1** pitch coords. Metres = ×(105.3, 68.0) (`games.py` constants). |
| Track slots | 46 anonymous slots (`Track 1..46`) for 22 players. `ShirtNumber` = slot index, **not a person**. Slots appear/vanish/swap. Mean ~20.5 objects during play; can exceed 22 (duplicates). |
| Ball visibility | Present on 56.9% of all frames; **59.3% of graded frames** (≈41% missing). Missing = `NaN`. Missing more often exactly when it matters (aerial balls, crowded box, occlusion). |
| Smart Tagging | 638 phases. Codes: PLAYERS IN THE BOX, BALL IN FINAL THIRD, BALL IN THE BOX, PROGRESSION, DEFENDING IN MIDDLE THIRD, SET PIECES, CREATING CHANCES, DEFENDING IN DEFENSIVE THIRD, LONG BALL, ATTACKING TRANSITION, DEFENSIVE TRANSITION, BUILD UP, DEFENDING IN ATTACKING THIRD, GOALS. Tag groups: Team (`Barcelona`/`Atletic Club`/`N/A`), Half, Type (Throw-in 26, Corner 12, Goal Kick 10, Free Kick 5, Kick Off After Goal 4, Kick Off Start 2), Side, Direction of ball entry, Max Players in the box (string, incl. `7+`). **No "foul" code exists.** |
| Smart Tagging caveats | Automatic, not ground truth. 36 instances have `Team = N/A`, **including every throw-in**. Phases ≠ possessions. Column names are Title Case, codes UPPER, values Title Case → match case-insensitively. |
| Homography | 3×3 per frame mapping normalised pitch → image pixels. Check `valid`; use `reprojection_error` as trust. Tactical median error ≈0.24 in play, ≈1.5 at halftime. |
| Public GT | `evaluation/gt/BAR-ATH_possession_gt_public.csv`: real labels only for frames **11185–18684** (H 3,454 · A 2,577 · D 1,469; 17 possessions, mean 14.2 s). Everything else `X`. For pipeline-correctness checks ONLY — **not** training, tuning or parameter estimation (RULES.md §2). |

Reader helpers already in repo: `starter/load_atd.py`, `load_homography.py`, `load_smart_tagging.py`, `baseline.py`, `make_dummy_submission.py`, notebooks `01–03`. Reuse them.

---

## 4. Label semantics (from `CHALLENGE.md`)

* Ball in play ⇒ **exactly one** team is in possession; no neutral state. Loose ball ⇒ last team in possession keeps it.
* Every restart (kick-off, throw-in, corner, goal kick, free kick) has a clear owner; owner of a *dead* ball is **not** asked — output `D`.
* Possession changes when the other team has the ball **under control**. Not changes: a single non-controlled touch; a deflected-but-not-intercepted pass; an aerial duel won without control.
* GT is **mechanically derived from corrected tracking**: `D` where corrected tracking marks the ball dead, otherwise the team the tracking names as owner. No smoothing/interpolation afterwards.
* `H H H A A` = 2 possessions. `H H D D H H` = 2 possessions. `D` is never a possession.

⇒ The real problem is detecting **when ownership changes** and **when the ball is dead**, from noisy automatic data.

---

## 5. Decision log — review of the owner's original plan

The owner's draft had five phases (YOLO + Re-ID healing → multimodal features → Bi-LSTM/Transformer → HMM → API vision audit). Verdict per item:

| Item | Decision | Reason |
|---|---|---|
| Ball recovery with a detector (YOLO) | **KEEP, but redesign** | Highest-value upgrade (ball is the best signal, missing on ~41% of graded frames). **Run on the TACTICAL video, not the broadcast video**: the broadcast is unsynchronised, so its detections cannot index the submission rows. Run only on frames where the ATD ball is missing. Check ball pixel size first (tactical camera may be wide ⇒ tiny ball ⇒ need tiled/SAHI-style inference or a small-object model). |
| Player Re-ID (FastReID etc.) | **DROP (deprioritise to "never")** | The task needs *team* membership, not identity. Track slots already carry team. Re-ID only matters for cross-team switches, which can be handled with a cheap consistency check. Huge cost, ~zero expected gain for spellF1. |
| Kalman/spline imputation ≤30 frames | **KEEP, shorten** | Cap gaps at ~12–25 frames (0.5–1 s); mark imputed frames with a flag + reduced reliability weight. A flying pass can vanish >1 s; do not interpolate across it blindly. |
| "Leave long dropouts empty to flag D" | **REJECT** | Long ball dropouts happen in air balls/crowds/occlusion, not only dead balls. Use missingness as a *feature* with context, never as a D label. |
| Feature set (velocity, accel, d*_H/d*_A, cosine, centroids, hull, boundary) | **KEEP, fix** | (a) Coordinates are 0–1, not centred metres: pitch check is `x∉[0,105.3]` etc., not `|x|>52.5`. (b) `time_since_foul` does not exist in Smart Tagging; use set-piece **Type** tags. (c) Add soft/weighted team influence (more robust than the single nearest player), ball-missing context, and attack-direction normalisation per half. |
| Bi-LSTM/Transformer on 250-frame windows, fine-tuned on the 5-minute clip | **KEEP only with external training + noise augmentation; start with GBDT. NO fine-tuning on the public 5 minutes** | Fine-tuning on the public GT is forbidden (RULES.md §2: not training data, not a tuning set). It would also memorise ≈30 windows. Build a LightGBM/logistic baseline first; the sequence model must beat it under leave-one-match-out CV on external data. |
| Multi-task heads (alive/dead + possession anchor) | **KEEP** (cheap, sensible factorisation) | Dead-ball and ownership depend on different cues. |
| HMM with `hmmlearn.CategoricalHMM`, 4 states incl. `X`, `self=0.985` | **REPLACE with custom 3-state log-Viterbi on soft posteriors** | (a) `X` is not a true state in GT (only H/A/D). (b) `0.985` self-transition ⇒ mean duration ≈ 67 frames (2.7 s), but real possessions are ≈ 14 s (≈ 355 frames) ⇒ need ≈ 0.997. (c) Categorical emissions over 4 symbols throw away the model's soft confidence. (d) H→A direct turnovers (interceptions) are common; do not over-penalise. Estimate transition parameters from data. |
| Minimum-spell rule (<25 frames → `X`/merge) | **KEEP as tunable post-process; merge, don't abstain** | Merge into neighbour with higher summed posterior; tune min length on a broad plateau (start ~12 frames). |
| API vision-audit loop (Gemini/GPT) | **DEFAULT OFF; optional gated experiment only** | Breaks byte-identical reproducibility unless responses are cached, sends match video to a third party (check enrolment terms), non-deterministic, costly, vision LLMs are weak at frame-accurate possession in 3-s low-res clips. Only adopt if it improves *external-data and public-window* bootstrap intervals; cache every response in the repo; temperature 0; consider an open-weights VLM for reproducibility. Model names in the old draft are outdated — use whatever current vision-capable model is available. |
| Pre-training on Metrica Open Data / StatsBomb | **KEEP, expanded; this is now the ONLY source of parameters/labels** | External games are explicitly allowed. Metrica sample data = real tracking+events (Game 3 uses the same EPTS/FIFA format as the competition ATD). StatsBomb = events only ⇒ priors only. SkillCorner open data = broadcast-derived tracking with possession events. All must exclude the provided game (Section 7). |

---

## 6. Target architecture

```
raw ATD + homography + Smart Tagging (+ tactical video)
        │
 S0  Ingest & align ───────────── frame index, graded mask, metres, attack-direction normalisation
        │
 S1  Ball healing ─────────────── short-gap imputation (≤~20 frames) + [optional] detector on frames with missing ball
        │
 S2  Feature extraction ───────── ball kinematics, soft team influence, control proxy, dead-ball cues, tag priors, missingness context
        │
 S3  Frame-level model ────────── P(H), P(A), P(D) + reliability weight w_t     (GBDT first, sequence model second)
        │
 S4  Structured decoder ───────── 3-state log-Viterbi per half, duration-aware transitions, w_t-scaled emissions
        │
 S5  Post-process ─────────────── min-spell merge, boundary snapping to control events
        │
 S6  [Optional] VLM audit ─────── only for low-confidence windows, cached, gated by experiment
        │
 S7  Export & validate ────────── 199,319 rows; X on ungraded frames; validate_submission.py
```

### S0 Ingest & align
* Load with `starter/load_atd.py` (kloppy EPTS reader under the hood). Convert `frame_id → video_frame = frame_id − 1`.
* Convert to metres (105.3 × 68.0). Create `graded` boolean mask from `half_video_blocks()`.
* Determine Home/Away team membership of each slot from the ATD metadata XML. **[VERIFY]** Home = Barcelona = `H`.
* Normalise attack direction per half (e.g. infer which goal each team attacks from goalkeeper/average positions or Smart Tagging) so features like "ball in attacking third" are consistent. Record the inferred direction in a log.
* De-duplicate objects (>22 tracked objects): merge near-coincident same-team tracks (< ~0.5 m).

### S1 Ball healing
* **Short gaps:** interpolate (cubic spline or Kalman smoother) for gaps ≤ `max_gap` (default 20 frames; tune 12–30). Add columns `ball_imputed` and `ball_missing_run_len`. Never extrapolate across long gaps.
* **Detector (optional, gated):**
  1. Inspect a handful of tactical frames: ball pixel diameter, camera framing. If ball ≲ 6–8 px, plain 640-px YOLO inference will fail → use tiled inference or crop around the projected last-known ball position.
  2. Candidate weights: community football-ball detectors (e.g. Roboflow football datasets/`roboflow/sports` examples) **[VERIFY availability + licence]**. Ultralytics YOLO is AGPL-3.0 **[VERIFY]** — compatible with a public repo but must be declared in the technical document.
  3. **Self-supervised evaluation, no labels needed:** on frames where ATD *has* the ball, project the ATD ball to pixels with the homography (`H @ [x, y, 1]`), check whether the detector finds the ball nearby. Report recall@r px. This also yields free pseudo-labels to fine-tune the detector on this very match (algorithmic, not hand annotation).
  4. Back-project detected pixels to pitch coordinates with `inverse(H)` → **only valid for balls near the ground plane**; airborne balls project with parallax error. Mark detector-derived ball positions with a lower reliability weight and a flag.
  5. Compute budget: run only on graded frames with missing ATD ball (≈64k frames), in batches, with on-disk caching (`cache/ball_detections.parquet`, **gitignored — derived frame-level data must not be published**). Pseudo-labels from the provided ATD ball are algorithmic and allowed. Detector inference must be deterministic enough that two runs yield an identical final CSV (verify by hash; if the GPU path is not bit-stable, quantise/round detector outputs before they enter features, or use CPU).
* **Broadcast tracking as ball fill (later experiment):** requires aligning the broadcast timeline to the tactical one (e.g. cross-correlate player-centroid/ball trajectories in normalised pitch coords where broadcast homography is valid). Only attempt after S3 baseline is strong.

### S2 Features (computed per frame, with ± context windows; offline ⇒ non-causal is allowed)
Group A — **Ball kinematics** (on healed ball): velocity/acceleration via central differences (in m/s, m/s²), speed, heading change, jerk, "time since last sharp velocity change", rolling stats at 0.2/0.5/1/2 s.
Group B — **ID-independent proximity:** `d_H*`, `d_A*`, `Δd = d_A* − d_H*`, k-nearest distances per team (k=1..3), counts of players of each team within 2/4/8 m, **soft influence** `Σ exp(−d_i/τ)` per team (τ ∈ {1.5, 3, 6} m).
Group C — **Control proxy:** time the ball stays within r (1.5–2.5 m) of the same team's nearest player with relative speed below a threshold; ball speed drop while a player is near; cosine between ball velocity and nearest-player velocity (guard divide-by-zero).
Group D — **Ball-missing context:** `ball_missing`, `ball_imputed`, run length, time since last seen, last-known owner, team centroid/compactness when blind.
Group E — **Team shape:** centroids, spread, convex hull area, line height, per team, in attack-normalised coords. **[HYPOTHESIS]** shape expands in possession; verify with data before trusting.
Group F — **Dead-ball cues:** ball inside/outside pitch (`x∉[0,105.3]` or `y∉[0,68]`), ball near touchline/goal line, ball nearly stationary near a restart spot, number of players stationary, max/mean player speed drop, team spread at restart.
Group G — **Smart Tagging priors:** active code(s), team tag (with explicit `N/A` handling), set-piece Type, frames since phase start/end, time-to/since nearest SET PIECES interval by type, flags for "ATTACKING/DEFENSIVE TRANSITION". Treat as **evidence, not labels**. Determine the *meaning* of the `Team` tag per code (team in possession vs team defending?) **without the public GT**: measure, on the provided game, its agreement with an algorithmic proxy (e.g. which team's players are nearest the ball during that phase) — a self-supervised check, no human labels — and cross-check with the Metrica tag documentation. Do not assume, and do not hard-code frame-specific rules.
Group H — **Context:** half, normalised match clock (only if it generalises), distance of ball to each goal.

Unit-test each feature on synthetic inputs (constant velocity ball, stationary ball, ball between two players).

### S3 Frame-level model
* **Targets:** 3-class `{H, A, D}` plus auxiliary binary `alive`. Output calibrated probabilities and a reliability weight `w_t ∈ (0, 1]` (low when ball missing/imputed/detector-derived, high when ball observed near a player).
* **Stage 3a (baseline, mandatory):** regularised logistic regression and shallow LightGBM (depth ≤ 4, strong regularisation, early stopping on grouped CV). Missing ball ⇒ native NaN handling (GBDT) + missingness features.
* **Stage 3b (if 3a plateaus):** 1D-TCN or Bi-GRU over ~10 s windows (250 frames), multi-task heads (alive/dead, possession H/A/None). Train primarily on external data with ATD-like noise (Section 8), do **not** fine-tune on the public 5 minutes (forbidden as training/tuning data). Must beat 3a in leave-one-match-out CV to be adopted.
* **Validation:** grouped by match/possession, never random frames (neighbouring frames are near-duplicates ⇒ leakage).

### S4 Structured decoder
Per half, over graded frames only, 3-state HMM with **log-domain Viterbi** (spec in Section 9). Emissions = scaled likelihoods from S3 posteriors, weighted by `w_t`. Transition matrix built from **mean durations** (≈ 355 frames for possession, estimated D duration from GT/external data) and leave-shares (H→A vs H→D, D→H vs D→A). Tune only 2–4 scalars; pick the centre of a flat plateau, not the peak.

### S5 Post-processing
* **Min-spell merge:** runs of H/A shorter than `min_len` (start 12 frames; tune) are merged into the adjacent run with the larger summed posterior. Do not convert to `X`.
* **Boundary snapping:** move each H↔A boundary to the nearest detected "control event" (ball-speed discontinuity near a player of the new team) within ±15 frames; move D boundaries to ball-stop/leave-pitch/restart events. Accept only if it helps in CV.
* **`X` policy:** default = **no `X` on graded frames** (macroF1 counts them as misses and the decoder already carries state through blind spells). `X` on ungraded frames (before kickoff, halftime, after final whistle) is free and recommended.

### S6 Optional VLM audit (DEFAULT OFF — see Section 2)
* Trigger only on windows where decoded-path posterior margin is below a threshold (e.g. bottom 2–3% of frames, merged into ≥2 s windows).
* Send a short, low-res clip + frame range; require strict JSON `{"state": "H"|"A"|"D"}`; temperature 0; log prompt, model id, and response to `cache/vlm/`; for the final file, responses would have to be cached in the repo (text only, never video/frames) so the regenerated CSV is identical without the API. **Ask the organisers (issue) before enabling**, and confirm the enrolment terms allow sending match clips to a third party.
* Adopt **only** if it yields a bootstrap-significant improvement on held-out **external** data (the public window may only be used as a correctness check). If it does not, disable it and document the negative result.

### S7 Export & validate
* Write `submissions/<team_name>.csv`, run `python evaluation/validate_submission.py submissions/<team_name>.csv` **before every submission** (row-count check = 199,319; ~164k rows means the broadcast timeline was used by mistake).
* Score locally: copy to `evaluation/results/` and run `python evaluation/run_evaluation.py --bootstrap 1000`.

---

## 7. External data plan (layouts of Metrica + SkillCorner verified 2026-10-24; StatsBomb and others still **[VERIFY]**)

Purpose: the public GT is only ~17 possessions **and may not be used for training/tuning at all** (RULES.md §2). External data about OTHER games is therefore the only source of (1) training/validation labels and (2) every decoder/threshold parameter. **Hard exclusion:** never ingest any data (events, possession, tracking, annotations) about FC Barcelona vs Athletic Club, LaLiga 2024-25 matchday 2. Full download instructions: Section 7A.

| Source | What it gives | Use | Caveats |
|---|---|---|---|
| **Metrica sample data** (`github.com/metrica-sports/sample-data`) | **Verified layout:** `data/Sample_Game_1|2/*_RawTrackingData_{Home,Away}_Team.csv` + `*_RawEventsData.csv`; `data/Sample_Game_3/Sample_Game_3_{tracking.txt,metadata.xml,events.json}`; `documentation/events-definitions.pdf`. 3 games; Games 1–2 CSV tracking+events; **Game 3 in EPTS/FIFA format + JSON events (same family as the competition ATD; kloppy-readable)**; 25 Hz; coords 0–1, 105×68 m; tracking and events synchronised; anonymised. | Primary training/validation source: derive H/A/D labels from events (below), corrupt the clean tracking with the ATD noise simulator (Section 8). | Clean tracking ⇒ domain gap; must simulate noise. Only 3 games ⇒ use leave-one-game-out CV. Repository asks to acknowledge the source and "use the data responsibly". |
| **SkillCorner open data** (`github.com/SkillCorner/opendata`, MIT) | **Verified:** 20 match folders in `data/matches/<id>/` (A-League 2024/25), each with `<id>_tracking_extrapolated.jsonl` (**Git LFS**, ≈90 MB each, 10 fps), `<id>_match.json`, `<id>_dynamic_events.csv`, `<id>_phases_of_play.csv`. Dynamic events include `event_type == player_possession` with `frame_start`, `frame_end`, `team_id`, plus `first/last_player_possession_in_team_possession` flags. Broadcast-derived, uncorrected ⇒ closest to ATD noise. | Possession labels (from `player_possession`), noise realism (missing players/ball patterns), robustness checks, extra training matches. | 10 fps vs 25 fps (resample); not all players visible per frame; ball/dead-ball periods are not explicit — derive `D` as frames between consecutive possession spells with a gap/ball-out evidence and **validate on Metrica**; needs `git lfs`. |
| **StatsBomb open data** (`github.com/statsbomb/open-data`) | Event data only (+ some 360 freeze frames), no continuous tracking. | **Priors only:** possession counts, duration distributions, turnover vs stoppage ratios, restart types for the decoder transition matrix. | Their "possession" definition differs from this challenge's. Do not use as frame-level labels. |
| **Other candidates to investigate** | Public/academic tracking+event releases (e.g. PFF FC World Cup 2022 tracking on request; a Bundesliga tracking/event dataset published with a Scientific Data paper). | Additional matches if licences allow. | Unverified from this file's author; check availability, licence, effort. |

### Deriving labels from Metrica events (algorithm sketch; **verified Game 1 event types:** PASS, RECOVERY, BALL LOST, CHALLENGE, SET PIECE, BALL OUT, SHOT, FAULT RECEIVED, CARD; subtypes include INTERCEPTION, THEFT, TACKLE-WON/LOST, AERIAL-WON/LOST, FREE KICK, THROW IN, GOAL KICK, CORNER KICK, KICK OFF, CLEARANCE, SAVED, GOAL; columns: Team, Type, Subtype, Period, Start/End Frame, From, To. Cross-check against `documentation/events-definitions.pdf`)
1. Parse events to `(team, type, subtype, start_frame, end_frame)`; ensure the frame base matches the tracking (both 25 fps).
2. **Alive/dead:** dead from `BALL OUT`, `FAULT RECEIVED`, goal-related events, until the restart `SET PIECE` (KICK OFF / THROW IN / GOAL KICK / CORNER KICK / FREE KICK) begins; alive otherwise.
3. **Possession owner while alive:** the team of the most recent *controlling* event (`PASS`, `SHOT`, `RECOVERY`, successful set-piece restarts), persisting through `CHALLENGE`/loose-ball events until the other team records a controlling event (`RECOVERY`/interception/successful theft). `BALL LOST` by team X ⇒ ownership is *pending* until the other team's control event. Mirror the challenge's rule: single uncontrolled touches and deflections are not changes.
4. **Sanity checks (must pass before training):** mean possession ≈ 10–20 s; every dead-ball stretch has a restart event after it; ball is slow/outside pitch on most `D` frames; label sequence has no illegal chains. Plot a few minutes and eyeball against tracking.
5. Label noise is expected. Treat these as **silver labels**; evaluate decoder logic with the same spellF1 implementation from `evaluation/possession_metrics.py`.

---

## 7A. External data setup — end to end (run now, before the match data arrives)

**Tested layout (2026-10-24):** Metrica sample-data ≈ 234 MB incl. `.git` (3 games, ~190 MB CSV/TXT). SkillCorner open data ≈ 131 MB of code/events + **20 tracking files stored in Git LFS (~90 MB each ⇒ expect ~2 GB total)**. StatsBomb is large; optional.

**Prerequisites (human):** `git` installed; **`git lfs`** installed (`https://git-lfs.com`, then once: `git lfs install`); ≥ 5 GB free disk (≥ 15 GB if you add StatsBomb); extra Python packages inside the venv:
`pip install lightgbm scipy pyarrow pytest ruff` (plus `kloppy` which `requirements-starter.txt` already installs).

**Run (from the repo root, venv active):**
```
python scripts/download_external.py                    # Metrica + SkillCorner (recommended now)
python scripts/download_external.py --with-statsbomb   # optional, large; priors only
```
The script clones shallowly into `external_data/<source>/`, pulls LFS files, writes `external_data/SOURCES.json` (URL + exact commit SHA + licence note), adds `external_data/` and `cache/` to `.gitignore`, and prints a verification summary (the tracking files must not be 133-byte LFS pointers).

**Expected result:**
```
external_data/
  SOURCES.json
  metrica_sample_data/data/Sample_Game_1/{..._RawTrackingData_Home_Team.csv, ..._Away_Team.csv, ..._RawEventsData.csv}
  metrica_sample_data/data/Sample_Game_2/{same three files}
  metrica_sample_data/data/Sample_Game_3/{Sample_Game_3_tracking.txt, Sample_Game_3_metadata.xml, Sample_Game_3_events.json}
  metrica_sample_data/documentation/events-definitions.pdf
  skillcorner_opendata/data/matches/<match_id>/{<id>_tracking_extrapolated.jsonl, <id>_match.json, <id>_dynamic_events.csv, <id>_phases_of_play.csv}   # x20
  skillcorner_opendata/data/matches.json
```
**Do not** place anything in `data/` (reserved for the competition bundle) and **do not** add anything about the Barcelona–Athletic match.

**Attribution / documentation duties (technical document):** list each dataset, URL, commit SHA (from `SOURCES.json`), licence note, which games were used for training vs validation, and how labels were derived. Metrica asks to acknowledge the source; SkillCorner is MIT (credit them); StatsBomb requires attribution (check its LICENSE.pdf).

**First loading checks (agent tasks):**
1. Metrica Game 3: `kloppy.metrica.load_tracking_epts(meta_data=..._metadata.xml, raw_data=..._tracking.txt)` — same reader family as the competition ATD; confirm frame rate 25, coords 0–1.
2. Metrica Games 1–2: parse CSVs (3 header rows) or use `kloppy.metrica.load_tracking_csv`; ball columns, `NaN`s, coordinate origin top-left.
3. SkillCorner: stream `*.jsonl` line by line; check ball/`player_data` structure and the `possession` field; join with `dynamic_events` on frame.
4. Plot 2–3 minutes of each source with derived labels; run the sanity checks in Section 7.

---

## 8. ATD-like noise simulator (build early, test with no match data)

Goal: make clean external tracking look like the competition ATD so models trained on it transfer. Calibrate parameters from the real ATD as soon as it is available; use defaults below until then.

* **Ball dropouts:** two-state Markov chain (visible/missing) with stationary missing ≈ 0.43 (all frames) / 0.41 (graded). With mean missing-run length `L` frames (start with L≈25, heavy-tailed; fit lognormal to real runs later), `P(miss→vis)=1/L`, `P(vis→miss)=(1/L)·π/(1−π)`. Make dropout probability higher when ball speed is high, ball is far from all players (air ball proxy), or many players are near (crowd).
* **Positional jitter:** ball σ≈0.3–0.5 m, players σ≈0.15–0.3 m (tune to real ATD smoothness).
* **Slot churn:** randomly retire/reassign track ids; drop individual players for 0.5–5 s; occasionally duplicate a player (+1 slot); rare swaps between same-team players; very rare cross-team label flips.
* **Missing-player fraction:** target mean ≈ 20.5 tracked objects.
* **Verification:** compare simulated vs real distributions (missing-run histogram, object count histogram, speed distributions). Report in the technical document.

### Decoder test harness without any match data
Create synthetic per-frame posteriors from the **external silver labels** (Metrica/SkillCorner; *not* the public GT, which may not be used for tuning): one-hot labels → add Dirichlet noise, random wrong-class bursts (1–5 s), and flat "blind" spells. Check that (a) raw argmax gets low spellF1, (b) the Viterbi decoder gets a large gain, (c) the gain is stable across parameter choices. This validates the decoder implementation and transition logic before any real data exists.

---

## 9. Decoder specification (reference implementation)

```python
# src/mtah/decode/viterbi.py
import numpy as np

STATES = ("H", "A", "D")            # indices 0, 1, 2

def build_log_transmat(mean_dur_frames, leave_share, eps=1e-12):
    """
    mean_dur_frames: dict {"H": ~355, "A": ~355, "D": estimated}
    leave_share: 3x3 row-stochastic matrix with zero diagonal, e.g.
        H -> [0, 0.5, 0.5]   (turnover vs stoppage; ESTIMATE from data)
        A -> [0.5, 0, 0.5]
        D -> [0.5, 0.5, 0]   (restart owner unknown; ESTIMATE)
    Self-transition = 1 - 1/mean_dur (geometric duration).
    """
    K = 3
    P = np.zeros((K, K))
    for i, s in enumerate(STATES):
        stay = 1.0 - 1.0 / float(mean_dur_frames[s])
        P[i, i] = stay
        P[i] += (1.0 - stay) * leave_share[i]
    return np.log(P + eps)

def viterbi_log(log_emit, log_trans, log_start):
    T, K = log_emit.shape
    dp = np.empty((T, K)); bp = np.empty((T, K), dtype=np.int8)
    dp[0] = log_start + log_emit[0]
    for t in range(1, T):
        scores = dp[t - 1][:, None] + log_trans          # K x K
        bp[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + log_emit[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = dp[-1].argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path

def scaled_log_emissions(post, prior, w, eps=1e-6):
    """
    post: (T,3) calibrated posteriors from S3; prior: (3,) class frequencies;
    w: (T,) reliability in (0,1]; low w flattens evidence (blind spells).
    """
    return w[:, None] * (np.log(post + eps) - np.log(prior + eps))
```

* Run **per half** over the graded block only (`half_video_blocks()`), never across halftime.
* Parameters to estimate (not guess): mean durations, leave shares, emission temperature (global multiplier on `w`), `min_len`. **Estimate from external silver labels only** (Metrica, SkillCorner, StatsBomb priors) and select by CV on a plateau. **Never from the public GT.**
* Order-of-magnitude sanity (from the organisers' README text, not to be tuned on): mean possession ≈ 14 s ≈ 350 frames ⇒ self-transition ≈ 0.997; the old draft's 0.985 (≈ 2.7 s) is far too jumpy. Actual values must come from external data.
* Post-processing helper (`min_spell_merge`) must preserve legal structure (never create a 1-frame D between same-team runs unless the evidence strongly supports it).

---

## 10. Validation protocol (non-negotiable)

1. **Use the repo's scorer.** Import from `evaluation/possession_metrics.py`; never reimplement spellF1/macroF1.
2. **Public window = pipeline correctness only** (frames aligned, file well-formed, labels mean what we think). It is **not a tuning set** (RULES.md §2): never choose between variants with it, never derive a parameter from it. (The repo shows it would pick the wrong smoothing window anyway.) A coarse "no catastrophic failure" check is fine.
3. **Primary model selection:** leave-one-match-out CV on external silver-labelled matches with the noise simulator (Section 8), scoring with the same metrics, reporting 1,000-sample bootstrap intervals over possessions. Prefer settings in a **flat region**, not the maximum.
4. **Decision rule:** adopt a change only if its bootstrap interval for ΔspellF1 excludes 0 on external CV. The public window may only veto a change that *catastrophically* breaks the pipeline (e.g. misalignment), never rank two working variants.
5. **Segment-count sanity on the real match** (uses only our own output; no GT; do not tune to it): number of possessions and `D` stretches must be plausible (order of tens of dead-ball stretches per half implied by ≈59 tagged restarts per match plus other stoppages; mean possession length in the same ballpark as 14 s — beware: 5 minutes is not necessarily representative). Thousands of segments ⇒ flicker; a few dozen ⇒ oversmoothing.
6. **Ablation table** for the technical document: baseline → +healing → +soft features → +GBDT → +decoder → +dead-ball cues/tags → +snapping → +detector → +VLM (if kept). Each row with CV score ± interval.
7. **Determinism:** fixed seeds; save config + git commit hash with every submission CSV (`submissions/<name>.json`).

---

## 11. Repository layout (additive; do not touch organisers' files)

```
src/mtah/
  config.py                 # dataclasses/YAML loading, paths via games.asset_paths()
  io/                       # ingest, alignment, graded mask, attack-direction normalisation
  healing/                  # short-gap imputation, [optional] detector wrapper, projection utils
  features/                 # feature groups A–H, each unit-tested
  labels/                   # silver label derivation from external events (Metrica, SkillCorner)
  sim/                      # ATD noise simulator + synthetic-posterior harness
  models/                   # logistic/LightGBM (3a), TCN/BiGRU (3b), calibration
  decode/                   # viterbi.py, post-processing, boundary snapping
  vlm_audit/                # optional, cached, gated
  eval/                     # thin wrappers around evaluation/possession_metrics.py, bootstrap CV runners
configs/                    # default.yaml, noise_sim.yaml, decoder.yaml
scripts/                    # 00_check_env, 01_build_silver_labels, 02_train, 03_decode, 04_export, 05_validate
tests/                      # pytest: features, viterbi, merge, simulator, alignment
docs/technical_report.md    # deliverable (Section 15)
cache/                      # gitignored: detections, features, vlm responses (commit only small VLM response logs if allowed)
submissions/                # CSV + json provenance (CSV may be committed; no match data)
external_data/              # gitignored; README listing sources, licences, download commands
```

Conventions: Python ≥3.9, type hints, `ruff`/`black`, `pytest`, no notebooks as the only record of an experiment (notebooks may exist for exploration but results must be reproducible from `scripts/`). A top-level `Makefile` or `run_all.sh` reproduces the final CSV from raw assets in one command. Add `requirements-solution.txt` (separate from the organisers' files) pinning added deps (e.g. `lightgbm`, `scipy`, `torch` if used, `ultralytics`/`opencv-python` only if the detector is adopted).

---

## 12. Work plan

### Phase A — NOW (no match data; ~Oct 24–28)
1. **Data access check (blocking, human task).** The README says data was released **Oct 20**. If you are enrolled and have not received a download link, check spam and email `hackathon.talentarena@mobileworldcapital.com`. If not enrolled, enrol immediately via the landing page in the README.
2. Read `FAQ.md`, `OVERVIEW.md`, and `evaluation/possession_metrics.py` (RULES.md/SUBMISSION.md are already incorporated). Skim the enrolment terms for cloud/API restrictions.
3. Run `python evaluation/run_evaluation.py`; confirm the four baselines reproduce the README numbers. Run `python evaluation/validate_submission.py` on a baseline CSV to see the output format and counts.
4. **Read the public GT only to understand label semantics** (how restarts, loose balls, D boundaries look) and to confirm the pipeline is correct. **Do not compute parameters from it.** Initialise decoder parameters from external data (Metrica/SkillCorner/StatsBomb priors).
5. Implement and unit-test: `viterbi.py`, `min_spell_merge`, export/validate wrapper, bootstrap CV runner.
6. Build the **synthetic-posterior decoder harness** (Section 8, built from external silver labels) and show the decoder beats raw argmax on spellF1 across a range of noise settings. *Acceptance:* large, stable spellF1 gain; no crashes on edge cases (all-X, single-class, short halves).
7. **Run `scripts/download_external.py`** (Section 7A) — Metrica + SkillCorner now, StatsBomb optional. Confirm `external_data/SOURCES.json` exists and no SkillCorner tracking file is an LFS pointer.
8. Implement label derivation from Metrica events; run the sanity checks in Section 7. *Acceptance:* plausible possession/dead statistics, plots reviewed.
9. Implement the **noise simulator** (defaults) and the feature extraction library with tests, run on Metrica Game 3 first (same format as competition ATD).
10. Train baseline (3a) on Metrica with simulated noise; evaluate leave-one-game-out + decoder. *Acceptance:* beats nearest-player-majority-vote baseline in the same CV setting with a bootstrap-significant gap.
11. Prepare the detector scripts (no weights committed) so they run the moment video arrives.

### Phase B — When the match data arrives (target within 48 h)
1. Run `starter/02_tracking_atd.ipynb` / `load_atd.py`; verify frame alignment, team assignment, graded masks, homography validity.
2. **Calibrate the noise simulator** to the real ATD (missing-run histogram, object counts, speeds). Re-train on externally derived data with calibrated noise.
3. Run S0–S5 end-to-end on the real match; produce first submission; validate; score on public window (correctness only). Compare to baselines. *Milestone: a validated CSV that beats 0.2433 spellF1 in external CV terms.*
4. Evaluate Smart Tagging usefulness (ablation on external-style CV where tags can be simulated/are absent ⇒ treat as optional features and test on the real match only through the segment-count sanity checks). Infer the meaning of the `Team` tag per code with the self-supervised proxy described in S2 Group G — **not** from the public GT.
5. Detector experiment (S1): inspect ball size on tactical frames; self-supervised recall test using ATD ball ↔ homography; decide go/no-go in ≤ 2 days of effort.

### Phase C — Improvement (≈ Nov 1–12)
* Sequence model (3b) with external pre-training; decide by CV.
* Boundary snapping; transition/duration refinement; segment-count sanity checks on the real match.
* Optional broadcast-ball fill; optional VLM audit (gated).
* Write the technical document continuously as experiments finish.

### Phase D — Freeze (≈ Nov 13–17)
* Freeze method by ~Nov 13. Final run from clean checkout via one command; save provenance JSON.
* **Reproducibility proof:** from a clean clone + data bundle run the one-command pipeline twice; the two CSV SHA-256 hashes must match, and the file must equal the one you upload.
* Validate the **exact upload file**, then submit via the Google Form (CSV `<team_name>.csv`, public repo URL, technical document `<team_name>.pdf`/`.md`) **before Nov 17**. You may edit the form response; the last one before the deadline counts. Submit early; there is no feedback after submission.

---

## 13. Compute notes

* S0–S5 on tabular features is light (CPU is fine; ~80k frames per half).
* The detector is the only heavy component: ≈64k graded frames with missing ball × tiles per frame. Estimate throughput on the owner's hardware first; if no suitable GPU, use a free/cheap GPU notebook environment (verify the match data licence permits cloud processing — the terms say data is licensed for this competition only; **[VERIFY]** before uploading data anywhere).
* Cache everything expensive (`cache/` parquet/npz) keyed by config hash.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overfitting / rule breach by tuning on the 17 public possessions | External-data CV; bootstrap; plateau selection; low-capacity models; public window only for correctness. |
| Domain gap external→competition | Calibrated noise simulator; Metrica Game 3 (same format); SkillCorner for noise realism; keep GBDT baseline as fallback. |
| Silver-label errors from events | Sanity checks; use for relative comparisons; do not over-claim absolute scores. |
| Detector fails (tiny ball) | Time-boxed experiment with explicit go/no-go; all other stages work without it. |
| Time (≈3 weeks) | Phase gating; first valid submission ASAP; freeze date Nov 13. |
| Rule violation | Section 2 checks; no hand labelling; no use of public GT for training/tuning/parameters; no human-produced data about the provided game; document every external dependency and licence; ask organisers (issue) when near a line. |
| Non-reproducible file (rule 3) | Pinned deps, seeds, deterministic ops, hash-compare two clean runs; VLM off by default; detector outputs stabilised. |
| Data-licence breach | Never commit data/derived frame-level exports; re-read enrolment terms before cloud/API use. |
| Vision-LLM non-determinism / cost | Optional, cached, temperature 0, adopt only on evidence. |
| Submission rejected by validator | Run `validate_submission.py` every time; keep graded/ungraded logic exact; 199,319 rows. |

---

## 15. Deliverables checklist (from `SUBMISSION.md`)

1. **Possession file `<team_name>.csv`** — header exactly `frame,label`; exactly **199,319** rows; `frame` = 0…199318 in order; labels in `{H,A,D,X}` (case/whitespace tolerant); no blank cells (write `X` to abstain); no index column. Validate with `python evaluation/validate_submission.py <file>`.
2. **Public code repository** (URL) — source, **pinned** dependencies, install/run instructions, the single command that regenerates the file **byte-for-byte**. **No match data, no frame-level derived exports, no clipped video.**
3. **Technical document `<team_name>.pdf` (or `.md`)** — short (a few pages). Must cover:
   * **Approach** — what was built and why that rather than alternatives.
   * **Assets used and how** — including those tried and **dropped** (e.g. Re-ID, VLM audit if disabled, broadcast alignment if abandoned).
   * **Training data** — every labelled dataset: source, who labelled it, how much; **none from the provided game**. (Here: Metrica sample games, SkillCorner A-League matches, StatsBomb priors; labels derived algorithmically from their events; commit SHAs from `external_data/SOURCES.json`.)
   * **Possession rules implemented** — where the line is drawn on loose balls, challenges, restarts, "control".
   * **Limitations** — failure cases and what you would fix with more time.
   * **Reproduction** — exact commands.
   * Plus (recommended): architecture diagram, noise-simulator calibration evidence, ablation table with bootstrap intervals, failure analysis, compliance statement (no hand annotation; no use of the public GT beyond correctness checks).
4. **Google Form** (link by email; Google account required): Team Name (as in registration), CSV, repo URL, technical document. One response per team; last response before **Nov 17, 2026** counts.
5. Team of 3–5 members for Innovation Challenge/Final Pitch eligibility.

---

## 16. Rules of engagement for the coding agent

* Prefer small, tested modules; run `pytest` and `ruff` before declaring a stage done.
* Every claim in a comment or doc about data must be traceable to the repo docs or measured on data; mark guesses as hypotheses.
* When uncertain about a rule, stop and ask the owner; do not assume permission.
* Log each experiment: config hash, data version, metric ± interval, decision. Keep `reports/experiments.md` up to date.
* **Never read, copy, or derive anything from `evaluation/gt/` for training, tuning or parameter choice.** Never ingest any external data about the provided game.
* Do not widen scope (no multi-object tracking rebuild, no Re-ID, no game-state reconstruction) — `CHALLENGE.md` explicitly states those are out of scope for Phase 1.
* If a stage fails its acceptance test, fix or simplify before moving on; a validated simple submission beats an unfinished ambitious one.

---

## Appendix A — Minimal skeletons

```python
# src/mtah/sim/noise.py  (ball dropout process)
import numpy as np

def simulate_ball_visibility(T, miss_frac=0.43, mean_miss_run=25, rng=None):
    rng = rng or np.random.default_rng(0)
    b = 1.0 / mean_miss_run                      # P(miss -> visible)
    a = b * miss_frac / (1.0 - miss_frac)        # P(visible -> miss)
    vis = np.ones(T, dtype=bool)
    for t in range(1, T):
        if vis[t - 1]:
            vis[t] = rng.random() >= a
        else:
            vis[t] = rng.random() < b
    return vis    # then modulate `a` by ball speed / isolation / crowding (see Section 8)
```

```python
# src/mtah/decode/postprocess.py  (min-spell merge, structure preserved)
import numpy as np

def runs(labels):
    """Return list of (start, end_exclusive, label) for a 1D int array."""
    idx = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, idx]; ends = np.r_[idx, len(labels)]
    return [(s, e, labels[s]) for s, e in zip(starts, ends)]

def min_spell_merge(labels, post, min_len=12):
    """Merge H/A runs shorter than min_len into the neighbour with larger summed posterior."""
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        R = runs(labels)
        for i, (s, e, lab) in enumerate(R):
            if lab in (0, 1) and (e - s) < min_len and 0 < i < len(R) - 1:
                left, right = R[i - 1], R[i + 1]
                cand = [left[2], right[2]]
                score = [post[s:e, c].sum() for c in cand]
                labels[s:e] = cand[int(np.argmax(score))]
                changed = True
                break
    return labels
```

```python
# scripts/04_export.py  (shape of final export)
# 1) allocate np.full(199_319, "X")
# 2) for each (half, first, last) in games.half_video_blocks("BAR-ATH"): fill labels on [first, last] from decoder
# 3) write frame,label CSV; run evaluation/validate_submission.py; save provenance JSON (git hash, config hash, seed)
```

## Appendix B — Quick reference of repo commands

```
pip install -r requirements.txt
python evaluation/run_evaluation.py                     # score everything in evaluation/results/ on the public window
python evaluation/run_evaluation.py --bootstrap 1000    # add intervals
python evaluation/validate_submission.py <file>.csv     # MUST pass before submitting
python starter/baseline.py                              # regenerate the four baselines
```
