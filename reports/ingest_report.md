# Ingest Verification Report: BAR-ATH Tactical

## 1. Team Mapping & Metadata
- **Home Team ID:** `MET_user_team_o4Hossvb8M` (Barcelona)
- **Away Team ID:** `MET_user_team_x7C4FFX2HF` (Atletic Club)
- **Home = Barcelona Mapping:** MATCH [OK]
- **Metadata Pitch Dimensions:** 105m x 68m (games.py: 105.3m x 68.0m)
- **Metadata FrameRate:** 25.0 fps

## 2. Tracking Alignment & Graded Blocks
- **Expected Total Frames:** 199319
- **Half 1 Graded Block:** video 11185..90651 (79467 frames)
- **Half 2 Graded Block:** video 110808..190197 (79390 frames)
- **Total Graded Frames:** 158857 (expected: 158,857)
- **Actual Tracking Records:** 199319
- **Frame ID Range:** 1..199319 (contiguous: True)
- **Alignment (video_frame = frame_id - 1):** MATCH [OK]
- **Ball Missing on Graded Frames:** 64643 / 158857 (40.69%, expected ~41%)
- **Mean Tracked Players on Graded Frames:** 20.49 (expected ~20.5)
- **Mean Tracked Objects (Players + Ball) on Graded Frames:** 21.08 (21-22 during play)

## 3. Homography Validity
- **Homography Rows:** 199319 (expected 199319)
- **Valid Homographies:** 199319 / 199319 (100.00%, expected 100.0%)

## 4. Smart Tagging Summary
- **Instance Count:** 638 (expected 638)
- **Team Tag Breakdown:**
  - `Barcelona`: 314
  - `Atletic Club`: 288
  - `N/A`: 36
- **Top Event Codes:**
  - `PLAYERS IN THE BOX`: 97
  - `BALL IN FINAL THIRD`: 93
  - `BALL IN THE BOX`: 72
  - `PROGRESSION`: 66
  - `DEFENDING IN MIDDLE THIRD`: 66
  - `SET PIECES`: 59
  - `CREATING CHANCES`: 30
  - `DEFENDING IN DEFENSIVE THIRD`: 30

## 5. Verification Verdict
- [x] Frame count is 199,319: **PASSED**
- [x] Frame alignment is frame_id - 1: **PASSED**
- [x] Graded blocks match games.py: **PASSED**
- [x] Homography 100% valid on tactical: **PASSED**
- [x] Home is Barcelona: **PASSED**
- [x] Ball missing ~41%: **PASSED**
- [x] Mean tracked players ~20.5 (20.49): **PASSED**
- [x] Smart Tagging instances 638: **PASSED**

**Overall Status:** ALL INGEST CHECKS PASSED [OK]
