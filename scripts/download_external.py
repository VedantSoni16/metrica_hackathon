#!/usr/bin/env python3
"""
Download the EXTERNAL datasets used for training / validation (other games only).

Put this file at:  scripts/download_external.py   (inside your more-than-a-hack-2027 clone)
Run from repo root:
    python scripts/download_external.py                 # Metrica + SkillCorner
    python scripts/download_external.py --with-statsbomb # also StatsBomb (priors only, large)
    python scripts/download_external.py --skip-skillcorner

What it does
  * clones each repo (shallow) into ./external_data/<name>
  * for SkillCorner, pulls the Git-LFS tracking files (needs `git lfs`)
  * records the exact commit SHA of every source in external_data/SOURCES.json
    (the organisers must be able to access what you used -> pin it and cite it)
  * NEVER touches ./data (that folder is for the competition match assets)

External data must be about OTHER games. Never place anything about the
Barcelona-Athletic 2024-25 match (events, possession, corrected tracking) here:
RULES.md section 2 forbids any human-produced data about the provided game.
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parent.name == "scripts") else Path.cwd()
EXT = ROOT / "external_data"

SOURCES = {
    "metrica_sample_data": {
        "url": "https://github.com/metrica-sports/sample-data.git",
        "what": "3 anonymised games, 25 Hz tracking + events (Game 3 in EPTS/FIFA format)",
        "licence_note": "No licence file in repo; README asks to acknowledge Metrica Sports and use responsibly.",
        "lfs": False,
    },
    "skillcorner_opendata": {
        "url": "https://github.com/SkillCorner/opendata.git",
        "what": "A-League 2024/25 broadcast tracking (10 fps) + dynamic events + phases of play",
        "licence_note": "MIT (repo LICENSE). Credit SkillCorner.",
        "lfs": True,
    },
    "statsbomb_open_data": {
        "url": "https://github.com/statsbomb/open-data.git",
        "what": "Event data only (no tracking). Used for duration/turnover PRIORS only.",
        "licence_note": "See LICENSE.pdf in repo; attribution to StatsBomb required.",
        "lfs": False,
    },
}


def run(cmd, cwd=None, check=True):
    print("$", " ".join(map(str, cmd)))
    return subprocess.run(cmd, cwd=cwd, check=check)


def need(tool):
    if shutil.which(tool) is None:
        sys.exit(f"ERROR: '{tool}' not found on PATH. Install it first (see strat.md, Section 7A).")


def lfs_available():
    r = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True)
    return r.returncode == 0


def clone(name, info, sparse=None):
    dest = EXT / name
    if dest.exists():
        print(f"[skip] {name} already exists at {dest} (delete the folder to re-download)")
    else:
        cmd = ["git", "clone", "--depth", "1"]
        if sparse:
            cmd += ["--filter=blob:none", "--sparse"]
        cmd += [info["url"], str(dest)]
        run(cmd)
        if sparse:
            run(["git", "sparse-checkout", "set", *sparse], cwd=dest)
    if info["lfs"]:
        if not lfs_available():
            sys.exit("ERROR: git-lfs is required for SkillCorner tracking files.\n"
                     "Install: https://git-lfs.com  then run:  git lfs install   and re-run this script.")
        run(["git", "lfs", "install", "--local"], cwd=dest)
        run(["git", "lfs", "pull"], cwd=dest)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True).stdout.strip()
    return {"path": str(dest.relative_to(ROOT)), "commit": sha, "url": info["url"],
            "what": info["what"], "licence_note": info["licence_note"],
            "downloaded": str(date.today())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-statsbomb", action="store_true")
    ap.add_argument("--skip-skillcorner", action="store_true")
    ap.add_argument("--skip-metrica", action="store_true")
    a = ap.parse_args()

    need("git")
    EXT.mkdir(exist_ok=True)
    manifest_path = EXT / "SOURCES.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if not a.skip_metrica:
        manifest["metrica_sample_data"] = clone("metrica_sample_data", SOURCES["metrica_sample_data"])
    if not a.skip_skillcorner:
        manifest["skillcorner_opendata"] = clone("skillcorner_opendata", SOURCES["skillcorner_opendata"])
    if a.with_statsbomb:
        # cone-mode sparse checkout: directories only; data/competitions.json comes along automatically
        manifest["statsbomb_open_data"] = clone("statsbomb_open_data", SOURCES["statsbomb_open_data"],
                                                sparse=["data/matches", "data/events"])

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("\nWrote", manifest_path)

    # safety: make sure external_data/ is git-ignored
    gi = ROOT / ".gitignore"
    txt = gi.read_text() if gi.exists() else ""
    if "external_data/" not in txt:
        gi.write_text(txt.rstrip() + "\n\n# external datasets (not redistributed from this repo)\nexternal_data/\ncache/\n")
        print("Added external_data/ and cache/ to .gitignore")

    # quick verification
    print("\nVerification:")
    m = EXT / "metrica_sample_data" / "data"
    for g in (1, 2):
        f = m / f"Sample_Game_{g}" / f"Sample_Game_{g}_RawTrackingData_Home_Team.csv"
        print(f"  Metrica Game {g} tracking:", "OK" if f.exists() else "MISSING")
    g3 = m / "Sample_Game_3" / "Sample_Game_3_tracking.txt"
    print("  Metrica Game 3 tracking:", "OK" if g3.exists() else "MISSING")
    sc = EXT / "skillcorner_opendata" / "data" / "matches"
    if sc.exists():
        small = [p for p in sc.glob("*/*_tracking_extrapolated.jsonl") if p.stat().st_size < 1000]
        n = len(list(sc.glob("*/*_tracking_extrapolated.jsonl")))
        print(f"  SkillCorner tracking files: {n} found, {len(small)} still LFS pointers"
              + ("  <-- run `git lfs pull` inside that folder" if small else "  (OK)"))


if __name__ == "__main__":
    main()