"""Test SkillCorner loader and label sanity against Metrica standards."""

from pathlib import Path

from mtah.labels.metrica import validate_derived_labels
from mtah.labels.skillcorner import load_skillcorner_labels

ROOT = Path(__file__).resolve().parents[1]


def test_skillcorner_loader_match_sanity():
    match_dir = ROOT / "external_data/skillcorner_opendata/data/matches/1874553"
    if not match_dir.exists():
        return

    labels_25, meta = load_skillcorner_labels(match_dir, target_fps=25, source_fps=10)
    assert len(labels_25) > 100000
    assert meta["home_name"] == "Brisbane Roar FC"
    assert meta["away_name"] == "Adelaide United Football Club"

    # Evaluate label statistics on in-play frames (excluding X)
    in_play = labels_25 != 3
    in_play_labels = labels_25[in_play]
    stats = validate_derived_labels(in_play_labels, fps=25)

    print(f"\nSkillCorner 1874553 stats: {stats}")
    # Mean duration between 8 and 30s
    assert 8.0 <= stats["mean_duration_s"] <= 30.0
    # Dead ball fraction between 20% and 65%
    assert 0.20 <= stats["dead_fraction"] <= 0.65
    assert stats["is_plausible"] is True
