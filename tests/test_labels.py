"""Unit tests for silver label derivation from Metrica events."""

import pandas as pd
import pytest

from mtah.labels.metrica import (
    derive_metrica_labels,
    validate_derived_labels,
)


def test_derive_metrica_labels_synthetic():
    # Sequence of events:
    # 1. Kickoff by Home at frame 10 (until then dead)
    # 2. Pass by Home at frame 100
    # 3. Interception by Away at frame 200
    # 4. Ball Out at frame 300
    # 5. Throw-in by Away at frame 400
    events = pd.DataFrame([
        {"team": 0, "type": "SET PIECE", "subtype": "KICK OFF", "period": 1, "start_frame": 10, "end_frame": 12},
        {"team": 0, "type": "PASS", "subtype": "", "period": 1, "start_frame": 100, "end_frame": 120},
        {"team": 1, "type": "RECOVERY", "subtype": "INTERCEPTION", "period": 1, "start_frame": 200, "end_frame": 200},
        {"team": None, "type": "BALL OUT", "subtype": "", "period": 1, "start_frame": 300, "end_frame": 300},
        {"team": 1, "type": "SET PIECE", "subtype": "THROW IN", "period": 1, "start_frame": 400, "end_frame": 405},
    ])

    labels = derive_metrica_labels(events, n_frames=500)
    assert len(labels) == 500

    # Before kickoff: Dead (2)
    assert labels[0] == 2
    assert labels[8] == 2

    # After kickoff: Home (0)
    assert labels[15] == 0
    assert labels[150] == 0

    # After interception: Away (1)
    assert labels[250] == 1

    # After ball out: Dead (2)
    assert labels[350] == 2

    # After throw in: Away (1)
    assert labels[450] == 1


def test_validate_derived_labels():
    # Create simple labels with plausible duration
    # 20 possessions of 300 frames (~12 s) separated by 100 frames dead (~4 s)
    seq = []
    for i in range(20):
        team = i % 2
        seq.extend([team] * 300)
        seq.extend([2] * 100)
    labels = pd.Series(seq).to_numpy()

    stats = validate_derived_labels(labels)
    assert stats["is_plausible"] is True
    assert stats["n_spells"] == 20
    assert stats["mean_duration_s"] == pytest.approx(12.0)
    assert stats["dead_fraction"] == pytest.approx(0.25)
