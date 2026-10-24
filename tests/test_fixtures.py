"""Unit tests for synthetic homography CSV and Smart Tagging XML fixtures and encoders."""

import numpy as np

from mtah.io.context_loaders import (
    encode_smart_tagging_features,
    load_homography_csv,
    load_smart_tagging_xml,
)

SAMPLE_HOMOGRAPHY_CSV = """frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid
0,1.5,0.0,10.0,0.0,1.5,20.0,0.0,0.0,1.0,0.21,1
1,1.6,0.0,10.5,0.0,1.6,20.5,0.0,0.0,1.0,0.25,1
2,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,12.5,0
"""

SAMPLE_SMART_TAGGING_XML = """<?xml version="1.0" encoding="utf-8"?>
<file>
  <ALL_INSTANCES>
    <instance>
      <ID>1</ID>
      <start>0.40</start>
      <end>2.00</end>
      <code>SET PIECES</code>
      <label>
        <group>Type</group>
        <text>Kick Off (Start)</text>
      </label>
      <label>
        <group>Team</group>
        <text>Barcelona</text>
      </label>
    </instance>
    <instance>
      <ID>2</ID>
      <start>5.00</start>
      <end>8.50</end>
      <code>PROGRESSION</code>
      <label>
        <group>Team</group>
        <text>Athletic Club</text>
      </label>
    </instance>
  </ALL_INSTANCES>
</file>
"""


def test_homography_loader_with_fixture(tmp_path):
    csv_file = tmp_path / "test_homography.csv"
    csv_file.write_text(SAMPLE_HOMOGRAPHY_CSV)

    res = load_homography_csv(csv_file)
    assert len(res["valid"]) == 3
    assert res["valid"][0] == True
    assert res["valid"][1] == True
    assert res["valid"][2] == False

    assert res["reprojection_error"][0] == 0.21
    assert res["matrices"].shape == (3, 3, 3)
    assert res["matrices"][0, 0, 0] == 1.5
    assert np.isnan(res["matrices"][2, 0, 0])


def test_smart_tagging_loader_and_encoder_with_fixture(tmp_path):
    xml_file = tmp_path / "test_smart_tagging.xml"
    xml_file.write_text(SAMPLE_SMART_TAGGING_XML)

    instances = load_smart_tagging_xml(xml_file, fps=25)
    assert len(instances) == 2
    assert instances[0]["code"] == "SET PIECES"
    assert instances[0]["start_frame"] == 10  # 0.40 * 25
    assert instances[0]["end_frame"] == 50    # 2.00 * 25
    assert instances[0]["tags"]["Team"] == "Barcelona"

    assert instances[1]["start_frame"] == 125 # 5.00 * 25
    assert instances[1]["end_frame"] == 212   # 8.50 * 25 (rounded)

    feats = encode_smart_tagging_features(instances, n_frames=250)
    assert len(feats) == 250
    assert "has_smart_tag" in feats.columns
    assert "is_set_piece" in feats.columns
    assert "tag_team" in feats.columns
    assert "dist_to_set_piece" in feats.columns

    # Frame 25 is within instance 1 (Set piece, Barcelona = 0)
    assert feats["is_set_piece"].iloc[25] == 1.0
    assert feats["tag_team"].iloc[25] == 0.0

    # Frame 150 is within instance 2 (Progression, Athletic = 1)
    assert feats["is_set_piece"].iloc[150] == 0.0
    assert feats["tag_team"].iloc[150] == 1.0
