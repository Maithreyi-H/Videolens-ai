import json
import pytest
from pathlib import Path
from app import (
    parse_claims,
    load_records,
    save_records,
    load_description_reviews,
    save_description_reviews,
    video_metadata,
    extract_frame,
)

def test_parse_claims_basic_sentences():
    text = "A person enters the room. They pick up a red bag. They exit quickly."
    claims = parse_claims(text)
    assert len(claims) == 3
    assert claims[0] == "A person enters the room."
    assert claims[1] == "They pick up a red bag."
    assert claims[2] == "They exit quickly."

def test_parse_claims_numbered_list():
    text = "1. First action occurs.\n2. Second action occurs.\n3. Third action occurs."
    claims = parse_claims(text)
    assert len(claims) == 3
    assert claims[0] == "First action occurs."
    assert claims[1] == "Second action occurs."
    assert claims[2] == "Third action occurs."

def test_parse_claims_bullet_points():
    text = "- Subject walks to table\n- Subject sets down coffee mug\n* Mug has white logo"
    claims = parse_claims(text)
    assert len(claims) == 3
    assert claims[0] == "Subject walks to table"
    assert claims[1] == "Subject sets down coffee mug"
    assert claims[2] == "Mug has white logo"

def test_parse_claims_decimal_numbers():
    text = "At 2.5 seconds, the ball bounces. It rolls 1.8 meters."
    claims = parse_claims(text)
    assert len(claims) == 2
    assert "2.5 seconds" in claims[0]
    assert "1.8 meters" in claims[1]

def test_parse_claims_empty_or_whitespace():
    assert parse_claims("") == []
    assert parse_claims("   \n\t  ") == []

def test_records_roundtrip(tmp_path, monkeypatch):
    test_file = tmp_path / "annotations.json"
    import app
    monkeypatch.setattr(app, "ANNOTATIONS_FILE", test_file)

    initial = load_records()
    assert initial == []

    records = [
        {"id": "ann-1", "label": "person walks", "confidence": "High", "review_status": "Reviewed"}
    ]
    assert save_records(records) is True
    loaded = load_records()
    assert loaded == records

def test_corrupt_records_handling(tmp_path, monkeypatch):
    test_file = tmp_path / "annotations.json"
    test_file.write_text("{invalid json", encoding="utf-8")
    import app
    monkeypatch.setattr(app, "ANNOTATIONS_FILE", test_file)

    loaded = load_records()
    assert loaded == []

def test_description_reviews_roundtrip(tmp_path, monkeypatch):
    test_file = tmp_path / "description_reviews.json"
    import app
    monkeypatch.setattr(app, "REVIEWS_FILE", test_file)

    assert load_description_reviews() == []

    reviews = [{
        "id": "rev-1",
        "video_name": "test.mp4",
        "claims": [{"claim": "Person runs", "verdict": "Supported", "rationale": "Visible at 1.2s"}]
    }]
    assert save_description_reviews(reviews) is True
    assert load_description_reviews() == reviews

def test_corrupt_description_reviews_handling(tmp_path, monkeypatch):
    test_file = tmp_path / "description_reviews.json"
    test_file.write_text("corrupted content", encoding="utf-8")
    import app
    monkeypatch.setattr(app, "REVIEWS_FILE", test_file)

    assert load_description_reviews() == []

def test_video_metadata_invalid_input():
    meta = video_metadata(b"not a valid video stream")
    assert meta["is_valid"] is False
    assert meta["duration"] == 0.0

def test_extract_frame_invalid_input():
    frame = extract_frame(b"not a valid video stream", 0.0)
    assert frame is None

def test_video_metadata_and_frame_extraction_real_video():
    video_path = Path("data/demo_clip.mp4")
    assert video_path.exists()
    meta = video_metadata(video_path)
    assert meta["is_valid"] is True
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["frames"] == 60
    assert meta["fps"] == 20.0
    assert meta["duration"] == 3.0

    # Extract frame at 1.0s
    frame = extract_frame(video_path, 1.0)
    assert frame is not None
    assert frame.shape == (240, 320, 3)

    # Also test byte input with real video bytes
    raw_bytes = video_path.read_bytes()
    meta_bytes = video_metadata(raw_bytes)
    assert meta_bytes["is_valid"] is True
    assert meta_bytes["frames"] == 60

    frame_bytes = extract_frame(raw_bytes, 1.5)
    assert frame_bytes is not None
    assert frame_bytes.shape == (240, 320, 3)

