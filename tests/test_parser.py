"""Unit tests for ScreenplayParser."""

import pytest
from pathlib import Path
from src.parser import ScreenplayParser, parse_timestamp, format_timestamp


@pytest.fixture
def parser():
    return ScreenplayParser()


def test_parse_timestamp():
    assert parse_timestamp("00:04.80") == 4.80
    assert parse_timestamp("(01:23.45)") == 83.45
    assert parse_timestamp("00:00.00") == 0.0
    assert parse_timestamp("01:00:15.50") == 3615.50
    assert parse_timestamp("12.34") == 12.34


def test_format_timestamp():
    assert format_timestamp(4.80) == "00:04.80"
    assert format_timestamp(83.45) == "01:23.45"
    assert format_timestamp(0.0) == "00:00.00"


def test_parse_episode_1(parser):
    ep1_path = Path("I Married a Broken Soldier. He was a Duke/episode-001-Episode 1671703.txt")
    if not ep1_path.exists():
        pytest.skip("Input pack not found")

    episode = parser.parse_file(ep1_path)
    assert episode.episode_id == 1
    assert "Juliana" in episode.logline
    assert len(episode.scenes) >= 1
    assert len(episode.events) > 50

    # Test event attributes
    ev1 = episode.events[0]
    assert ev1.event_id == "E01_001"
    assert ev1.event_type == "camera"
    assert ev1.start_time == 0.0
    assert ev1.end_time == 4.80

    # Find dialogue event
    dialogues = [e for e in episode.events if e.event_type == "dialogue"]
    assert len(dialogues) > 10
    d1 = dialogues[0]
    assert d1.character is not None
    assert len(d1.text) > 0


def test_parse_all_50_episodes(parser):
    pack_dir = Path("I Married a Broken Soldier. He was a Duke")
    if not pack_dir.exists():
        pytest.skip("Input pack not found")

    files = sorted(list(pack_dir.glob("*.txt")))
    assert len(files) == 50, f"Expected 50 files, got {len(files)}"

    for f in files:
        ep = parser.parse_file(f)
        assert ep.episode_id >= 1
        assert len(ep.events) > 10, f"Episode {ep.episode_id} parsed fewer than 10 events"
        # Verify event ordering
        for i in range(len(ep.events) - 1):
            assert ep.events[i].start_time <= ep.events[i + 1].start_time + 5.0, (
                f"Severe timestamp reversal in {f.name}: "
                f"{ep.events[i].event_id} ({ep.events[i].start_time}) vs "
                f"{ep.events[i+1].event_id} ({ep.events[i+1].start_time})"
            )


def test_malformed_screenplay_handling(parser):
    malformed_text = """
    EPISODE NUMBER: Episode 99
    
    SCENE 01
    Scene heading: EXT. CASTLE - DUSK
    Scene start: 00:00.00
    Scene end: 00:45.00
    
    **CAMERA:**
    Start: (00:00.00) | End: (00:05.00)
    Establishing shot with weird whitespace and missing colons.
    
    **KING (Furious):**
    Start: (00:05.10) | End: (00:09.50)
    "How dare you defy me!"
    
    **ACTION**
    Start: (00:09.50) | End: (00:12.00)
    He draws his sword in anger.
    """
    ep = parser.parse_text(malformed_text, fallback_episode_id=99)
    assert ep.episode_id == 99
    assert len(ep.events) == 3
    assert ep.events[0].event_type == "camera"
    assert ep.events[1].event_type == "dialogue"
    assert ep.events[1].character == "KING"
    assert ep.events[1].parenthetical == "Furious"
    assert ep.events[1].text == "How dare you defy me!"
    assert ep.events[2].event_type == "action"

