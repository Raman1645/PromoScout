"""Unit tests for boundary validation, trimming, and post-repair checks."""

import pytest
from src.models import Episode, Scene, ScreenplayEvent, CandidateProposal
from src.validator import BoundaryValidator


@pytest.fixture
def sample_episode():
    # Construct a synthetic episode with known timestamps
    events = [
        ScreenplayEvent(
            event_id="E01_001", episode_id=1, scene_num=1, event_type="camera",
            start_time=0.0, end_time=2.0, text="Wide shot of cathedral."
        ),
        ScreenplayEvent(
            event_id="E01_002", episode_id=1, scene_num=1, event_type="dialogue",
            character="JULIANA", start_time=2.1, end_time=4.5, text="Gerald, what is the meaning of this?"
        ),
        ScreenplayEvent(
            event_id="E01_003", episode_id=1, scene_num=1, event_type="dialogue",
            character="GERALD", start_time=4.8, end_time=15.0, text="You were never worthy of my family name."
        ),
        ScreenplayEvent(
            event_id="E01_004", episode_id=1, scene_num=1, event_type="dialogue",
            character="CLARISSA", start_time=15.2, end_time=25.0, text="He is marrying me now, sister."
        ),
        ScreenplayEvent(
            event_id="E01_005", episode_id=1, scene_num=1, event_type="dialogue",
            character="JULIANA", start_time=25.2, end_time=32.0, text="Then may heaven curse you both!"
        ),
        ScreenplayEvent(
            event_id="E01_006", episode_id=1, scene_num=1, event_type="camera",
            start_time=32.0, end_time=35.0, text="Close up on guests gasping."
        ),
    ]
    return Episode(
        episode_id=1, episode_number_raw="Episode 1", logline="Altar betrayal",
        continuity_notes="", scenes=[Scene(scene_num=1, heading="INT. CATHEDRAL", start_time=0.0, end_time=35.0)],
        events=events
    )


def test_valid_proposal(sample_episode):
    validator = BoundaryValidator(min_duration=20.0, max_duration=60.0)
    # Clip from E01_001 (0.0s) to E01_005 (32.0s) -> 32.0s duration
    proposal = CandidateProposal(
        start_event_id="E01_001",
        end_event_id="E01_005",
        hook_quote="Gerald, what is the meaning of this?",
        cliffhanger_quote="Then may heaven curse you both!",
        spoiler_risk="low",
        proposer_reason="High stakes public betrayal."
    )
    cand, err = validator.validate_candidate(proposal, sample_episode, 1)
    assert cand is not None
    assert err is None
    assert cand.duration == 32.0
    assert cand.start_time == 0.0
    assert cand.end_time == 32.0


def test_short_clip_rejected(sample_episode):
    validator = BoundaryValidator(min_duration=20.0, max_duration=60.0)
    # E01_001 to E01_003 is 15.0s -> under 20s
    proposal = CandidateProposal(
        start_event_id="E01_001",
        end_event_id="E01_003",
        hook_quote="Gerald, what is the meaning of this?",
        cliffhanger_quote="You were never worthy of my family name.",
        spoiler_risk="low",
        proposer_reason="Short clip"
    )
    cand, err = validator.validate_candidate(proposal, sample_episode, 1)
    assert cand is None
    assert "under minimum" in err


def test_hook_too_late_rejected(sample_episode):
    validator = BoundaryValidator(min_duration=20.0, max_duration=60.0)
    # If hook quote is in E01_003 (starts at 4.8s, which is > 3.0s from start 0.0s)
    proposal = CandidateProposal(
        start_event_id="E01_001",
        end_event_id="E01_005",
        hook_quote="You were never worthy",
        cliffhanger_quote="Then may heaven curse you both!",
        spoiler_risk="low",
        proposer_reason="Late hook"
    )
    cand, err = validator.validate_candidate(proposal, sample_episode, 1)
    assert cand is None
    assert "exceeds 3.0s limit" in err


def test_cliffhanger_too_early_rejected(sample_episode):
    validator = BoundaryValidator(min_duration=20.0, max_duration=60.0)
    # Clip ends at E01_006 (35.0s), but cliffhanger is in E01_004 (ends at 25.0s, offset = 10.0s > 5.0s)
    proposal = CandidateProposal(
        start_event_id="E01_001",
        end_event_id="E01_006",
        hook_quote="Gerald, what is the meaning",
        cliffhanger_quote="He is marrying me now",
        spoiler_risk="low",
        proposer_reason="Early cliffhanger"
    )
    cand, err = validator.validate_candidate(proposal, sample_episode, 1)
    assert cand is None
    assert "exceeds 5.0s limit" in err

