"""Unit tests for scoring formulas, quality gates, and per-episode ranking."""

import pytest
from src.models import PromoCandidate, ColdReadScore, ScoredCandidate
from src.scoring import score_candidate, assign_per_episode_rankings, select_top_candidates


@pytest.fixture
def sample_candidate():
    return PromoCandidate(
        candidate_id="C_E01_01", episode_id=1, start_event_id="E01_001", end_event_id="E01_010",
        start_time=0.0, end_time=30.0, duration=30.0, hook_quote="Hook text",
        cliffhanger_quote="Cliffhanger text", spoiler_risk="low", proposer_reason="Good reason"
    )


def test_quality_gate_passed(sample_candidate):
    scores = ColdReadScore(
        hook_score=8.0, cliffhanger_score=8.5, standalone_score=7.0,
        emotion_score=7.5, promo_suitability=8.0, explanation="Strong clip"
    )
    # Weighted base (Section 3.I): (0.25*8 + 0.25*8.5 + 0.20*7 + 0.15*7.5 + 0.15*8.0)*10
    # = (2.0 + 2.125 + 1.4 + 1.125 + 1.2) * 10 = 7.85 * 10 = 78.50
    scored = score_candidate(sample_candidate, scores)
    assert scored.passed_gates is True
    assert scored.weighted_score == 78.50
    assert scored.spoiler_penalty == 0.0
    assert scored.final_score == 78.50


def test_quality_gate_failed_hook(sample_candidate):
    scores = ColdReadScore(
        hook_score=6.5,  # Below 7.0!
        cliffhanger_score=8.5, standalone_score=7.0, emotion_score=7.5,
        coherence_score=8.0, explanation="Weak hook"
    )
    scored = score_candidate(sample_candidate, scores)
    assert scored.passed_gates is False
    assert "Hook score 6.5" in scored.rejection_reason


def test_spoiler_penalty(sample_candidate):
    sample_candidate.spoiler_risk = "high"
    scores = ColdReadScore(
        hook_score=8.0, cliffhanger_score=8.0, standalone_score=8.0,
        emotion_score=8.0, coherence_score=8.0, explanation="High spoiler"
    )
    scored = score_candidate(sample_candidate, scores)
    assert scored.passed_gates is True
    assert scored.weighted_score == 80.0
    assert scored.spoiler_penalty == 10.0
    assert scored.final_score == 70.0


def test_per_episode_ranking():
    c1 = PromoCandidate(
        candidate_id="C_E01_01", episode_id=1, start_event_id="E01_001", end_event_id="E01_010",
        start_time=0.0, end_time=30.0, duration=30.0, hook_quote="H1", cliffhanger_quote="C1",
        spoiler_risk="low", proposer_reason="R1"
    )
    c2 = PromoCandidate(
        candidate_id="C_E01_02", episode_id=1, start_event_id="E01_015", end_event_id="E01_025",
        start_time=40.0, end_time=70.0, duration=30.0, hook_quote="H2", cliffhanger_quote="C2",
        spoiler_risk="low", proposer_reason="R2"
    )
    sc1 = ScoredCandidate(candidate=c1, final_score=75.0, passed_gates=True)
    sc2 = ScoredCandidate(candidate=c2, final_score=85.0, passed_gates=True)

    ranked = assign_per_episode_rankings([sc1, sc2])
    # sc2 has higher score -> rank 1, sc1 -> rank 2
    by_id = {item.candidate.candidate_id: item.per_episode_rank for item in ranked}
    assert by_id["C_E01_02"] == 1
    assert by_id["C_E01_01"] == 2

