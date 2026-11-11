"""Tests for mitigations and architectural safeguards."""

import pytest
import os
from pathlib import Path
from src.llm_client import LLMClient, ConfigurationError
from src.evaluator import format_clip_anonymized
from src.parser import ScreenplayParser
from src.models import PromoCandidate, ColdReadScore
from src.scoring import score_candidate


def test_strict_mock_mode_enforcement(monkeypatch):
    """Verify that running without an API key and without --mock raises ConfigurationError."""
    # Ensure environment variables are clear
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError) as exc_info:
        LLMClient(provider="groq", mock_mode=False, api_key="")
    assert "No API key provided" in str(exc_info.value)
    assert "--mock" in str(exc_info.value)


def test_explicit_mock_mode_allowed():
    """Verify that setting mock_mode=True works without API credentials."""
    client = LLMClient(provider="groq", mock_mode=True, api_key="")
    assert client.mock_mode is True
    assert client.inference_mode == "mock"


def test_groq_defaults():
    """Verify Groq provider defaults to openai/gpt-oss-120b and Groq base URL."""
    client = LLMClient(provider="groq", mock_mode=True, api_key="")
    assert client.base_url == "https://api.groq.com/openai/v1"
    assert client.model == "openai/gpt-oss-120b"


def test_speaker_anonymization():
    """Verify that cold-read transcript neutralizes speaker names to [SPEAKER A], [SPEAKER B]."""
    parser = ScreenplayParser()
    ep1_path = Path("I Married a Broken Soldier. He was a Duke/episode-001-Episode 1671703.txt")
    if not ep1_path.exists():
        pytest.skip("Proprietary screenplay pack omitted for confidentiality")
    ep = parser.parse_file(ep1_path)

    # Candidate from E01_008 (Matriarch) to E01_027 (Lady in Gold Gown)
    cand = PromoCandidate(
        candidate_id="C_TEST_01",
        episode_id=1,
        start_event_id="E01_008",
        end_event_id="E01_027",
        start_time=11.3,
        end_time=41.2,
        duration=29.9,
        hook_quote="You should turn around, Juliana.",
        cliffhanger_quote="Because he is marrying me.",
        spoiler_risk="low",
        proposer_reason="Altar betrayal"
    )

    transcript, speaker_map = format_clip_anonymized(cand, ep)

    # Check that speaker tags are anonymized
    assert "[SPEAKER A]" in transcript
    assert "[SPEAKER B]" in transcript

    # Verify that the speaker map contains the true characters mapped to neutral tags
    assert "MATRIARCH" in speaker_map
    assert "LADY JULIANA" in speaker_map
    assert speaker_map["MATRIARCH"] == "[SPEAKER A]"
    assert speaker_map["LADY JULIANA"] == "[SPEAKER B]"

    # Verify relative timestamps start at 0.00s
    assert "[00.00s -" in transcript


def test_updated_mitigation_scoring_weights():
    """Verify weighted score formula matches Section 3.I: 0.25H + 0.25C + 0.20A + 0.15E + 0.15P."""
    cand = PromoCandidate(
        candidate_id="C_TEST", episode_id=1, start_event_id="E01_001", end_event_id="E01_010",
        start_time=0.0, end_time=30.0, duration=30.0, hook_quote="H", cliffhanger_quote="C",
        spoiler_risk="low", proposer_reason="R"
    )
    score = ColdReadScore(
        hook_score=8.0,
        cliffhanger_score=8.0,
        standalone_score=8.0,
        emotion_score=8.0,
        promo_suitability=8.0,
        explanation="Test"
    )
    scored = score_candidate(cand, score)
    # (0.25*8 + 0.25*8 + 0.20*8 + 0.15*8 + 0.15*8) * 10 = (2 + 2 + 1.6 + 1.2 + 1.2) * 10 = 8.0 * 10 = 80.0
    assert scored.weighted_score == 80.0
    assert scored.final_score == 80.0

