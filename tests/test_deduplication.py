"""Unit tests for IoU and deduplication logic."""

import pytest
from src.models import PromoCandidate
from src.deduplication import compute_iou, deduplicate_candidates


def test_compute_iou():
    # Exact overlap
    assert compute_iou(0.0, 30.0, 0.0, 30.0) == 1.0

    # No overlap
    assert compute_iou(0.0, 20.0, 25.0, 45.0) == 0.0

    # 50% overlap: [0, 30] and [10, 40] -> intersection 20, union 40 -> 0.5
    assert compute_iou(0.0, 30.0, 10.0, 40.0) == 0.5

    # 66.6% overlap: [0, 30] and [10, 30] -> intersection 20, union 30 -> 0.6667
    assert compute_iou(0.0, 30.0, 10.0, 30.0) == 0.6667


def test_deduplicate_candidates():
    c1 = PromoCandidate(
        candidate_id="C_E01_01", episode_id=1, start_event_id="E01_001", end_event_id="E01_010",
        start_time=0.0, end_time=30.0, duration=30.0, hook_quote="Hook 1", cliffhanger_quote="Cliff 1",
        spoiler_risk="low", proposer_reason="Strong reason here."
    )
    # Heavy overlap with c1 (IoU > 0.5), but higher spoiler risk
    c2 = PromoCandidate(
        candidate_id="C_E01_02", episode_id=1, start_event_id="E01_002", end_event_id="E01_010",
        start_time=2.0, end_time=30.0, duration=28.0, hook_quote="Hook 2", cliffhanger_quote="Cliff 1",
        spoiler_risk="high", proposer_reason="Another reason."
    )
    # Different moment in same episode (no overlap)
    c3 = PromoCandidate(
        candidate_id="C_E01_03", episode_id=1, start_event_id="E01_015", end_event_id="E01_025",
        start_time=50.0, end_time=80.0, duration=30.0, hook_quote="Hook 3", cliffhanger_quote="Cliff 3",
        spoiler_risk="low", proposer_reason="Third reason."
    )

    deduped = deduplicate_candidates([c1, c2, c3], iou_threshold=0.5)
    assert len(deduped) == 2
    # c1 should be kept over c2 because c1 has low spoiler risk vs high
    retained_ids = [c.candidate_id for c in deduped]
    assert "C_E01_01" in retained_ids
    assert "C_E01_03" in retained_ids
    assert "C_E01_02" not in retained_ids

