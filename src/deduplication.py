"""Candidate deduplication module using temporal Intersection-over-Union (IoU)."""

from typing import List, Dict
from src.models import PromoCandidate


def compute_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Calculate temporal Intersection over Union between two intervals."""
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0.0:
        return 0.0
    return round(intersection / union, 4)


def deduplicate_candidates(
    candidates: List[PromoCandidate],
    iou_threshold: float = 0.5
) -> List[PromoCandidate]:
    """
    Remove heavily overlapping candidates within the same episode (IoU > threshold).
    Retains candidate with lower spoiler risk or clearer editorial rationale.
    """
    if not candidates:
        return []

    # Group by episode
    by_episode: Dict[int, List[PromoCandidate]] = {}
    for c in candidates:
        by_episode.setdefault(c.episode_id, []).append(c)

    deduped: List[PromoCandidate] = []

    for ep_id, ep_candidates in by_episode.items():
        # Sort candidates by preference: lower spoiler risk, longer duration
        risk_priority = {"low": 3, "medium": 2, "high": 1}
        sorted_candidates = sorted(
            ep_candidates,
            key=lambda c: (risk_priority.get(c.spoiler_risk, 0), len(c.proposer_reason)),
            reverse=True
        )

        survivors: List[PromoCandidate] = []
        for cand in sorted_candidates:
            is_duplicate = False
            for s in survivors:
                iou = compute_iou(cand.start_time, cand.end_time, s.start_time, s.end_time)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                survivors.append(cand)

        deduped.extend(survivors)

    return deduped

