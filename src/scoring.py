"""Quality gates, weighted scoring formulas, and per-episode ranking."""

from typing import List, Dict, Tuple
from src.models import PromoCandidate, ColdReadScore, ScoredCandidate


# Weights matching mitigations.md Section 3.I
WEIGHT_HOOK = 0.25
WEIGHT_CLIFFHANGER = 0.25
WEIGHT_STANDALONE = 0.20
WEIGHT_EMOTION = 0.15
WEIGHT_PROMO_SUITABILITY = 0.15

# Spoiler penalties
SPOILER_PENALTIES = {
    "low": 0.0,
    "medium": 5.0,
    "high": 10.0
}


def score_candidate(
    candidate: PromoCandidate,
    score: ColdReadScore,
    hook_threshold: float = 7.0,
    cliffhanger_threshold: float = 7.0
) -> ScoredCandidate:
    """Apply quality gates, compute weighted scores, and apply spoiler penalties."""
    # 1. Quality Gates (Section 3.H: Hook >= 7, Cliffhanger >= 7, 20-60s)
    passed_gates = True
    rejection_reason = None

    if score.hook_score < hook_threshold:
        passed_gates = False
        rejection_reason = f"Hook score {score.hook_score:.1f} is below threshold {hook_threshold:.1f}"
    elif score.cliffhanger_score < cliffhanger_threshold:
        passed_gates = False
        rejection_reason = f"Cliffhanger score {score.cliffhanger_score:.1f} is below threshold {cliffhanger_threshold:.1f}"
    elif candidate.duration < 20.0 or candidate.duration > 60.0:
        passed_gates = False
        rejection_reason = f"Duration {candidate.duration:.2f}s outside strict 20-60s window"

    # 2. Weighted Score (0-100 scale, Section 3.I)
    promo_val = score.promo_suitability if score.promo_suitability is not None else (score.coherence_score or 8.0)
    base_score = (
        WEIGHT_HOOK * score.hook_score
        + WEIGHT_CLIFFHANGER * score.cliffhanger_score
        + WEIGHT_STANDALONE * score.standalone_score
        + WEIGHT_EMOTION * score.emotion_score
        + WEIGHT_PROMO_SUITABILITY * promo_val
    ) * 10.0  # normalize from 10 to 100

    base_score = round(base_score, 2)

    # 3. Spoiler Penalty (Section 3.H)
    penalty = SPOILER_PENALTIES.get(candidate.spoiler_risk.lower(), 0.0)
    final_score = round(max(0.0, base_score - penalty), 2)

    return ScoredCandidate(
        candidate=candidate,
        scores=score,
        weighted_score=base_score,
        spoiler_penalty=penalty,
        final_score=final_score,
        passed_gates=passed_gates,
        rejection_reason=rejection_reason
    )


def assign_per_episode_rankings(scored_candidates: List[ScoredCandidate]) -> List[ScoredCandidate]:
    """
    Rank candidates within each episode by final_score descending.
    Establishes per_episode_rank (1, 2, 3...) needed for HitRate@3 gold evaluation.
    """
    by_episode: Dict[int, List[ScoredCandidate]] = {}
    for sc in scored_candidates:
        by_episode.setdefault(sc.candidate.episode_id, []).append(sc)

    ranked_all: List[ScoredCandidate] = []
    for ep_id, ep_list in by_episode.items():
        ep_list_sorted = sorted(ep_list, key=lambda x: x.final_score, reverse=True)
        for rank_idx, item in enumerate(ep_list_sorted, start=1):
            item.per_episode_rank = rank_idx
            ranked_all.append(item)

    return ranked_all


def select_top_candidates(
    scored_candidates: List[ScoredCandidate],
    top_n: int = 20
) -> List[ScoredCandidate]:
    """Select top candidates across all episodes that passed quality gates."""
    valid_candidates = [sc for sc in scored_candidates if sc.passed_gates]
    sorted_candidates = sorted(valid_candidates, key=lambda sc: sc.final_score, reverse=True)
    return sorted_candidates[:top_n]
