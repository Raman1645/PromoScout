"""Listwise comparative ranker for final promo selection."""

import random
from typing import List, Dict, Optional
from src.models import ScoredCandidate, FinalPromo, RankedItem, ListwiseRankingResponse
from src.llm_client import LLMClient
from src.deduplication import compute_iou


RANKER_SYSTEM_PROMPT = """You are the Senior Editorial Director for a leading microdrama streaming platform.
You are reviewing candidate promotional video clips (20-60s) for TikTok, Instagram Reels, and YouTube Shorts.

TASK:
You are provided with a shortlist of candidate promotional clips with their episode, duration, hook, cliffhanger, and cold-read scores.
Which would you post first, and why?
Rank ALL candidate clips in order of posting priority (from #1 most urgent/compelling promo to post first, down to the last).

CRITICAL REQUIREMENT:
You MUST return a full valid permutation of ALL candidate IDs provided in the prompt.
Do NOT omit or duplicate any candidate ID.
For every candidate ID, provide a crisp, one-sentence editorial rationale in 'post_first_because'.

Return JSON conforming to:
{
  "ranked_candidates": [
    {"candidate_id": "C_E01_01", "post_first_because": "Opens on an immediate, shocking public humiliation and cuts right before the counter-attack, making it impossible to scroll past."},
    ...
  ]
}
"""


class ListwiseRanker:
    """Ranks shortlisted candidates comparatively using a listwise prompt and enforces editorial diversity constraints."""

    def __init__(self, llm_client: LLMClient):
        self.client = llm_client

    def rank(
        self,
        shortlist: List[ScoredCandidate],
        target_top_n: int = 10,
        max_per_episode: int = 2,
        iou_threshold: float = 0.5
    ) -> List[FinalPromo]:
        """
        Shuffle candidates, request listwise ranking from LLM, validate permutation,
        and apply diversity and overlap constraints to yield final Top 10 promos.
        """
        if not shortlist:
            return []

        # If shortlist is smaller than target, adjust target
        target_count = min(target_top_n, len(shortlist))

        # Shuffle candidates to eliminate position bias
        shuffled = list(shortlist)
        random.seed(42)  # deterministic shuffle for reproducible evaluation
        random.shuffle(shuffled)

        candidate_map = {sc.candidate.candidate_id: sc for sc in shortlist}
        candidate_ids_set = set(candidate_map.keys())

        # Build candidate summaries for the prompt
        summary_lines = []
        for idx, sc in enumerate(shuffled, start=1):
            c = sc.candidate
            s = sc.scores
            scores_str = f"H:{s.hook_score:.1f} C:{s.cliffhanger_score:.1f} A:{s.standalone_score:.1f}" if s else ""
            summary_lines.append(
                f"Candidate {c.candidate_id} (Ep {c.episode_id} | {c.duration:.1f}s | Score: {sc.final_score:.1f} | {scores_str}):\n"
                f"  - Hook Quote: \"{c.hook_quote}\"\n"
                f"  - Cliffhanger Quote: \"{c.cliffhanger_quote}\"\n"
                f"  - Proposer Reason: {c.proposer_reason}\n"
            )

        prompt = (
            f"Here are {len(shuffled)} shortlisted candidate promo clips:\n\n"
            f"{''.join(summary_lines)}\n\n"
            f"Which would you post first, and why? "
            f"Rank all {len(shuffled)} candidates by posting priority. "
            f"Ensure ALL candidate IDs are included in the permutation."
        )

        response: ListwiseRankingResponse = self.client.call_json(
            prompt=prompt,
            response_model=ListwiseRankingResponse,
            system_instruction=RANKER_SYSTEM_PROMPT,
            fixture_name="mock_ranking",
            fixture_key="default"
        )

        ranked_items = response.ranked_candidates
        received_ids = [item.candidate_id for item in ranked_items]

        # Verify full permutation
        is_valid_permutation = (
            len(received_ids) == len(candidate_ids_set)
            and set(received_ids) == candidate_ids_set
        )

        ordered_candidates: List[ScoredCandidate] = []
        rationales: Dict[str, str] = {}

        if is_valid_permutation:
            for item in ranked_items:
                ordered_candidates.append(candidate_map[item.candidate_id])
                rationales[item.candidate_id] = item.post_first_because
        else:
            # Fallback to score order if permutation is incomplete
            ordered_candidates = sorted(shortlist, key=lambda sc: sc.final_score, reverse=True)
            for item in ranked_items:
                rationales[item.candidate_id] = item.post_first_because

        # Apply editorial selection constraints in Python:
        # 1. Max 2 promos per episode
        # 2. No overlapping promos with IoU > 0.5
        final_selected: List[FinalPromo] = []
        episode_counts: Dict[int, int] = {}

        for sc in ordered_candidates:
            if len(final_selected) >= target_count:
                break

            cand = sc.candidate
            ep_id = cand.episode_id

            # Constraint 1: Max per episode
            if episode_counts.get(ep_id, 0) >= max_per_episode:
                continue

            # Constraint 2: Temporal IoU with already selected promos in same episode
            too_close = False
            for prev in final_selected:
                if prev.episode_id == ep_id:
                    iou = compute_iou(cand.start_time, cand.end_time, prev.start_time, prev.end_time)
                    if iou > iou_threshold:
                        too_close = True
                        break

            if too_close:
                continue

            episode_counts[ep_id] = episode_counts.get(ep_id, 0) + 1
            reason = rationales.get(
                cand.candidate_id,
                f"Selected for exceptional hook ({sc.scores.hook_score if sc.scores else 0:.1f}) and cliffhanger intensity."
            )

            final_selected.append(FinalPromo(
                rank=len(final_selected) + 1,
                candidate_id=cand.candidate_id,
                episode_id=cand.episode_id,
                start_time=cand.start_time,
                end_time=cand.end_time,
                duration=cand.duration,
                start_event_id=cand.start_event_id,
                end_event_id=cand.end_event_id,
                hook_quote=cand.hook_quote,
                cliffhanger_quote=cand.cliffhanger_quote,
                spoiler_risk=cand.spoiler_risk,
                scores=sc.scores,
                final_score=sc.final_score,
                post_first_because=reason
            ))

        return final_selected

