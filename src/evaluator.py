"""Cold-Read Evaluator module evaluating clips with relative timestamps and stripped metadata."""

from typing import List, Optional
from src.models import PromoCandidate, Episode, ColdReadScore, ScreenplayEvent
from src.llm_client import LLMClient


def format_clip_relative(candidate: PromoCandidate, episode: Episode) -> str:
    """
    Format clip events using strictly relative timestamps (0.0s = clip start).
    Strips event IDs, absolute times, and episode metadata so evaluator cannot infer position.
    """
    event_map = {e.event_id: e for e in episode.events}
    start_ev = event_map.get(candidate.start_event_id)
    end_ev = event_map.get(candidate.end_event_id)

    if not start_ev or not end_ev:
        return ""

    start_idx = episode.events.index(start_ev)
    end_idx = episode.events.index(end_ev)
    clip_events = episode.events[start_idx : end_idx + 1]

    clip_start_t = candidate.start_time
    lines = []
    lines.append(f"CLIP DURATION: {candidate.duration:.2f} seconds")
    lines.append("TRANSCRIPT:")

    for ev in clip_events:
        rel_start = max(0.0, round(ev.start_time - clip_start_t, 2))
        rel_end = max(rel_start, round(ev.end_time - clip_start_t, 2))
        speaker_label = ev.character if ev.event_type == "dialogue" else f"[{ev.event_type.upper()}]"
        lines.append(f"[{rel_start:04.2f}s - {rel_end:04.2f}s] {speaker_label}: {ev.text}")

    return "\n".join(lines)


COLD_READ_SYSTEM_PROMPT = """You are a discerning mobile video editor and social media audience evaluator.
You are evaluating a 20 to 60-second video clip for TikTok, Instagram Reels, or YouTube Shorts.
You have NEVER seen or heard of this show before. You know ZERO backstory.

Evaluate the clip on 5 editorial dimensions on a scale from 1.0 to 10.0:
1. hook_score (1.0-10.0): Does something happen in the FIRST 3.0 SECONDS (slap, confrontation, shocking accusation, dramatic visual) that forces a scrolling viewer to stop?
2. cliffhanger_score (1.0-10.0): Does the clip end at an unresolved, high-stakes moment that makes the viewer NEED to know what happens next?
3. standalone_score (1.0-10.0): Can a total stranger understand the emotional stakes and conflict without needing prior lore or confusing character genealogies?
4. emotion_score (1.0-10.0): How intense is the visceral emotion (humiliation, triumph, betrayal, shock)?
5. coherence_score (1.0-10.0): Does the clip hold together as a complete, dramatic scene without feeling chopped up or missing essential context?

Return a JSON object conforming to:
{
  "hook_score": 8.5,
  "cliffhanger_score": 9.0,
  "standalone_score": 7.5,
  "emotion_score": 8.0,
  "coherence_score": 8.0,
  "explanation": "Detailed 2-3 sentence editorial critique explaining why this clip works or falls short."
}
"""


class ColdReadEvaluator:
    """Evaluates promotional clips in complete isolation."""

    def __init__(self, llm_client: LLMClient):
        self.client = llm_client

    def evaluate(self, candidate: PromoCandidate, episode: Episode) -> ColdReadScore:
        """Evaluate a promo candidate in isolation using relative timestamps."""
        relative_text = format_clip_relative(candidate, episode)
        prompt = (
            f"Here is the continuous clip transcript with relative timing from 0.0s:\n\n"
            f"{relative_text}\n\n"
            f"Evaluate this clip strictly as a cold viewer on social media. "
            f"Provide scores from 1.0 to 10.0 for hook, cliffhanger, standalone clarity, emotion, and coherence, "
            f"along with a concise editorial explanation."
        )

        score: ColdReadScore = self.client.call_json(
            prompt=prompt,
            response_model=ColdReadScore,
            system_instruction=COLD_READ_SYSTEM_PROMPT,
            fixture_name="mock_evaluations",
            fixture_key=candidate.candidate_id
        )

        return score

