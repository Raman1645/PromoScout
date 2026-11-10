"""Cold-Read Evaluator module with strict speaker anonymization and isolated context."""

from typing import List, Tuple, Dict, Optional
from src.models import PromoCandidate, Episode, ColdReadScore, ScreenplayEvent
from src.llm_client import LLMClient


def format_clip_relative(candidate: PromoCandidate, episode: Episode) -> str:
    """Format clip events using relative timestamps with original character names for UI inspection."""
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
    lines.append("SCENE DIALOGUE & VISIBLE ACTION (Relative timing, original character names):")

    for ev in clip_events:
        rel_start = max(0.0, round(ev.start_time - clip_start_t, 2))
        rel_end = max(rel_start, round(ev.end_time - clip_start_t, 2))
        tag = ev.character if ev.event_type == "dialogue" else f"[{ev.event_type.upper()}]"
        lines.append(f"[{rel_start:05.2f}s - {rel_end:05.2f}s] {tag}: {ev.text}")

    return "\n".join(lines)


def format_clip_anonymized(candidate: PromoCandidate, episode: Episode) -> Tuple[str, Dict[str, str]]:
    """
    Format clip events using strictly relative timestamps (0.0s = clip start),
    strip event IDs and metadata, and neutralize character identities to [SPEAKER A], [SPEAKER B], etc.
    Filters out production-only camera instructions to focus purely on visible action and dialogue.
    """
    event_map = {e.event_id: e for e in episode.events}
    start_ev = event_map.get(candidate.start_event_id)
    end_ev = event_map.get(candidate.end_event_id)

    if not start_ev or not end_ev:
        return "", {}

    start_idx = episode.events.index(start_ev)
    end_idx = episode.events.index(end_ev)
    clip_events = episode.events[start_idx : end_idx + 1]

    clip_start_t = candidate.start_time
    speaker_map: Dict[str, str] = {}
    speaker_letter_idx = 0

    lines = []
    lines.append(f"CLIP DURATION: {candidate.duration:.2f} seconds")
    lines.append("SCENE DIALOGUE & VISIBLE ACTION (Transcribed with relative timing):")

    for ev in clip_events:
        rel_start = max(0.0, round(ev.start_time - clip_start_t, 2))
        rel_end = max(rel_start, round(ev.end_time - clip_start_t, 2))

        if ev.event_type == "dialogue":
            char_key = ev.character or "UNKNOWN"
            if char_key not in speaker_map:
                speaker_label = f"[SPEAKER {chr(65 + speaker_letter_idx)}]"
                speaker_map[char_key] = speaker_label
                speaker_letter_idx += 1
            else:
                speaker_label = speaker_map[char_key]

            # In dialogue text, keep words as spoken
            lines.append(f"[{rel_start:05.2f}s - {rel_end:05.2f}s] {speaker_label}: \"{ev.text}\"")

        elif ev.event_type == "action":
            # Action line describes what a viewer sees
            lines.append(f"[{rel_start:05.2f}s - {rel_end:05.2f}s] [ACTION]: {ev.text}")

        # Note: Production-only CAMERA shots are omitted from evaluator text to prevent treating camera framing as story facts

    return "\n".join(lines), speaker_map


COLD_READ_SYSTEM_PROMPT = """You are a professional mobile video editor and social media audience evaluator.
You are evaluating a 20 to 60-second video promo clip for TikTok, Instagram Reels, and YouTube Shorts.
You have NEVER watched this show before. Character identities are anonymized ([SPEAKER A], [SPEAKER B]) so you have ZERO prior knowledge.

Evaluate the clip on 5 editorial dimensions on a scale from 1.0 to 10.0:
1. hook_score (1.0-10.0): Does something happen in the FIRST 3.0 SECONDS (slap, confrontation, accusation, shock) that forces a scrolling viewer to stop?
2. standalone_score (1.0-10.0): Can a new viewer understand the situation and stakes without needing prior lore or backstory?
3. cliffhanger_score (1.0-10.0): Does the ending create intense curiosity or suspense, cutting at an unresolved peak?
4. emotion_score (1.0-10.0): Does the segment generate a visceral emotional reaction (betrayal, shock, indignation)?
5. promo_suitability (1.0-10.0): Would this clip compel a viewer to immediately click and watch the full show?

Return a JSON object conforming to:
{
  "hook_score": 8.5,
  "standalone_score": 7.5,
  "cliffhanger_score": 9.0,
  "emotion_score": 8.0,
  "promo_suitability": 8.5,
  "explanation": "Detailed 2-3 sentence editorial critique explaining why this clip works or falls short."
}
"""


class ColdReadEvaluator:
    """Evaluates promotional clips in complete isolation with neutralized speaker identities."""

    def __init__(self, llm_client: LLMClient, evaluator_model: Optional[str] = None):
        self.client = llm_client
        # Default evaluator model: openai/gpt-oss-20b for Groq, or inherited
        if evaluator_model:
            self.evaluator_model = evaluator_model
        elif self.client.provider == "groq":
            self.evaluator_model = "openai/gpt-oss-20b"
        else:
            self.evaluator_model = self.client.model

    def evaluate(self, candidate: PromoCandidate, episode: Episode) -> ColdReadScore:
        """Evaluate a promo candidate in isolation using relative timestamps and anonymized speakers."""
        relative_text, _ = format_clip_anonymized(candidate, episode)
        prompt = (
            f"Here is the continuous clip transcript with relative timing from 0.0s and neutralized speakers:\n\n"
            f"{relative_text}\n\n"
            f"Evaluate this clip strictly as a cold viewer on social media. "
            f"Provide scores from 1.0 to 10.0 for hook, standalone clarity, cliffhanger, emotion, and promo suitability, "
            f"along with a concise editorial explanation."
        )

        score: ColdReadScore = self.client.call_json(
            prompt=prompt,
            response_model=ColdReadScore,
            system_instruction=COLD_READ_SYSTEM_PROMPT,
            model_override=self.evaluator_model,
            fixture_name="mock_evaluations",
            fixture_key=candidate.candidate_id
        )

        return score
