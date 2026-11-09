"""Candidate Proposer module for discovering promo candidates from microdrama episodes."""

from typing import List, Optional
from src.models import Episode, CandidateProposal, ProposerResponse
from src.llm_client import LLMClient


def serialize_episode_compact(episode: Episode) -> str:
    """Serialize episode events into a token-efficient compact format."""
    lines = []
    lines.append(f"EPISODE {episode.episode_id}")
    if episode.logline:
        lines.append(f"LOGLINE: {episode.logline}")
    lines.append("EVENTS:")

    for ev in episode.events:
        tag_str = ev.character if ev.event_type == "dialogue" else f"[{ev.event_type.upper()}]"
        dur = round(ev.end_time - ev.start_time, 2)
        lines.append(f"[{ev.event_id}] ({ev.start_time:05.2f} - {ev.end_time:05.2f} | {dur:04.1f}s) {tag_str}: {ev.text}")

    return "\n".join(lines)


PROPOSER_SYSTEM_PROMPT = """You are an expert promotional video editor for mobile microdramas (TikTok, Instagram Reels, YouTube Shorts).
Your task is to identify 1 to 3 continuous 20 to 60-second promotional clips from the provided timestamped screenplay.

CRITICAL REQUIREMENTS:
1. Continuous Clip: Must be a continuous stretch between start_event_id and end_event_id.
2. Target Duration: 20 to 60 seconds (duration = end_time - start_time).
3. The Hook: Must open on an explosive hook in the FIRST 3 SECONDS (slap, public humiliation, shocking line, intense confrontation). The start event MUST be the hook.
4. The Cliffhanger: Must end on an unresolved, high-stakes dramatic cliffhanger (a revelation, ultimatum, or question left unanswered).
5. Exact Quotes: hook_quote and cliffhanger_quote MUST be verbatim quotes from the chosen start and end events.
6. Spoiler Risk:
   - 'low': Intriguing hook that sets up initial premise without spoiling major climax.
   - 'medium': Reveals a twist from mid-episode.
   - 'high': Reveals the ultimate resolution or payoff of the episode/arc.

Return a JSON object conforming to:
{
  "candidates": [
    {
      "start_event_id": "E01_015",
      "end_event_id": "E01_028",
      "hook_quote": "...",
      "cliffhanger_quote": "...",
      "spoiler_risk": "low" | "medium" | "high",
      "proposer_reason": "One line explaining why this clip grabs attention and compels viewers to watch."
    }
  ]
}
"""


class CandidateProposer:
    """Discovers promo clip candidates from an episode using LLM with feedback retry for short clips."""

    def __init__(self, llm_client: LLMClient):
        self.client = llm_client

    def propose(self, episode: Episode) -> List[CandidateProposal]:
        """Propose candidates for an episode, retrying short (<20s) candidates with feedback."""
        compact_text = serialize_episode_compact(episode)
        prompt = (
            f"Here is the timestamped screenplay for Episode {episode.episode_id}:\n\n"
            f"{compact_text}\n\n"
            f"Identify 1 to 3 high-converting promotional clip candidates (20-60 seconds each). "
            f"Ensure start_event_id hooks viewers immediately and end_event_id leaves them hanging."
        )

        response: ProposerResponse = self.client.call_json(
            prompt=prompt,
            response_model=ProposerResponse,
            system_instruction=PROPOSER_SYSTEM_PROMPT,
            fixture_name="mock_proposals",
            fixture_key=f"ep_{episode.episode_id}"
        )

        candidates = response.candidates
        if not candidates:
            return []

        # Event map for quick lookup
        event_map = {e.event_id: e for e in episode.events}

        # Check for clips < 20s and perform a single targeted retry if needed
        valid_candidates: List[CandidateProposal] = []
        for cand in candidates:
            start_ev = event_map.get(cand.start_event_id)
            end_ev = event_map.get(cand.end_event_id)

            if not start_ev or not end_ev:
                continue

            duration = round(end_ev.end_time - start_ev.start_time, 2)
            if duration < 20.0:
                # Feedback retry
                retry_prompt = (
                    f"{compact_text}\n\n"
                    f"ATTENTION: Your proposed candidate {cand.start_event_id} to {cand.end_event_id} has duration {duration:.1f}s, "
                    f"which is UNDER the strict 20.0s minimum requirement.\n"
                    f"Please re-propose this candidate, extending the boundary so total duration is between 20.0s and 60.0s. "
                    f"Keep the hook at the start and ensure the clip still ends on a powerful cliffhanger."
                )
                try:
                    retry_resp: ProposerResponse = self.client.call_json(
                        prompt=retry_prompt,
                        response_model=ProposerResponse,
                        system_instruction=PROPOSER_SYSTEM_PROMPT,
                        fixture_name="mock_proposals",
                        fixture_key=f"ep_{episode.episode_id}_retry"
                    )
                    if retry_resp.candidates:
                        valid_candidates.append(retry_resp.candidates[0])
                    else:
                        valid_candidates.append(cand)
                except Exception:
                    valid_candidates.append(cand)
            else:
                valid_candidates.append(cand)

        return valid_candidates

