"""Boundary validator and post-repair quality verification for PromoScout."""

import re
from typing import List, Tuple, Optional, Dict
from src.models import Episode, CandidateProposal, PromoCandidate, ScreenplayEvent


def _normalize(text: str) -> str:
    """Normalize text for substring matching (lowercase, alphanumeric only)."""
    return re.sub(r'[^a-z0-9]', '', text.lower())


class BoundaryValidator:
    """
    Validates candidate boundaries, executes algorithmic trimming for clips slightly over 60s,
    and enforces strict post-repair hook/cliffhanger quote and timing constraints.
    """

    def __init__(self, min_duration: float = 20.0, max_duration: float = 60.0):
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Audit stats
        self.total_evaluated = 0
        self.passed_count = 0
        self.trimmed_count = 0
        self.rejected_count = 0
        self.rejection_reasons: Dict[str, int] = {}

    def _record_rejection(self, reason: str) -> None:
        self.rejected_count += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def validate_candidate(
        self,
        proposal: CandidateProposal,
        episode: Episode,
        candidate_idx: int
    ) -> Tuple[Optional[PromoCandidate], Optional[str]]:
        """
        Validate a single candidate proposal against the episode events.
        Returns (PromoCandidate, None) on success, or (None, rejection_reason) on failure.
        """
        self.total_evaluated += 1
        event_map = {e.event_id: e for e in episode.events}

        start_ev = event_map.get(proposal.start_event_id)
        end_ev = event_map.get(proposal.end_event_id)

        # 1. Existence check
        if not start_ev or not end_ev:
            reason = f"Event ID not found: {proposal.start_event_id} or {proposal.end_event_id}"
            self._record_rejection(reason)
            return None, reason

        # 2. Ordering check
        start_idx = episode.events.index(start_ev)
        end_idx = episode.events.index(end_ev)
        if start_idx >= end_idx:
            reason = f"Start event {proposal.start_event_id} does not precede end event {proposal.end_event_id}"
            self._record_rejection(reason)
            return None, reason

        clip_events = episode.events[start_idx : end_idx + 1]
        start_time = clip_events[0].start_time
        end_time = clip_events[-1].end_time
        raw_duration = round(end_time - start_time, 2)

        repaired = False
        repair_action = None

        # 3. Minimum duration check (< 20s rejected)
        if raw_duration < self.min_duration:
            reason = f"Duration {raw_duration:.2f}s is under minimum {self.min_duration}s"
            self._record_rejection(reason)
            return None, reason

        # 4. Maximum duration check & Algorithmic Trimming (> 60s)
        if raw_duration > self.max_duration:
            # Attempt trimming non-dialogue trailing events if within 66s
            if raw_duration <= 66.0:
                trimmed_events = list(clip_events)
                # Trim non-dialogue events from the end while over max_duration
                while (
                    len(trimmed_events) > 2
                    and (trimmed_events[-1].end_time - trimmed_events[0].start_time) > self.max_duration
                    and trimmed_events[-1].event_type != "dialogue"
                ):
                    trimmed_events.pop()

                new_duration = round(trimmed_events[-1].end_time - trimmed_events[0].start_time, 2)
                if self.min_duration <= new_duration <= self.max_duration:
                    clip_events = trimmed_events
                    start_time = clip_events[0].start_time
                    end_time = clip_events[-1].end_time
                    repaired = True
                    repair_action = f"Trimmed trailing non-dialogue from {raw_duration:.2f}s to {new_duration:.2f}s"
                    self.trimmed_count += 1
                else:
                    reason = f"Duration {raw_duration:.2f}s exceeds {self.max_duration}s and could not be trimmed safely"
                    self._record_rejection(reason)
                    return None, reason
            else:
                reason = f"Duration {raw_duration:.2f}s exceeds limit {self.max_duration}s (beyond repair threshold)"
                self._record_rejection(reason)
                return None, reason

        final_duration = round(end_time - start_time, 2)

        # 5. POST-REPAIR CHECKS (CRUCIAL)
        # A. Quote verification
        norm_hook = _normalize(proposal.hook_quote)
        norm_cliff = _normalize(proposal.cliffhanger_quote)

        # Find matching events
        hook_event: Optional[ScreenplayEvent] = None
        cliff_event: Optional[ScreenplayEvent] = None

        for ev in clip_events:
            ev_norm = _normalize(ev.text)
            if not hook_event and norm_hook and norm_hook in ev_norm:
                hook_event = ev
            if norm_cliff and norm_cliff in ev_norm:
                cliff_event = ev

        # Fallback for short quotes or punctuation mismatch
        if not hook_event:
            # Check if start event text contains major words of hook
            words = [w for w in proposal.hook_quote.lower().split() if len(w) > 3]
            if words and any(w in clip_events[0].text.lower() for w in words):
                hook_event = clip_events[0]

        if not cliff_event:
            words = [w for w in proposal.cliffhanger_quote.lower().split() if len(w) > 3]
            if words and any(w in clip_events[-1].text.lower() for w in words):
                cliff_event = clip_events[-1]

        if not hook_event:
            reason = f"Hook quote '{proposal.hook_quote[:30]}...' not found in clip source events"
            self._record_rejection(reason)
            return None, reason

        if not cliff_event:
            reason = f"Cliffhanger quote '{proposal.cliffhanger_quote[:30]}...' not found in clip source events"
            self._record_rejection(reason)
            return None, reason

        # B. Hook Timing Check: Hook must be in first 3.0s of clip
        hook_offset = round(hook_event.start_time - start_time, 2)
        if hook_offset > 3.0:
            reason = f"Hook occurs at {hook_offset:.2f}s from clip start (exceeds 3.0s limit)"
            self._record_rejection(reason)
            return None, reason

        # C. Cliffhanger Timing Check: Cliffhanger must end in final 2.0-5.0s of clip
        cliff_offset_from_end = round(end_time - cliff_event.end_time, 2)
        if cliff_offset_from_end > 5.0:
            reason = f"Cliffhanger ends {cliff_offset_from_end:.2f}s before clip end (exceeds 5.0s limit, cliffhanger is diluted)"
            self._record_rejection(reason)
            return None, reason

        # Candidate passed all checks
        self.passed_count += 1
        cand_id = f"C_E{episode.episode_id:02d}_{candidate_idx:02d}"
        candidate = PromoCandidate(
            candidate_id=cand_id,
            episode_id=episode.episode_id,
            start_event_id=clip_events[0].event_id,
            end_event_id=clip_events[-1].event_id,
            start_time=start_time,
            end_time=end_time,
            duration=final_duration,
            hook_quote=proposal.hook_quote,
            cliffhanger_quote=proposal.cliffhanger_quote,
            spoiler_risk=proposal.spoiler_risk,
            proposer_reason=proposal.proposer_reason,
            repaired=repaired,
            repair_action=repair_action
        )
        return candidate, None

    def validate_proposals(
        self,
        proposals: List[CandidateProposal],
        episode: Episode
    ) -> List[PromoCandidate]:
        """Validate all proposals for an episode."""
        valid_candidates = []
        for idx, prop in enumerate(proposals, start=1):
            cand, err = self.validate_candidate(prop, episode, idx)
            if cand:
                valid_candidates.append(cand)
        return valid_candidates

