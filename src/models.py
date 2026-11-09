"""Core Pydantic models for PromoScout."""

from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field


class ScreenplayEvent(BaseModel):
    """A single atomic timestamped event in an episode screenplay."""
    event_id: str = Field(description="Unique event ID, e.g. E01_015")
    episode_id: int = Field(description="Episode number as integer")
    scene_num: int = Field(description="Scene number")
    event_type: Literal["dialogue", "action", "camera", "sound"] = Field(description="Event classification")
    character: Optional[str] = Field(default=None, description="Character/speaker name if dialogue")
    parenthetical: Optional[str] = Field(default=None, description="Parenthetical emotion or direction")
    start_time: float = Field(description="Event start time in seconds (0.01s precision)")
    end_time: float = Field(description="Event end time in seconds (0.01s precision)")
    text: str = Field(description="Dialogue or direction text")

    @property
    def duration(self) -> float:
        return round(max(0.0, self.end_time - self.start_time), 2)


class Scene(BaseModel):
    """A scene boundary inside an episode."""
    scene_num: int
    heading: str
    start_time: float
    end_time: float


class Episode(BaseModel):
    """Full structured representation of an episode."""
    episode_id: int
    episode_number_raw: str
    logline: str
    continuity_notes: str
    scenes: List[Scene]
    events: List[ScreenplayEvent]


class CandidateProposal(BaseModel):
    """Proposer LLM output format for a single candidate promo."""
    start_event_id: str = Field(description="Event ID where promo begins (must be hook)")
    end_event_id: str = Field(description="Event ID where promo ends (must be cliffhanger)")
    hook_quote: str = Field(description="Exact quote from source event that hooks viewer")
    cliffhanger_quote: str = Field(description="Exact quote from source event ending on cliffhanger")
    spoiler_risk: Literal["low", "medium", "high"] = Field(description="Assessed spoiler risk")
    proposer_reason: str = Field(description="Editorial rationale for why this clip works")


class ProposerResponse(BaseModel):
    """Expected JSON response from candidate proposer."""
    candidates: List[CandidateProposal] = Field(default_factory=list)


class PromoCandidate(BaseModel):
    """Validated promo candidate with concrete boundaries and duration."""
    candidate_id: str
    episode_id: int
    start_event_id: str
    end_event_id: str
    start_time: float
    end_time: float
    duration: float
    hook_quote: str
    cliffhanger_quote: str
    spoiler_risk: Literal["low", "medium", "high"]
    proposer_reason: str
    repaired: bool = False
    repair_action: Optional[str] = None


class ColdReadScore(BaseModel):
    """Cold-read evaluation scores across 5 editorial dimensions."""
    hook_score: float = Field(ge=1.0, le=10.0, description="Hook strength in first 3s (1-10)")
    cliffhanger_score: float = Field(ge=1.0, le=10.0, description="Unresolved cliffhanger strength (1-10)")
    standalone_score: float = Field(ge=1.0, le=10.0, description="Standalone clarity to cold viewer (1-10)")
    emotion_score: float = Field(ge=1.0, le=10.0, description="Emotional impact and stakes (1-10)")
    promo_suitability: float = Field(default=8.0, ge=1.0, le=10.0, description="Would clip encourage viewer to watch show (1-10)")
    coherence_score: Optional[float] = Field(default=8.0, description="Narrative coherence and flow (1-10)")
    explanation: str = Field(description="Detailed editorial explanation of evaluation")


class ScoredCandidate(BaseModel):
    """Candidate with cold-read scores, quality gate status, and weighted rank."""
    candidate: PromoCandidate
    scores: Optional[ColdReadScore] = None
    weighted_score: float = 0.0
    spoiler_penalty: float = 0.0
    final_score: float = 0.0
    passed_gates: bool = False
    rejection_reason: Optional[str] = None
    per_episode_rank: Optional[int] = None


class RankedItem(BaseModel):
    """Item in final listwise ranking."""
    candidate_id: str
    post_first_because: str


class ListwiseRankingResponse(BaseModel):
    """Response from listwise ranker."""
    ranked_candidates: List[RankedItem]


class FinalPromo(BaseModel):
    """Final output representation of a ranked promo."""
    rank: int
    candidate_id: str
    episode_id: int
    start_time: float
    end_time: float
    duration: float
    start_event_id: str
    end_event_id: str
    hook_quote: str
    cliffhanger_quote: str
    spoiler_risk: str
    scores: ColdReadScore
    final_score: float
    post_first_because: str

