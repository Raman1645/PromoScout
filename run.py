"""Modular, inspectable CLI orchestrator for PromoScout.
Supports step-by-step pipeline execution, stage inspection, and verbose logging.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.parser import ScreenplayParser
from src.llm_client import LLMClient
from src.proposer import CandidateProposer, serialize_episode_compact
from src.validator import BoundaryValidator
from src.deduplication import deduplicate_candidates
from src.evaluator import ColdReadEvaluator, format_clip_relative
from src.scoring import score_candidate, assign_per_episode_rankings, select_top_candidates
from src.ranker import ListwiseRanker
from src.reporter import ReportGenerator
from src.models import Episode, PromoCandidate, ScoredCandidate, FinalPromo


def parse_episode_filter(ep_str: str) -> List[int]:
    """Parse episode argument e.g. '1', '1,2,3', '1-5', or 'all'."""
    if not ep_str or ep_str.lower() == "all":
        return list(range(1, 51))
    episodes = set()
    for part in ep_str.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-")
            episodes.update(range(int(start_s), int(end_s) + 1))
        elif part.isdigit():
            episodes.add(int(part))
    return sorted(list(episodes))


def run_pipeline(args):
    input_dir = Path(args.input_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_stage = args.stage.lower() if args.stage else "all"

    print("=" * 70)
    print("PROMOSCOUT: STRUCTURED EDITORIAL PIPELINE")
    print(f"Target Stage: {target_stage.upper()} | Episodes: {args.episodes}")
    print(f"Model: {args.model} | Run ID: {args.run_id} | Mock Replay: {args.mock}")
    print("=" * 70)

    # Initialize modules
    screenplay_parser = ScreenplayParser()
    llm_client = LLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        run_id=args.run_id,
        rpm_limit=args.rpm,
        no_cache=args.no_cache,
        mock_mode=args.mock
    )
    proposer = CandidateProposer(llm_client)
    validator = BoundaryValidator(min_duration=20.0, max_duration=60.0)
    evaluator = ColdReadEvaluator(llm_client)
    ranker = ListwiseRanker(llm_client)
    reporter = ReportGenerator(output_dir=str(output_dir))

    target_episodes = parse_episode_filter(args.episodes)

    # =========================================================================
    # STAGE 1: SCREENPLAY PARSING
    # =========================================================================
    print(f"\n[STAGE 1: SCREENPLAY PARSER]")
    parsed_episodes: Dict[int, Episode] = {}
    for ep_id in target_episodes:
        matched = list(input_dir.glob(f"episode-{ep_id:03d}*.txt"))
        if not matched:
            continue
        ep = screenplay_parser.parse_file(matched[0])
        parsed_episodes[ep.episode_id] = ep
        if args.verbose or len(target_episodes) <= 3:
            print(f"  ✓ Episode {ep.episode_id}: {len(ep.events)} events, {len(ep.scenes)} scenes, Duration: {ep.events[-1].end_time:.1f}s")
            if args.inspect:
                print(f"    Sample Event: {ep.events[0].event_id} ({ep.events[0].start_time}s-{ep.events[0].end_time}s) [{ep.events[0].event_type}] {ep.events[0].text[:60]}...")

    with open(raw_dir / "parsed_screenplays.json", "w", encoding="utf-8") as f:
        json.dump({
            ep_id: {
                "events_count": len(ep.events),
                "scenes_count": len(ep.scenes),
                "duration_seconds": ep.events[-1].end_time,
                "logline": ep.logline
            } for ep_id, ep in parsed_episodes.items()
        }, f, indent=2)

    if target_stage == "parse":
        print(f"\n[Done] Stage 'parse' completed. Summary saved to {raw_dir / 'parsed_screenplays.json'}.")
        return

    # =========================================================================
    # STAGE 2 & 3: PROPOSER & BOUNDARY VALIDATION
    # =========================================================================
    print(f"\n[STAGE 2: CANDIDATE PROPOSER & STAGE 3: BOUNDARY VALIDATOR]")
    raw_proposals: Dict[int, Any] = {}
    all_validated: List[PromoCandidate] = []

    for ep_id, ep in parsed_episodes.items():
        proposals = proposer.propose(ep)
        raw_proposals[ep_id] = [p.model_dump() for p in proposals]

        validated = validator.validate_proposals(proposals, ep)
        all_validated.extend(validated)

        print(f"  • Episode {ep_id}: {len(proposals)} proposed -> {len(validated)} validated.")
        if args.inspect:
            for c in validated:
                print(f"    ↳ Validated Candidate {c.candidate_id}: {c.duration:.1f}s ({c.start_time:.1f}s - {c.end_time:.1f}s)")
                print(f"      Hook: \"{c.hook_quote[:60]}...\"")
                print(f"      Cliffhanger: \"{c.cliffhanger_quote[:60]}...\"")

    with open(raw_dir / "proposed_candidates.json", "w", encoding="utf-8") as f:
        json.dump(raw_proposals, f, indent=2)

    with open(raw_dir / "validated_candidates.json", "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_validated], f, indent=2)

    print(f"  Summary: {len(all_validated)} candidates passed validation (Trimmed: {validator.trimmed_count}, Rejected: {validator.rejected_count})")

    if target_stage in ("propose", "validate"):
        print(f"\n[Done] Stages 'propose' & 'validate' completed.")
        return

    # =========================================================================
    # STAGE 4: CANDIDATE DEDUPLICATION (IoU > 0.5)
    # =========================================================================
    print(f"\n[STAGE 4: CANDIDATE DEDUPLICATION]")
    deduped = deduplicate_candidates(all_validated, iou_threshold=0.5)
    print(f"  • Total before: {len(all_validated)} -> After IoU pruning: {len(deduped)}")

    if target_stage == "deduplicate":
        print(f"\n[Done] Stage 'deduplicate' completed.")
        return

    # =========================================================================
    # STAGE 5 & 6: COLD-READ EVALUATION & QUALITY GATES
    # =========================================================================
    print(f"\n[STAGE 5: COLD-READ EVALUATION & STAGE 6: QUALITY GATES]")
    scored_candidates: List[ScoredCandidate] = []
    raw_evaluations: Dict[str, Any] = {}

    for cand in deduped:
        ep = parsed_episodes[cand.episode_id]
        score = evaluator.evaluate(cand, ep)
        raw_evaluations[cand.candidate_id] = score.model_dump()

        scored = score_candidate(cand, score)
        scored_candidates.append(scored)

        status_str = "PASSED" if scored.passed_gates else f"FAILED: {scored.rejection_reason}"
        if args.verbose or len(deduped) <= 5:
            print(f"  • {cand.candidate_id} (Ep {cand.episode_id}): Final Score={scored.final_score:.1f} [{status_str}]")
            if args.inspect:
                print(f"    Scores: Hook={score.hook_score:.1f}, Cliff={score.cliffhanger_score:.1f}, Standalone={score.standalone_score:.1f}")
                print(f"    Critique: {score.explanation[:80]}...")

    with open(raw_dir / "cold_read_evaluations.json", "w", encoding="utf-8") as f:
        json.dump(raw_evaluations, f, indent=2)

    scored_candidates = assign_per_episode_rankings(scored_candidates)
    shortlist = select_top_candidates(scored_candidates, top_n=20)

    with open(raw_dir / "scored_shortlist.json", "w", encoding="utf-8") as f:
        json.dump([sc.model_dump() for sc in shortlist], f, indent=2)

    print(f"  Summary: {len(shortlist)} candidates passed quality gates into Top 20 shortlist.")

    if target_stage in ("evaluate", "score"):
        print(f"\n[Done] Stages 'evaluate' & 'score' completed.")
        return

    # =========================================================================
    # STAGE 7: COMPARATIVE LISTWISE RANKING
    # =========================================================================
    print(f"\n[STAGE 7: COMPARATIVE LISTWISE RANKER]")
    final_promos = ranker.rank(shortlist, target_top_n=10, max_per_episode=2, iou_threshold=0.5)

    with open(raw_dir / "listwise_ranking_raw.json", "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in final_promos], f, indent=2)

    print(f"  Selected final Top {len(final_promos)} promotional clips across the show:")
    for p in final_promos:
        print(f"    #{p.rank}: Candidate {p.candidate_id} (Ep {p.episode_id} | {p.duration:.1f}s | Score: {p.final_score:.1f})")
        print(f"       Why Post First: {p.post_first_because}")

    # =========================================================================
    # STAGE 8: DELIVERABLES EXPORT
    # =========================================================================
    print(f"\n[STAGE 8: REPORTING & ARTIFACT GENERATION]")
    json_path = reporter.export_json(final_promos)
    csv_path = reporter.export_csv(final_promos)
    html_path = reporter.export_html(
        final_promos,
        episodes_analyzed=len(parsed_episodes),
        total_candidates_found=len(all_validated)
    )

    print(f"  ✓ JSON Deliverable: {json_path}")
    print(f"  ✓ CSV Deliverable:  {csv_path}")
    print(f"  ✓ HTML Dashboard:   {html_path}")
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PromoScout: Structured Editorial Pipeline")
    parser.add_argument("--stage", default="all", choices=["all", "parse", "propose", "validate", "deduplicate", "evaluate", "score", "rank"], help="Execute a specific pipeline stage or 'all'")
    parser.add_argument("--input-dir", default="I Married a Broken Soldier. He was a Duke", help="Screenplay directory")
    parser.add_argument("--episodes", default="all", help="Episodes to process: '1', '1,2,3', '1-5', or 'all'")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible LLM base URL")
    parser.add_argument("--api-key", default=None, help="LLM API Key")
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
    parser.add_argument("--run-id", default="default", help="Unique run identifier for cache isolation")
    parser.add_argument("--rpm", type=int, default=15, help="Requests per minute rate limit")
    parser.add_argument("--no-cache", action="store_true", help="Bypass disk cache")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode using fixture replay")
    parser.add_argument("--verbose", action="store_true", help="Print detailed step-by-step logs")
    parser.add_argument("--inspect", action="store_true", help="Inspect payload data at each stage")
    parser.add_argument("--output-dir", default="outputs", help="Directory for final reports")
    parser.add_argument("--raw-dir", default="raw_outputs", help="Directory for unedited raw pipeline outputs")
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
