"""Modular, inspectable CLI orchestrator for PromoScout.
Supports Groq primary models (openai/gpt-oss-120b & openai/gpt-oss-20b),
resumability (--resume), stage inspection, and complete run metadata generation.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.parser import ScreenplayParser
from src.llm_client import LLMClient, ConfigurationError
from src.proposer import CandidateProposer
from src.validator import BoundaryValidator
from src.deduplication import deduplicate_candidates
from src.evaluator import ColdReadEvaluator
from src.scoring import score_candidate, assign_per_episode_rankings, select_top_candidates
from src.ranker import ListwiseRanker
from src.reporter import ReportGenerator
from src.checkpoint import CheckpointManager
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
    start_wall_time = time.time()
    input_dir = Path(args.input_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_stage = args.stage.lower() if args.stage else "all"
    checkpoint_mgr = CheckpointManager(checkpoint_dir=args.checkpoint_dir)

    # Resolve primary model and evaluator model
    primary_model = args.model or ("openai/gpt-oss-120b" if args.provider == "groq" else "gpt-4o-mini")
    evaluator_model = args.evaluator_model or ("openai/gpt-oss-20b" if args.provider == "groq" else primary_model)

    inference_mode_str = "MOCK FIXTURE REPLAY" if args.mock else f"LIVE INFERENCE ({args.provider.upper()})"

    print("=" * 72)
    print("PROMOSCOUT: OPERATIONAL PIPELINE & RISK MITIGATION ENGINE")
    print(f"Mode: {inference_mode_str} | Run ID: {args.run_id}")
    print(f"Provider: {args.provider.upper()} | Primary Model: {primary_model}")
    print(f"Evaluator Model: {evaluator_model} | Resume Enabled: {args.resume}")
    print("=" * 72)

    # Initialize components
    screenplay_parser = ScreenplayParser()
    try:
        llm_client = LLMClient(
            provider=args.provider,
            base_url=args.base_url,
            api_key=args.api_key,
            model=primary_model,
            temperature=args.temperature,
            run_id=args.run_id,
            rpm_limit=args.rpm,
            no_cache=args.no_cache,
            mock_mode=args.mock
        )
    except ConfigurationError as e:
        print(f"\n[CONFIGURATION ERROR] {e}")
        sys.exit(1)

    proposer = CandidateProposer(llm_client)
    validator = BoundaryValidator(min_duration=20.0, max_duration=60.0)
    evaluator = ColdReadEvaluator(llm_client, evaluator_model=evaluator_model)
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

    # Standardized output name: parsed_episodes.json
    parsed_summary = {
        ep_id: {
            "events_count": len(ep.events),
            "scenes_count": len(ep.scenes),
            "duration_seconds": ep.events[-1].end_time,
            "logline": ep.logline
        } for ep_id, ep in parsed_episodes.items()
    }
    with open(output_dir / "parsed_episodes.json", "w", encoding="utf-8") as f:
        json.dump(parsed_summary, f, indent=2)
    with open(raw_dir / "parsed_episodes.json", "w", encoding="utf-8") as f:
        json.dump(parsed_summary, f, indent=2)

    if target_stage == "parse":
        print(f"\n[Done] Stage 'parse' completed. Summary saved to {output_dir / 'parsed_episodes.json'}.")
        return

    # =========================================================================
    # STAGE 2 & 3: PROPOSER & BOUNDARY VALIDATION
    # =========================================================================
    print(f"\n[STAGE 2: CANDIDATE PROPOSER & STAGE 3: BOUNDARY VALIDATOR]")
    raw_proposals: Dict[int, Any] = {}
    all_validated: List[PromoCandidate] = []

    for ep_id, ep in parsed_episodes.items():
        # Checkpoint support
        cached_val = checkpoint_mgr.load("validated", str(ep_id)) if args.resume else None
        if cached_val:
            val_candidates = [PromoCandidate.model_validate(c) for c in cached_val]
            all_validated.extend(val_candidates)
            print(f"  • Episode {ep_id}: Resumed {len(val_candidates)} candidates from checkpoint.")
            continue

        proposals = proposer.propose(ep)
        raw_proposals[ep_id] = [p.model_dump() for p in proposals]

        validated = validator.validate_proposals(proposals, ep)
        all_validated.extend(validated)

        if args.resume:
            checkpoint_mgr.save("validated", str(ep_id), [c.model_dump() for c in validated])

        print(f"  • Episode {ep_id}: {len(proposals)} proposed -> {len(validated)} validated.")
        if args.inspect:
            for c in validated:
                print(f"    ↳ Validated Candidate {c.candidate_id}: {c.duration:.1f}s ({c.start_time:.1f}s - {c.end_time:.1f}s)")
                print(f"      Hook: \"{c.hook_quote[:60]}...\"")
                print(f"      Cliffhanger: \"{c.cliffhanger_quote[:60]}...\"")

    # Standardized output names: proposals.json, validated_candidates.json
    with open(output_dir / "proposals.json", "w", encoding="utf-8") as f:
        json.dump(raw_proposals, f, indent=2)
    with open(output_dir / "validated_candidates.json", "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_validated], f, indent=2)
    with open(raw_dir / "proposals.json", "w", encoding="utf-8") as f:
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

        # Resumability check
        cached_score = checkpoint_mgr.load("evaluation", cand.candidate_id) if args.resume else None
        if cached_score:
            score = evaluator.evaluate(cand, ep)  # will hit client cache or checkpoint
        else:
            score = evaluator.evaluate(cand, ep)
            if args.resume:
                checkpoint_mgr.save("evaluation", cand.candidate_id, score.model_dump())

        raw_evaluations[cand.candidate_id] = score.model_dump()
        scored = score_candidate(cand, score)
        scored_candidates.append(scored)

        status_str = "PASSED" if scored.passed_gates else f"FAILED: {scored.rejection_reason}"
        if args.verbose or len(deduped) <= 5:
            print(f"  • {cand.candidate_id} (Ep {cand.episode_id}): Final Score={scored.final_score:.1f} [{status_str}]")

    # Standardized output name: evaluations.json
    with open(output_dir / "evaluations.json", "w", encoding="utf-8") as f:
        json.dump(raw_evaluations, f, indent=2)
    with open(raw_dir / "evaluations.json", "w", encoding="utf-8") as f:
        json.dump(raw_evaluations, f, indent=2)

    scored_candidates = assign_per_episode_rankings(scored_candidates)
    shortlist = select_top_candidates(scored_candidates, top_n=20)

    print(f"  Summary: {len(shortlist)} candidates passed quality gates into Top 20 shortlist.")

    if target_stage in ("evaluate", "score"):
        print(f"\n[Done] Stages 'evaluate' & 'score' completed.")
        return

    # =========================================================================
    # STAGE 7: COMPARATIVE LISTWISE RANKING
    # =========================================================================
    print(f"\n[STAGE 7: COMPARATIVE LISTWISE RANKER]")
    final_promos = ranker.rank(shortlist, target_top_n=10, max_per_episode=2, iou_threshold=0.5)

    print(f"  Selected final Top {len(final_promos)} promotional clips across the show:")
    for p in final_promos:
        print(f"    #{p.rank}: Candidate {p.candidate_id} (Ep {p.episode_id} | {p.duration:.1f}s | Score: {p.final_score:.1f})")
        print(f"       Why Post First: {p.post_first_because}")

    # =========================================================================
    # STAGE 8: DELIVERABLES & RUN METADATA EXPORT
    # =========================================================================
    print(f"\n[STAGE 8: REPORTING & METADATA GENERATION]")
    json_path = reporter.export_json(final_promos, filename="final_rankings.json")
    csv_path = reporter.export_csv(final_promos, filename="final_rankings.csv")
    html_path = reporter.export_html(
        final_promos,
        episodes_analyzed=len(parsed_episodes),
        total_candidates_found=len(all_validated),
        filename="report.html"
    )

    # Emit comprehensive run_metadata.json (Section 6)
    usage = llm_client.get_usage_summary()
    total_elapsed = round(time.time() - start_wall_time, 2)

    run_metadata = {
        "run_id": args.run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": args.provider,
        "primary_model": primary_model,
        "evaluator_model": evaluator_model,
        "inference_mode": "mock" if args.mock else "live",
        "resumed": args.resume,
        "episodes_processed": list(parsed_episodes.keys()),
        "counts": {
            "episodes_count": len(parsed_episodes),
            "events_count": sum(len(e.events) for e in parsed_episodes.values()),
            "proposed_count": sum(len(p) for p in raw_proposals.values()),
            "validated_count": len(all_validated),
            "deduped_count": len(deduped),
            "shortlisted_count": len(shortlist),
            "final_top10_count": len(final_promos)
        },
        "validator_audit": {
            "trimmed_count": validator.trimmed_count,
            "rejected_count": validator.rejected_count,
            "rejection_reasons": validator.rejection_reasons
        },
        "token_usage": usage,
        "total_elapsed_seconds": total_elapsed,
        "output_artifacts": {
            "parsed_episodes": str(output_dir / "parsed_episodes.json"),
            "proposals": str(output_dir / "proposals.json"),
            "validated_candidates": str(output_dir / "validated_candidates.json"),
            "evaluations": str(output_dir / "evaluations.json"),
            "final_rankings_json": str(json_path),
            "final_rankings_csv": str(csv_path),
            "report_html": str(html_path)
        }
    }

    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)
    with open(raw_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"  ✓ JSON Deliverable:    {json_path}")
    print(f"  ✓ CSV Deliverable:     {csv_path}")
    print(f"  ✓ HTML Dashboard:      {html_path}")
    print(f"  ✓ Run Metadata Audit:  {metadata_path}")
    print("\n" + "=" * 72)
    print(f"PIPELINE COMPLETE. Time: {total_elapsed}s | Mode: {inference_mode_str}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="PromoScout: Operational Pipeline & Risk Mitigation Engine")
    parser.add_argument("--stage", default="all", choices=["all", "parse", "propose", "validate", "deduplicate", "evaluate", "score", "rank"], help="Execute a specific stage or 'all'")
    parser.add_argument("--input-dir", default="I Married a Broken Soldier. He was a Duke", help="Screenplay directory")
    parser.add_argument("--episodes", default="all", help="Episodes to process: '1', '1,2,3', '1-5', or 'all'")
    parser.add_argument("--provider", default="groq", choices=["groq", "openai", "gemini", "custom"], help="Primary LLM provider (default: groq)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="LLM API Key")
    parser.add_argument("--model", default=None, help="Primary model name (default on Groq: openai/gpt-oss-120b)")
    parser.add_argument("--evaluator-model", default=None, help="Evaluator model name (default on Groq: openai/gpt-oss-20b)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
    parser.add_argument("--run-id", default="default", help="Unique run identifier for cache isolation")
    parser.add_argument("--rpm", type=int, default=15, help="Requests per minute rate limit")
    parser.add_argument("--no-cache", action="store_true", help="Bypass disk cache")
    parser.add_argument("--mock", action="store_true", help="Explicitly enable mock fixture replay mode")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted execution from stage checkpoints")
    parser.add_argument("--checkpoint-dir", default="cache/checkpoints", help="Directory for stage checkpoints")
    parser.add_argument("--verbose", action="store_true", help="Print detailed step-by-step logs")
    parser.add_argument("--inspect", action="store_true", help="Inspect payload data at each stage")
    parser.add_argument("--output-dir", default="outputs", help="Directory for final reports")
    parser.add_argument("--raw-dir", default="raw_outputs", help="Directory for unedited raw pipeline outputs")
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
