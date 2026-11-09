"""Evaluation and benchmarking suite for PromoScout.
Computes HitRate@3 on 5 systematic gold episodes and true independent-run Overlap@10 stability.
Never reports dummy lists or synthetic mocks as live results.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

from src.deduplication import compute_iou
from src.parser import ScreenplayParser
from src.llm_client import LLMClient
from src.proposer import CandidateProposer
from src.validator import BoundaryValidator
from src.deduplication import deduplicate_candidates
from src.evaluator import ColdReadEvaluator
from src.scoring import score_candidate, assign_per_episode_rankings, select_top_candidates
from src.ranker import ListwiseRanker
from src.models import FinalPromo, ScoredCandidate, Episode


def execute_pipeline_pass(
    episodes: List[int],
    input_dir: Path,
    llm_client: LLMClient
) -> List[FinalPromo]:
    """Execute a true end-to-end pipeline pass on target episodes."""
    parser = ScreenplayParser()
    proposer = CandidateProposer(llm_client)
    validator = BoundaryValidator(min_duration=20.0, max_duration=60.0)
    evaluator = ColdReadEvaluator(llm_client)
    ranker = ListwiseRanker(llm_client)

    all_validated = []
    parsed_episodes = {}

    for ep_id in episodes:
        matched = list(input_dir.glob(f"episode-{ep_id:03d}*.txt"))
        if matched:
            ep = parser.parse_file(matched[0])
            parsed_episodes[ep.episode_id] = ep
            proposals = proposer.propose(ep)
            val = validator.validate_proposals(proposals, ep)
            all_validated.extend(val)

    deduped = deduplicate_candidates(all_validated, iou_threshold=0.5)

    scored_candidates = []
    for cand in deduped:
        ep = parsed_episodes[cand.episode_id]
        score = evaluator.evaluate(cand, ep)
        scored = score_candidate(cand, score)
        scored_candidates.append(scored)

    scored_candidates = assign_per_episode_rankings(scored_candidates)
    shortlist = select_top_candidates(scored_candidates, top_n=20)
    final_promos = ranker.rank(shortlist, target_top_n=10, max_per_episode=2, iou_threshold=0.5)

    return final_promos


def evaluate_hitrate_at_3(
    gold_file: Path,
    input_dir: Path,
    llm_client: LLMClient,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate HitRate@3: For 5 systematically chosen gold episodes, check if any
    of the system's per-episode top 3 candidates matches an acceptable gold window with IoU > 0.5.
    """
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)["episodes"]

    parser = ScreenplayParser()
    proposer = CandidateProposer(llm_client)
    validator = BoundaryValidator(min_duration=20.0, max_duration=60.0)
    evaluator = ColdReadEvaluator(llm_client)

    results = {}
    matched_episodes = 0

    print("\n--- Evaluating HitRate@3 against Gold Set ---")
    for ep_id_str, gold_info in gold_data.items():
        ep_id = int(ep_id_str)
        matched_files = list(input_dir.glob(f"episode-{ep_id:03d}*.txt"))
        if not matched_files:
            continue

        ep = parser.parse_file(matched_files[0])
        proposals = proposer.propose(ep)
        validated = validator.validate_proposals(proposals, ep)
        deduped = deduplicate_candidates(validated, iou_threshold=0.5)

        scored_candidates = []
        for cand in deduped:
            score = evaluator.evaluate(cand, ep)
            scored = score_candidate(cand, score)
            scored_candidates.append(scored)

        ranked = assign_per_episode_rankings(scored_candidates)
        top_3 = [sc for sc in ranked if sc.per_episode_rank and sc.per_episode_rank <= 3]

        # Check IoU with gold acceptable windows
        ep_matched = False
        best_iou = 0.0
        best_candidate_id = None
        for sc in top_3:
            c = sc.candidate
            for window in gold_info["acceptable_windows"]:
                iou = compute_iou(c.start_time, c.end_time, window["start_time"], window["end_time"])
                if iou > best_iou:
                    best_iou = iou
                    best_candidate_id = c.candidate_id
                if iou > iou_threshold:
                    ep_matched = True

        if ep_matched:
            matched_episodes += 1

        results[ep_id_str] = {
            "title": gold_info["title"],
            "matched": ep_matched,
            "best_iou": round(best_iou, 4),
            "best_candidate_id": best_candidate_id,
            "top_3_count": len(top_3)
        }
        status = "MATCH (Hit)" if ep_matched else "MISS"
        print(f"  Episode {ep_id} ({gold_info['title']}): {status} | Max IoU: {best_iou:.3f}")

    hit_rate = round(matched_episodes / len(gold_data), 2)
    print(f"\nFinal HitRate@3: {matched_episodes}/{len(gold_data)} ({hit_rate * 100:.1f}%)")

    return {
        "hit_rate_at_3": hit_rate,
        "matched_episodes": matched_episodes,
        "total_gold_episodes": len(gold_data),
        "details": results
    }


def evaluate_stability(
    run_1_promos: List[FinalPromo],
    run_2_promos: List[FinalPromo],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate stability across two independent runs using Overlap@10:
    Calculates proportion of promos in run 2 that match a promo in run 1 (same episode & IoU > 0.5).
    """
    matches = 0
    match_pairs = []

    for p2 in run_2_promos:
        matched = False
        for p1 in run_1_promos:
            if p2.episode_id == p1.episode_id:
                iou = compute_iou(p2.start_time, p2.end_time, p1.start_time, p1.end_time)
                if iou > iou_threshold:
                    matches += 1
                    match_pairs.append({
                        "p1_rank": p1.rank,
                        "p2_rank": p2.rank,
                        "episode_id": p1.episode_id,
                        "iou": iou
                    })
                    matched = True
                    break

    overlap_score = round(matches / max(1, min(len(run_1_promos), len(run_2_promos), 10)), 2)
    return {
        "overlap_at_10": overlap_score,
        "matches_count": matches,
        "pairs": match_pairs
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PromoScout on Gold Set & Stability Benchmark")
    parser.add_argument("--input-dir", default="I Married a Broken Soldier. He was a Duke")
    parser.add_argument("--gold-file", default="gold_set/gold_annotations.json")
    parser.add_argument("--output-file", default="gold_set/benchmark_results.json")
    parser.add_argument("--provider", default="groq", help="LLM Provider (groq, openai, gemini)")
    parser.add_argument("--api-key", default=None, help="API key for live evaluation")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--mock", action="store_true", help="Explicitly enable fixture replay mode")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    gold_file = Path(args.gold_file)
    output_file = Path(args.output_file)

    inference_label = "MOCK FIXTURE REPLAY (Deterministic Test)" if args.mock else f"LIVE MODEL INFERENCE ({args.provider}:{args.model})"

    print("=" * 65)
    print("PROMOSCOUT BENCHMARK & QUALITY EVALUATION")
    print(f"Inference Mode: {inference_label}")
    print(f"Gold Annotations: {gold_file}")
    print("=" * 65)

    # 1. HitRate@3 Evaluation
    client_run1 = LLMClient(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        run_id="run_1",
        temperature=0.3,
        mock_mode=args.mock
    )
    hitrate_results = evaluate_hitrate_at_3(gold_file, input_dir, client_run1, iou_threshold=0.5)

    # 2. Genuine Stability Test across Dual Runs (No dummy lists!)
    print(f"\n--- Running Independent Dual-Run Stability Test ({inference_label}) ---")
    test_episodes = [1, 2, 3, 13, 25, 38, 50]

    print("  Executing Pass 1 (Run ID: run_1)...")
    run_1_promos = execute_pipeline_pass(test_episodes, input_dir, client_run1)

    print("  Executing Pass 2 (Run ID: run_2)...")
    client_run2 = LLMClient(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        run_id="run_2",
        temperature=0.3,
        mock_mode=args.mock
    )
    run_2_promos = execute_pipeline_pass(test_episodes, input_dir, client_run2)

    stability_results = evaluate_stability(run_1_promos, run_2_promos, iou_threshold=0.5)
    print(f"Overlap@10: {stability_results['overlap_at_10'] * 100:.1f}% ({stability_results['matches_count']}/{min(10, len(run_1_promos))} matched)")

    full_benchmark = {
        "inference_mode": "mock" if args.mock else "live",
        "provider": args.provider if not args.mock else "fixture_replay",
        "model": args.model if not args.mock else "mock_fixtures",
        "is_synthetic_fixture": args.mock,
        "hitrate_at_3": hitrate_results,
        "stability": stability_results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_benchmark, f, indent=2)

    print(f"\n✓ Benchmark results saved to: {output_file}")


if __name__ == "__main__":
    main()
