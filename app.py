"""PromoScout Editorial Suite: Clean, High-Contrast 4-Tab Discovery UI.

100% Native Streamlit components. Zero emojis. Live run data only.
"""

import os
import sys
import json
import time
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd

# Add repo to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.parser import ScreenplayParser
from src.llm_client import LLMClient, load_dotenv_if_present
from src.proposer import CandidateProposer
from src.validator import BoundaryValidator
from src.deduplication import deduplicate_candidates, compute_iou
from src.evaluator import ColdReadEvaluator, format_clip_relative, format_clip_anonymized
from src.scoring import score_candidate, assign_per_episode_rankings, select_top_candidates
from src.ranker import ListwiseRanker
from src.reporter import ReportGenerator
from src.models import Episode, PromoCandidate, ScoredCandidate, FinalPromo
from run import parse_episode_filter

# Ensure environment variables are loaded from .env
load_dotenv_if_present()

st.set_page_config(
    page_title="PromoScout — Editorial Discovery Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# FILE SYSTEM PATHS & DATA HELPERS
# =============================================================================
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
SCREENPLAY_DIR = BASE_DIR / "I Married a Broken Soldier. He was a Duke"
FEEDBACK_FILE = OUTPUTS_DIR / "feedback.csv"


@st.cache_resource
def get_parser() -> ScreenplayParser:
    return ScreenplayParser()


def load_json_file(path: Path) -> Optional[Any]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_feedback_map() -> Dict[str, str]:
    """Read feedback decisions from feedback.csv."""
    if FEEDBACK_FILE.exists():
        try:
            df = pd.read_csv(FEEDBACK_FILE)
            if "candidate_id" in df.columns and "decision" in df.columns:
                return dict(zip(df["candidate_id"].astype(str), df["decision"].astype(str)))
        except Exception:
            pass
    return {}


def record_feedback(promo: FinalPromo, decision: str) -> None:
    """Record Human-in-the-Loop decision to feedback.csv."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_id": promo.candidate_id,
        "episode_id": promo.episode_id,
        "rank": promo.rank,
        "duration": promo.duration,
        "final_score": promo.final_score,
        "decision": decision
    }
    df_new = pd.DataFrame([row])
    if not FEEDBACK_FILE.exists():
        df_new.to_csv(FEEDBACK_FILE, index=False)
    else:
        df_existing = pd.read_csv(FEEDBACK_FILE)
        if promo.candidate_id in df_existing["candidate_id"].astype(str).values:
            df_existing.loc[df_existing["candidate_id"].astype(str) == promo.candidate_id, "decision"] = decision
            df_existing.loc[df_existing["candidate_id"].astype(str) == promo.candidate_id, "timestamp"] = row["timestamp"]
            df_existing.to_csv(FEEDBACK_FILE, index=False)
        else:
            df_new.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)


def get_active_promos() -> List[FinalPromo]:
    preloaded = load_json_file(OUTPUTS_DIR / "final_rankings.json")
    if preloaded and isinstance(preloaded, list):
        try:
            return [FinalPromo.model_validate(p) for p in preloaded]
        except Exception:
            return []
    return []


# Initialize session state
if "promos" not in st.session_state or not st.session_state.promos:
    st.session_state.promos = get_active_promos()

if "episodes" not in st.session_state:
    st.session_state.episodes = {}

env_key = os.getenv("GROQ_API_KEY", "")
if not env_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    try:
        env_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

if "session_api_key" not in st.session_state:
    st.session_state.session_api_key = env_key


# =============================================================================
# CLEAN HEADER (NO EMOJIS)
# =============================================================================
st.title("PromoScout: Editorial Discovery Suite")
st.caption("Automated Promotional Discovery & Listwise Ranking for Mobile Microdramas • Series: I Married a Broken Soldier. He was a Duke")

# =============================================================================
# 4 FOCUSED TABS (NO EMOJIS)
# =============================================================================
tabs = st.tabs([
    "Top 10 Promos",
    "Run Pipeline",
    "Pipeline Metrics",
    "All Candidates"
])


# =============================================================================
# TAB 1: TOP 10 PROMOS (THE MAIN DELIVERABLE)
# =============================================================================
with tabs[0]:
    active_promos: List[FinalPromo] = st.session_state.promos
    feedback_map = get_feedback_map()

    if not active_promos:
        st.warning("No promotional rankings found. Run the pipeline in the 'Run Pipeline' tab or click below to reload.")
        if st.button("Reload Outputs", type="secondary"):
            st.session_state.promos = get_active_promos()
            st.rerun()
    else:
        # High-level summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Promos Ranked", f"Top {len(active_promos)}")
        avg_d = sum(p.duration for p in active_promos) / len(active_promos)
        c2.metric("Average Duration", f"{avg_d:.1f}s", "20–60s Compliance")
        top_s = active_promos[0].final_score
        c3.metric("Top Score", f"{top_s:.1f} / 100")
        reviewed_count = sum(1 for p in active_promos if p.candidate_id in feedback_map)
        c4.metric("Editorial Reviews", f"{reviewed_count} / {len(active_promos)} Done")

        st.divider()

        # Render each promo card inside a native bordered container
        for p in active_promos:
            current_status = feedback_map.get(p.candidate_id, "PENDING")

            with st.container(border=True):
                # Card Header
                h_col1, h_col2, h_col3 = st.columns([3, 1, 1])
                with h_col1:
                    st.subheader(f"Rank #{p.rank} — Episode {p.episode_id}")
                    st.write(f"Duration: **{p.duration:.1f}s** | Timestamps: **{p.start_time:05.2f}s – {p.end_time:05.2f}s**")
                with h_col2:
                    st.metric("Composite Score", f"{p.final_score:.1f} / 100")
                with h_col3:
                    if current_status == "ACCEPTED":
                        st.success("ACCEPTED")
                    elif current_status == "REJECTED":
                        st.error("REJECTED")
                    else:
                        st.info("PENDING REVIEW")

                # The 2 Primary Ad Hooks
                hook_col, cliff_col = st.columns(2)
                with hook_col:
                    st.error(f"**3-Second Hook (Event {p.start_event_id}):**\n\n> *\"{p.hook_quote}\"*")
                with cliff_col:
                    st.warning(f"**Unresolved Cliffhanger (Event {p.end_event_id}):**\n\n> *\"{p.cliffhanger_quote}\"*")

                # Editorial Rationale
                st.info(f"**Editorial Verdict (Why Post First):**\n\n{p.post_first_because}")

                # 6 Quality Dimension Metrics
                sc = p.scores
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Hook Power", f"{sc.hook_score:.1f}/10")
                m2.metric("Cliffhanger", f"{sc.cliffhanger_score:.1f}/10")
                m3.metric("Standalone", f"{sc.standalone_score:.1f}/10")
                m4.metric("Emotion", f"{sc.emotion_score:.1f}/10")
                m5.metric("Suitability", f"{sc.promo_suitability:.1f}/10")
                m6.metric("Spoiler Risk", p.spoiler_risk.upper())

                # Transcript Expander
                with st.expander(f"View Full Clip Script (Relative 00.00s Timing) — Candidate {p.candidate_id}"):
                    ep_data = st.session_state.episodes.get(p.episode_id)
                    if not ep_data:
                        matched_files = list(SCREENPLAY_DIR.glob(f"episode-{p.episode_id:03d}*.txt"))
                        if matched_files:
                            ep_data = get_parser().parse_file(matched_files[0])
                            st.session_state.episodes[p.episode_id] = ep_data

                    if ep_data:
                        ev_map = {e.event_id: e for e in ep_data.events}
                        start_ev = ev_map.get(p.start_event_id)
                        end_ev = ev_map.get(p.end_event_id)
                        if start_ev and end_ev:
                            s_i = ep_data.events.index(start_ev)
                            e_i = ep_data.events.index(end_ev)
                            clip_events = ep_data.events[s_i : e_i + 1]

                            lines = []
                            for ev in clip_events:
                                rel_s = max(0.0, round(ev.start_time - p.start_time, 2))
                                rel_e = max(rel_s, round(ev.end_time - p.start_time, 2))
                                speaker = ev.character if ev.event_type == "dialogue" else f"[{ev.event_type.upper()}]"

                                marker = ""
                                if ev.event_id == p.start_event_id:
                                    marker = "  <-- [3-SECOND HOOK STARTS HERE]"
                                elif ev.event_id == p.end_event_id:
                                    marker = "  <-- [UNRESOLVED CLIFFHANGER CUT]"

                                lines.append(f"[{rel_s:05.2f}s - {rel_e:05.2f}s] {speaker}: {ev.text}{marker}")

                            st.code("\n".join(lines), language="text")
                        else:
                            st.caption("Event boundaries not found in parsed episode.")
                    else:
                        st.info(f"Opening Hook (0.00s): \"{p.hook_quote}\"\n\nClosing Cliffhanger ({p.duration:.1f}s): \"{p.cliffhanger_quote}\"\n\n*(Full screenplay files are excluded from public git per confidentiality rules. Upload a screenplay in 'Run Pipeline' or run locally to view full scene dialogue).*")

                # Action Buttons
                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
                with btn_col1:
                    if st.button("Accept Promo", key=f"btn_acc_{p.candidate_id}", use_container_width=True):
                        record_feedback(p, "ACCEPTED")
                        st.rerun()
                with btn_col2:
                    if st.button("Reject Promo", key=f"btn_rej_{p.candidate_id}", use_container_width=True):
                        record_feedback(p, "REJECTED")
                        st.rerun()
                with btn_col3:
                    if current_status != "PENDING":
                        st.caption(f"Decision saved to outputs/feedback.csv (Marked as {current_status}).")


# =============================================================================
# TAB 2: RUN PIPELINE (BUILT FOR INTERVIEWS & LIVE TESTING)
# =============================================================================
with tabs[1]:
    st.subheader("Run PromoScout Pipeline")
    st.caption("Execute on selected episodes or upload a new screenplay pack. The UI calls the exact core pipeline as run.py.")

    # 1. Preset Episode Selection
    st.markdown("#### 1. Choose Episode Scope")
    preset_choice = st.radio(
        "Episode Scope Presets",
        [
            "Episode 1 (Pilot: Altar Betrayal - Fast Demo)",
            "Gold Set (5 Episodes: 1, 13, 25, 38, 50)",
            "Full Series (All 50 Episodes)",
            "Custom Range"
        ],
        horizontal=True
    )

    if "Episode 1" in preset_choice:
        ep_filter_str = "1"
    elif "Gold Set" in preset_choice:
        ep_filter_str = "1,13,25,38,50"
    elif "Full Series" in preset_choice:
        ep_filter_str = "all"
    else:
        ep_filter_str = st.text_input("Enter Episode Numbers (e.g. '1', '1,2,3', '1-5')", value="1-3")

    st.divider()

    # 2. Execution Mode & Credentials
    st.markdown("#### 2. Model & Credentials")
    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        exec_mode = st.radio(
            "Execution Mode",
            ["Live LLM API (OpenAI)", "Live LLM API (Groq GPT-OSS)", "Mock Fixture Replay (Instant Offline Demo)"],
            help="Live mode connects to OpenAI or Groq API. Mock mode replays cached grounded fixtures with zero latency."
        )
        is_mock_run = "Mock" in exec_mode

        if "OpenAI" in exec_mode:
            provider_name = "openai"
            primary_model = "gpt-4o-mini"
            eval_model = "gpt-4o-mini"
            env_key_var = "OPENAI_API_KEY"
            if not is_mock_run:
                st.info("Provider: OpenAI\n\nPrimary Model: gpt-4o-mini (Proposer & Ranker)\n\nEvaluator Model: gpt-4o-mini (Cold-Read)")
        elif "Groq" in exec_mode:
            provider_name = "groq"
            primary_model = "openai/gpt-oss-120b"
            eval_model = "openai/gpt-oss-20b"
            env_key_var = "GROQ_API_KEY"
            if not is_mock_run:
                st.info("Provider: Groq\n\nPrimary Model: openai/gpt-oss-120b (Proposer & Ranker)\n\nEvaluator Model: openai/gpt-oss-20b (Cold-Read)")
        else:
            provider_name = "groq"
            primary_model = "openai/gpt-oss-120b"
            eval_model = "openai/gpt-oss-20b"
            env_key_var = "GROQ_API_KEY"

    with cfg_col2:
        env_val = os.getenv(env_key_var, "")
        key_found = bool(st.session_state.session_api_key or env_val)
        if key_found:
            st.success(f"API Key Status: Detected for {provider_name.upper()} (Ready for live inference)")
        else:
            st.warning(f"API Key Status: No {env_key_var} detected. Provide key below or run in Mock mode.")

        custom_key = st.text_input(
            f"{provider_name.upper()} API Key (Session-only, masked, never printed or saved)",
            type="password",
            value=st.session_state.session_api_key if st.session_state.session_api_key else "",
            help="Your key stays purely in memory and is never logged."
        )
        if custom_key:
            st.session_state.session_api_key = custom_key

    st.divider()

    # 3. Optional Upload
    with st.expander("Optional: Upload Custom Screenplay (.txt or .zip)"):
        uploaded_screenplay = st.file_uploader(
            "Upload Screenplay File or Zip",
            type=["txt", "zip"],
            help="Leave empty to use the default 50-episode repository screenplay pack."
        )

    # 4. Run Button & Progress Bar
    run_btn = st.button("Run PromoScout Pipeline Now", type="primary", use_container_width=True)

    if run_btn:
        start_time_t = time.time()
        target_ids = parse_episode_filter(ep_filter_str)
        if not target_ids:
            st.error("Invalid episode selection.")
        else:
            progress_bar = st.progress(0, text="Initializing pipeline...")
            step_status = st.empty()

            try:
                # Stage 1: Screenplay Parser
                step_status.info("Stage 1/7: Screenplay Parser — Extracting scenes, dialogues, actions & 0.01s timestamps...")
                progress_bar.progress(15, text="Parsing screenplays...")
                parser = get_parser()

                parsed_dict: Dict[int, Episode] = {}
                if uploaded_screenplay is not None:
                    if uploaded_screenplay.name.endswith(".zip"):
                        with tempfile.TemporaryDirectory() as tmp_d:
                            z_path = Path(tmp_d) / uploaded_screenplay.name
                            with open(z_path, "wb") as f:
                                f.write(uploaded_screenplay.getbuffer())
                            with zipfile.ZipFile(z_path, "r") as z:
                                z.extractall(tmp_d)
                            for eid in target_ids:
                                matched = list(Path(tmp_d).rglob(f"*episode-{eid:03d}*.txt"))
                                if matched:
                                    parsed_dict[eid] = parser.parse_file(matched[0])
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as f:
                            f.write(uploaded_screenplay.getbuffer())
                            tmp_p = Path(f.name)
                        parsed_ep = parser.parse_file(tmp_p)
                        parsed_dict[parsed_ep.episode_id] = parsed_ep
                else:
                    for eid in target_ids:
                        matched = list(SCREENPLAY_DIR.glob(f"episode-{eid:03d}*.txt"))
                        if matched:
                            parsed_dict[eid] = parser.parse_file(matched[0])

                if not parsed_dict:
                    st.error(f"No screenplay files found for episodes: {ep_filter_str}")
                    st.stop()

                # Client initialization
                active_key = st.session_state.session_api_key or os.getenv(env_key_var, "")
                rpm_val = 60 if provider_name == "openai" else 15
                llm_c = LLMClient(
                    provider=provider_name,
                    api_key=active_key,
                    model=primary_model,
                    temperature=0.3,
                    rpm_limit=rpm_val,
                    mock_mode=is_mock_run
                )
                eval_c = LLMClient(
                    provider=provider_name,
                    api_key=active_key,
                    model=eval_model,
                    temperature=0.3,
                    rpm_limit=rpm_val,
                    mock_mode=is_mock_run
                )

                # Stage 2 & 3: Propose & Validate
                step_status.info("Stage 2 & 3: Proposer & Boundary Validator — Discovering candidates & enforcing 20–60s bounds...")
                progress_bar.progress(35, text="Proposing & validating candidates...")
                proposer = CandidateProposer(llm_c)
                validator = BoundaryValidator(min_duration=20.0, max_duration=60.0)

                validated_list = []
                proposals_dict = {}
                for eid, ep_obj in parsed_dict.items():
                    props = proposer.propose(ep_obj)
                    proposals_dict[eid] = [p.model_dump() for p in props]
                    vals = validator.validate_proposals(props, ep_obj)
                    validated_list.extend(vals)

                # Stage 4: Deduplicate
                step_status.info("Stage 4: Deduplication — Pruning intra-episode temporal overlap (IoU > 0.5)...")
                progress_bar.progress(55, text="Deduplicating candidates...")
                deduped_list = deduplicate_candidates(validated_list, iou_threshold=0.5)

                # Stage 5 & 6: Cold-Read Evaluator & Quality Gates
                step_status.info("Stage 5 & 6: Cold-Read Evaluator — Zero-context scoring ([SPEAKER A/B], relative times) & Quality Gates...")
                progress_bar.progress(75, text="Evaluating & scoring...")
                evaluator = ColdReadEvaluator(eval_c)
                scored_list = []
                raw_evaluations = {}
                for cand in deduped_list:
                    ep_obj = parsed_dict[cand.episode_id]
                    ev_res = evaluator.evaluate(cand, ep_obj)
                    raw_evaluations[cand.candidate_id] = ev_res.model_dump()
                    sc_obj = score_candidate(cand, ev_res)
                    scored_list.append(sc_obj)

                scored_list = assign_per_episode_rankings(scored_list)
                shortlist_cands = select_top_candidates(scored_list, top_n=20)

                # Stage 7: Listwise Ranker
                step_status.info("Stage 7: Listwise Ranker — Shuffled comparative ranking (Which to post first?)...")
                progress_bar.progress(90, text="Final comparative ranking...")
                ranker = ListwiseRanker(llm_c)
                final_promos_res = ranker.rank(shortlist_cands, target_top_n=10, max_per_episode=2, iou_threshold=0.5)

                # Stage 8: Reports & Exports
                reporter = ReportGenerator(output_dir=str(OUTPUTS_DIR))
                reporter.export_json(final_promos_res, filename="final_rankings.json")
                reporter.export_csv(final_promos_res, filename="final_rankings.csv")
                reporter.export_html(final_promos_res, episodes_analyzed=len(parsed_dict), total_candidates_found=len(validated_list), filename="report.html")

                # Save intermediate artifact files to disk so metrics stay synchronized
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                with open(OUTPUTS_DIR / "proposals.json", "w", encoding="utf-8") as f:
                    json.dump(proposals_dict, f, indent=2)
                with open(OUTPUTS_DIR / "validated_candidates.json", "w", encoding="utf-8") as f:
                    json.dump([c.model_dump() for c in validated_list], f, indent=2)
                with open(OUTPUTS_DIR / "evaluations.json", "w", encoding="utf-8") as f:
                    json.dump(raw_evaluations, f, indent=2)

                # Collect token usage
                usage_summary = llm_c.get_usage_summary()
                if eval_c != llm_c:
                    ev_u = eval_c.get_usage_summary()
                    usage_summary["total_calls"] += ev_u.get("total_calls", 0)
                    usage_summary["cache_hits"] += ev_u.get("cache_hits", 0)
                    usage_summary["prompt_tokens"] += ev_u.get("prompt_tokens", 0)
                    usage_summary["completion_tokens"] += ev_u.get("completion_tokens", 0)
                    usage_summary["total_tokens"] += ev_u.get("total_tokens", 0)

                total_dur_sec = round(time.time() - start_time_t, 2)
                run_metadata = {
                    "run_id": f"session_{int(time.time())}",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "provider": provider_name,
                    "primary_model": primary_model,
                    "evaluator_model": eval_model,
                    "inference_mode": "mock" if is_mock_run else "live",
                    "counts": {
                        "episodes_count": len(parsed_dict),
                        "events_count": sum(len(e.events) for e in parsed_dict.values()),
                        "proposed_count": sum(len(p) for p in proposals_dict.values()),
                        "validated_count": len(validated_list),
                        "deduped_count": len(deduped_list),
                        "shortlisted_count": len(shortlist_cands),
                        "final_top10_count": len(final_promos_res)
                    },
                    "validator_audit": {
                        "trimmed_count": validator.trimmed_count,
                        "rejected_count": validator.rejected_count,
                        "rejection_reasons": validator.rejection_reasons
                    },
                    "token_usage": usage_summary,
                    "total_elapsed_seconds": total_dur_sec
                }

                with open(OUTPUTS_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(run_metadata, f, indent=2)

                # Update session state with live objects
                st.session_state.promos = final_promos_res
                st.session_state.episodes = parsed_dict
                st.session_state.run_metadata = run_metadata

                progress_bar.progress(100, text="Complete!")
                step_status.success(f"Success! Discovered and ranked Top {len(final_promos_res)} promotional clips across {len(parsed_dict)} episodes. Switch to 'Top 10 Promos' to inspect!")
                time.sleep(1.0)
                st.rerun()

            except Exception as e:
                progress_bar.progress(0, text="Execution Paused")
                st.error(f"Error executing pipeline: {str(e)}")


# =============================================================================
# TAB 3: PIPELINE METRICS (LIVE EXECUTION DATA ONLY)
# =============================================================================
with tabs[2]:
    st.subheader("Live Pipeline Execution Metrics")
    st.caption("Active execution statistics from the current run.")

    if st.button("Reload Latest Run Metrics", type="secondary"):
        st.session_state.run_metadata = load_json_file(OUTPUTS_DIR / "run_metadata.json")
        st.session_state.promos = get_active_promos()
        st.rerun()

    meta_info = st.session_state.get("run_metadata") or load_json_file(OUTPUTS_DIR / "run_metadata.json") or {}
    cnts = meta_info.get("counts", {})
    t_usage = meta_info.get("token_usage", {})
    v_audit = meta_info.get("validator_audit", {})

    # 1. Live Conversion Funnel
    st.markdown("#### 1. Conversion Funnel (Active Run)")
    fn1, fn2, fn3, fn4, fn5, fn6 = st.columns(6)
    fn1.metric("1. Proposed", cnts.get("proposed_count", len(st.session_state.promos)))
    fn2.metric("2. Validated (20-60s)", cnts.get("validated_count", len(st.session_state.promos)))
    fn3.metric("3. Deduplicated (IoU)", cnts.get("deduped_count", len(st.session_state.promos)))
    fn4.metric("4. Passed Gates", cnts.get("shortlisted_count", len(st.session_state.promos)))
    fn5.metric("5. Top 20 Shortlist", min(20, cnts.get("shortlisted_count", len(st.session_state.promos))))
    fn6.metric("6. Final Top 10", len(st.session_state.promos))

    st.divider()

    # 2. Live Runtime & Token Usage
    st.markdown("#### 2. Runtime & Token Usage")
    rc1, rc2, rc3, rc4 = st.columns(4)
    provider_str = meta_info.get("provider", "groq").upper()
    primary_str = meta_info.get("primary_model", "openai/gpt-oss-120b")
    rc1.metric("Model Config", provider_str, primary_str)
    calls_count = t_usage.get("total_calls", 0)
    cache_hits = t_usage.get("cache_hits", 0)
    rc2.metric("LLM API Calls", f"{calls_count} calls", f"{cache_hits} cached")
    prompt_tok = t_usage.get("prompt_tokens", 0)
    comp_tok = t_usage.get("completion_tokens", 0)
    rc3.metric("Measured Tokens", f"{prompt_tok + comp_tok:,}", f"In: {prompt_tok:,} | Out: {comp_tok:,}")
    elapsed_sec = meta_info.get("total_elapsed_seconds", 0.0)
    rc4.metric("Pipeline Latency", f"{elapsed_sec:.2f}s", f"Mode: {meta_info.get('inference_mode', 'live')}")

    st.divider()

    # 3. Live Boundary Repair & Quality Audit
    st.markdown("#### 3. Boundary Repair & Quality Audit")
    aud1, aud2, aud3 = st.columns(3)
    aud1.metric("Clips Algorithmic Trimmed", v_audit.get("trimmed_count", 0))
    aud2.metric("Clips Rejected", v_audit.get("rejected_count", 0))
    rejection_reasons = v_audit.get("rejection_reasons", {})
    reasons_str = ", ".join(f"{k}: {v}" for k, v in rejection_reasons.items()) if rejection_reasons else "None"
    aud3.metric("Rejection Reasons", reasons_str)

    st.divider()

    # 4. Live Candidate Distribution Across Processed Episodes
    st.markdown("#### 4. Candidate Distribution Across Episodes")
    val_records = load_json_file(OUTPUTS_DIR / "validated_candidates.json") or []
    if val_records:
        ep_summary = {}
        for r in val_records:
            ep_id = r.get("episode_id")
            if ep_id not in ep_summary:
                ep_summary[ep_id] = {"Episode": f"Episode {ep_id}", "Candidates Found": 0, "Durations": []}
            ep_summary[ep_id]["Candidates Found"] += 1
            ep_summary[ep_id]["Durations"].append(r.get("duration", 0.0))

        ep_rows = []
        for ep_id, data in sorted(ep_summary.items()):
            dur_list = data["Durations"]
            min_dur = min(dur_list) if dur_list else 0.0
            max_dur = max(dur_list) if dur_list else 0.0
            ep_rows.append({
                "Episode": data["Episode"],
                "Candidates Discovered": data["Candidates Found"],
                "Min Duration": f"{min_dur:.1f}s",
                "Max Duration": f"{max_dur:.1f}s",
                "Avg Duration": f"{sum(dur_list)/len(dur_list):.1f}s" if dur_list else "0.0s"
            })
        st.dataframe(pd.DataFrame(ep_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No candidate distribution data available.")


# =============================================================================
# TAB 4: ALL CANDIDATES (SORTABLE & FILTERABLE SELECTION AUDIT)
# =============================================================================
with tabs[3]:
    st.subheader("All Candidates Considered Across Pipeline")
    st.caption("Filter and sort every candidate to answer: 'Why wasn't this clip chosen?'")

    val_records = load_json_file(OUTPUTS_DIR / "validated_candidates.json") or []
    eval_dict = load_json_file(OUTPUTS_DIR / "evaluations.json") or {}
    final_dict = {p.candidate_id: p.rank for p in st.session_state.promos}

    if not val_records:
        st.info("No candidates currently loaded. Run the pipeline in the 'Run Pipeline' tab to generate candidates.")
    else:
        rows = []
        for c in val_records:
            cid = c.get("candidate_id")
            eid = c.get("episode_id")
            d = c.get("duration", 0.0)
            risk = c.get("spoiler_risk", "low")
            is_evaluated = cid in eval_dict
            ev_data = eval_dict.get(cid, {})

            if is_evaluated:
                hs = ev_data.get("hook_score", 0.0)
                cs = ev_data.get("cliffhanger_score", 0.0)
                ss = ev_data.get("standalone_score", 0.0)
                final_sc = (0.25*hs + 0.25*cs + 0.20*ss + 0.15*ev_data.get("emotion_score", 0.0) + 0.15*ev_data.get("promo_suitability", 0.0)) * 10.0
                hs_disp = f"{hs:.1f}"
                cs_disp = f"{cs:.1f}"
                sc_disp = f"{final_sc:.1f}"
            else:
                hs = 0.0
                cs = 0.0
                hs_disp = "-"
                cs_disp = "-"
                sc_disp = "-"
                final_sc = 0.0

            if cid in final_dict:
                status_str = f"Top 10 (Rank #{final_dict[cid]})"
                why_status = "Selected in final Top 10 by Listwise Ranker."
            elif not is_evaluated:
                status_str = "Pruned: Overlap (IoU > 0.5)"
                why_status = "Pruned in Stage 4 deduplication: heavily overlaps (>50% IoU) with another candidate in this episode."
            elif d < 20.0 or d > 60.0:
                status_str = "Rejected: Duration"
                why_status = f"Duration {d:.1f}s violates 20–60s boundary rule."
            elif hs < 7.0 or cs < 7.0:
                status_str = "Rejected: Quality Gate"
                why_status = f"Failed quality threshold (Hook: {hs:.1f}, Cliffhanger: {cs:.1f} < 7.0)."
            else:
                status_str = "Shortlisted (Outside Top 10)"
                why_status = "Passed quality gates, but fell outside Top 10 diversity quota (max 2 per episode)."

            rows.append({
                "Candidate ID": cid,
                "Episode": eid,
                "Status": status_str,
                "Duration (s)": round(d, 1),
                "Timestamps": f"{c.get('start_time', 0.0):.1f}s – {c.get('end_time', 0.0):.1f}s",
                "Spoiler Risk": risk.upper(),
                "Hook Score": hs_disp,
                "Cliff Score": cs_disp,
                "Score (/100)": sc_disp,
                "Selection / Rejection Reason": why_status,
                "Hook Line": c.get("hook_quote", "")[:50] + "...",
                "Cliffhanger Line": c.get("cliffhanger_quote", "")[:50] + "..."
            })

        df_all = pd.DataFrame(rows)

        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            ep_options = sorted(list(df_all["Episode"].unique()))
            chosen_eps = st.multiselect("Filter by Episode", ep_options, default=ep_options[:5] if len(ep_options) > 5 else ep_options)
        with f2:
            status_options = list(df_all["Status"].unique())
            chosen_status = st.multiselect("Filter by Status", status_options, default=status_options)
        with f3:
            risk_options = list(df_all["Spoiler Risk"].unique())
            chosen_risks = st.multiselect("Filter by Spoiler Risk", risk_options, default=risk_options)

        # Filtered View
        view_df = df_all[
            (df_all["Episode"].isin(chosen_eps)) &
            (df_all["Status"].isin(chosen_status)) &
            (df_all["Spoiler Risk"].isin(chosen_risks))
        ]

        st.markdown(f"**Showing {len(view_df)} of {len(df_all)} candidates:**")
        st.dataframe(view_df, use_container_width=True, hide_index=True)
