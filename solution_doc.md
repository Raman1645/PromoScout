# PromoScout: Automated Promotional Clip Discovery & Ranking System
**Candidate Solution Document — Kuku AI Generalist Assignment (Problem 1: Find the Promos / Ads)**  
**Author:** AI Generalist Candidate  
**Target Show:** *I Married a Broken Soldier. He was a Duke* (Episodes 1–50)  
**Repository:** [https://github.com/Raman1645/PromoScout](https://github.com/Raman1645/PromoScout)  

---

## 1. Problem Choice & Framing

Microdramas live or die by performance marketing on TikTok, Instagram Reels, and YouTube Shorts. Each 1–2 minute vertical episode is densely packed with drama, but editors currently spend hours manually scrubbing through footage to find 20–60-second promo clips. With weekly series releases, this manual editorial process is an unsustainable bottleneck.

### Scoping & Sensible Assumptions
Where the assignment brief was silent, we established clear operational assumptions:
1. **The Promo Hook (First 3 Seconds):** Scrolling social media users decide within 3 seconds whether to stay. Therefore, the clip must open on immediate narrative ignition—a slap, a public humiliation, or a shocking reveal—at the screenplay's timestamp resolution (0.1 s).
2. **The Cliffhanger (Unresolved Tension):** The clip must cut precisely during the climax of tension (an unanswered ultimatum, gasps after an accusation) rather than letting the scene resolve.
3. **Standalone Comprehension (Cold Read):** Viewers encountering the ad have never seen the series. The dramatic conflict must be self-evident without needing external lore or character lineage.
4. **Operating Boundaries:** Screenplay event timestamps define boundaries; dialogue, action, camera, and sound lines are treated as events at 0.1 s resolution.
5. **Confidentiality:** Screenplay texts are proprietary. Pipeline prompts, screenplays, and raw outputs are kept out of public repositories; the public GitHub repository contains clean code, documentation, test suites, and deliverables.

---

## 2. Approach, Architecture & Scale

### Division of Labor: Python vs. LLM
Our core design principle is:
> **"Python does parsing, timestamps, boundaries and rules; LLMs do candidate discovery, creative evaluation and comparative ranking."**

- **Deterministic Python Engine:**
  - Fast regex parser extracting 3,727 events across 50 episodes (mean duration 123.2 s, range 69.2–223.7 s) with zero misses.
  - Duration validation enforcing strict 20.0s to 60.0s bounds.
  - Algorithmic trimming for clips $> 60\text{s}$ (stripping non-dialogue buffer while preserving dialogue).
  - Feedback-driven proposer retry for clips $< 20\text{s}$ (prompting the model once with explicit duration feedback).
  - Strict post-repair validation: verbatim quote verification in source text, hook starting $\le 3.0\text{s}$ from clip start, cliffhanger ending $\le 5.0\text{s}$ from clip end.
  - Intra-episode candidate deduplication ($\text{IoU} > 0.5$): No LLM involved; avoids spending evaluation calls on near-duplicate cuts.
  - Diversity constraints: maximum 2 promos per episode across the final Top 10.
- **Creative LLM Stages:**
  - **Models Actually Executed:** The live pipeline run used Groq-hosted `openai/gpt-oss-120b` for candidate proposal across Episodes 1–27, and `openai/gpt-oss-20b` for cold-read evaluation and listwise ranking. The codebase also supports OpenAI `gpt-4o-mini` interchangeably via CLI and UI.
  - **Stage 1 (Proposer - `openai/gpt-oss-120b`):** Inspects compact episode event feeds and proposes 1–3 candidate promos per episode to capture distinct dramatic beats (confrontation, reveal, climax) while avoiding candidate explosion. Allocating 2,500 completion tokens ensures reasoning models have sufficient headroom to complete thinking without returning empty JSON.
  - **Stage 2 (Cold-Read Evaluator - `openai/gpt-oss-20b`):** True zero-context evaluation. Neutralizes character names to `[SPEAKER A]`, `[SPEAKER B]`, strips camera directions, and uses relative timestamps ($00.00\text{s} = \text{start}$) to isolate dialogue tension and prevent protagonist bias or lore assumptions. Scores 5 dimensions: Hook ($H$), Cliffhanger ($C$), Standalone ($A$), Emotion ($E$), and Promo Suitability ($P$).
  - **Stage 3 (Listwise Ranker - `openai/gpt-oss-20b`):** Candidate order is shuffled to eliminate position bias. Answers *"Which would you post first, and why?"* and outputs a strict permutation of candidate IDs with editorial rationales.
- **Resilience & Checkpointing:**
  - Passing `--resume` stores stage outputs per episode on disk (`cache/checkpoints/`). Completed episode stages are loaded directly from disk upon resumption and never re-queried, protecting progress against interruptions.
  - Sliding-window rate limiting parses suggested wait times from HTTP 429 headers (`try again in X.Xs`) with automatic backoff.
  - An explicit fixture mode (`--mock`) enables reproducible testing without network calls.

### Decision Logic & Scoring Formula
Candidates must be 20–60 seconds and score at least 7/10 on both hook and cliffhanger. The weighted score is:
$$S = (0.25H + 0.25C + 0.20A + 0.15E + 0.15P) \times 10$$
Medium spoiler risk incurs a 5-point penalty on the final 100-point score (high risk incurs a 10-point penalty).

### Scalability & Token Economics (Projected Estimates)
- **Token Efficiency:** A 50-episode show contains $\approx 140\text{k}$ screenplay input tokens. End-to-end processing requires $\approx 50$ proposer calls, $\approx 100$ cold-read evaluations, and 1 listwise ranking call, totaling roughly 150k input tokens and 60k completion tokens ($\approx 210\text{k}$ total tokens).
- **Cost Estimate (as of November 2026 pricing, not measured billing):**
  - On OpenAI `gpt-4o-mini` (\$0.15/1M input, \$0.60/1M output), the estimated cost is **under \$0.06 per 50-episode show** (under 10 cents).
  - On flagship `gpt-4o` (\$2.50/\$10.00), it is roughly **\$0.95 per show**.
  - On Groq on-demand, it is $\approx \mathbf{\$0.05\text{--}\$0.15}$ per show.
- **Human-in-the-Loop:** Editors review the interactive Streamlit Studio (`app.py`), accepting or rejecting ranked clips. Decisions are appended to `outputs/feedback.csv` to build a preference dataset for recalibrating scoring gates and weights over time.

---

## 3. Quality Checking, Benchmarks & Honest Failures

To evaluate retrieval quality objectively, we manually labelled promo windows for five sample episodes (1, 13, 25, 38, 50) before inspecting system output. HitRate@3 measures whether at least one of an episode's three candidate proposals overlaps the annotated gold window with temporal $\text{IoU} > 0.50$, where $\text{IoU} = \frac{\text{overlap}(A, B)}{\text{union}(A, B)}$. A high IoU verifies that the model identified the human-chosen dramatic scene window; it measures retrieval accuracy, while the cold-read evaluator measures standalone viewer appeal.

### 3.1 Empirical Findings Summary

| Check | Status | Empirical Findings and Engineering Evidence |
|---|---|---|
| **Parser, all 50 episodes** | Done | 3,727 events extracted across 50 episodes (mean duration 123.2 s, range 69.2–223.7 s). 17 unit tests pass unconditionally on synthetic fixtures; 3 integration tests verify the full 50-episode pack when present locally (skipped when proprietary screenplays are omitted for repository confidentiality). |
| **Live run, Episodes 1–27** | Done (Live) | 27 continuous episodes (2,032 events) proposed live with `gpt-oss-120b`. 56 raw candidates extracted; Stage 4 temporal deduplication pruned 5 overlapping variants ($\text{IoU} > 0.5$), yielding 51 distinct candidates. Evaluated live with `gpt-oss-20b`: 12 passed gates ($H \ge 7, C \ge 7$), yielding a live Top 10 with editorial rationales (Rank #1: Ep 12 gallows threat, Score 78.2; Rank #2: Ep 19 informant mystery, Score 73.5). 53 live API calls, 82,168 tokens, 533.6 s runtime. |
| **Partial live Gold Set (Ep 1, 13, 25)** | Done (Live) | 3 of the 5 gold episodes fall within the live 27-episode run. Across their candidate pools ($\le 3$ proposals per episode), best candidate IoU per episode: Ep 1 = 0.773 (Hit), Ep 13 = 0.738 (Hit), Ep 25 = 0.000 (Miss, model selected later confession). Partial live HitRate@3 is **2/3 (66.7%)**. |
| **Harness & Stability** | Mock only, not model evidence | Replaying static offline fixtures yields HitRate@3 of 4/5 and Overlap@10 of 10/10. This validates metric calculation plumbing, not live model quality or live stability. |
| **Full-series run (Ep 28–50)** | Paused | Paused at Ep 28 when Groq free-tier reached its daily limit (200,000 tokens/day for `gpt-oss-120b`). The reported 82,168 tokens (53 calls) reflect the evaluation and ranking stage on `openai/gpt-oss-20b`; earlier candidate proposals across Episodes 1–27 combined with prior test iterations accounted for the remaining daily quota on `gpt-oss-120b`. Processing can resume via `--resume`. |

### 3.2 Where the AI Got It Wrong (Honest Failure Cases)

1. **Episode 25 Gold Set Miss ($\text{IoU} = 0.000$):**
   *Observed:* Human annotation targeted the opening injury confrontation (5.1–32.5 s, *"Why are you walking like that?"*). The live proposer instead selected later scenes (47.0–97.4 s and 128.0–156.2 s), prioritizing mid-episode emotional confessions over the opening confrontation beat.
   *Editorial Takeaway:* Proposers need explicit weighting to favor cold-open confrontations when an episode contains multiple dramatic peaks.
2. **Episode 1 Dual-Gate Rejection:**
   *Observed:* In Episode 1, the model proposed 4 candidate cuts, but the cold-read evaluator gave all four sub-threshold hook scores ($H = 6.5 < 7.0$). Lines like *"I changed my mind"* provided conversational exposition rather than immediate visceral shock.
   *Editorial Takeaway:* The quality gate correctly rejected all four, producing zero promos for Episode 1 rather than lowering editorial standards.
3. **Screenplay vs. Video Modality Gap (Expected Risk):**
   *Observed:* Screenplay text lacks visual framing, sound design, and micro-acting nuances; an editor must visually verify whether a high-scoring text exchange holds visual energy on screen.

---

## 4. Prototype & Production Deliverables

- **CLI Tool (`run.py`):** Modular CLI supporting Groq and OpenAI models (`--provider groq --model openai/gpt-oss-20b`), `--episodes`, `--rpm`, `--stage`, `--resume`, `--inspect`, and `--mock` (fixture replay without API keys).
- **Interactive Web Studio (`app.py`):** Streamlit application with 4 interactive tabs: Top 10 Promos, Run Pipeline (supports Groq & OpenAI with masked keys), Live Pipeline Metrics (100% active execution data), and All Candidates Audit.
- **Evaluation Suite (`evaluate.py`):** Benchmarking harness for $\text{HitRate@3}$ and dual-run $\text{Overlap@10}$.
- **Reporting (`outputs/`):**
  - `final_rankings.json`: Structured JSON containing live Top 10 promos, timestamps, verbatim quotes, cold-read scores, and editorial rationales.
  - `final_rankings.csv`: Production table for media buyers.
  - `report.html`: Standalone, interactive dashboard with responsive promo cards, score breakdowns, and verbatim quotes.
  - `run_metadata.json`: Complete execution audit with token accounting and validator statistics.
- **Test Suite (`tests/`):** 20 automated tests: 17 unit tests and 3 integration tests passing (`pytest tests`).
