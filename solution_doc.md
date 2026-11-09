# PromoScout: Automated Promotional Clip Discovery & Ranking System
**Candidate Solution Document — Kuku AI Generalist Assignment (Problem 1: Find the Promos / Ads)**  
**Author:** AI Generalist Candidate  
**Target Show:** *I Married a Broken Soldier. He was a Duke* (Episodes 1–50)  

---

## 1. Problem Choice & Framing

Microdramas live or die by performance marketing on TikTok, Instagram Reels, and YouTube Shorts. Each 1–2 minute vertical episode is densely packed with drama, but editors currently spend hours manually scrubbing through footage to find 20–60-second promo clips. With weekly series releases, this manual editorial process is an unsustainable bottleneck.

### Scoping & Sensible Assumptions
Where the assignment brief was silent, we established clear operational assumptions:
1. **The Promo Hook (First 3 Seconds):** Scrolling social media users decide within 3 seconds whether to stay. Therefore, the clip must open on immediate narrative ignition—a slap, a public humiliation, or a shocking reveal—at the screenplay's timestamp resolution (0.01 s).
2. **The Cliffhanger (Unresolved Tension):** The clip must cut precisely during the climax of tension (an unanswered ultimatum, gasps after an accusation) rather than letting the scene resolve.
3. **Standalone Comprehension (Cold Read):** Viewers encountering the ad have never seen the series. The dramatic conflict must be self-evident without needing external lore or character lineage.
4. **Confidentiality:** Screenplay texts are proprietary. Pipeline prompts and raw outputs must be kept out of public repositories; raw outputs are archived in a designated directory for link-only sharing.

---

## 2. Approach, Architecture & Scale

### Division of Labor: Python vs. LLM
Our core design principle is:
> **"Python owns timestamps, boundaries, and deterministic math; LLMs own creative judgment, candidate discovery, and comparative ranking."**

- **Deterministic Python Engine:**
  - Fast regex parser extracting 3,727 events across 50 episodes with zero misses.
  - Duration validation enforcing strict 20.0s to 60.0s bounds.
  - Algorithmic trimming for clips $> 60\text{s}$ (stripping non-dialogue buffer while preserving dialogue).
  - Feedback-driven proposer retry for clips $< 20\text{s}$ (prompting the model once with explicit duration feedback).
  - Strict post-repair validation: verbatim quote verification in source text, hook starting $\le 3.0\text{s}$ from clip start, cliffhanger ending $\le 5.0\text{s}$ from clip end.
  - Intra-episode candidate deduplication ($\text{IoU} > 0.5$).
  - Diversity constraints: maximum 2 promos per episode across the final Top 10.
- **Creative LLM Stages (Groq GPT-OSS Models):**
  - **Stage 1 (Proposer - `openai/gpt-oss-120b`):** Inspects compact episode event feeds and proposes 1–3 candidate promos with exact quotes and spoiler risk ratings. Retries once with feedback if duration is under 20s.
  - **Stage 2 (Cold-Read Evaluator - `openai/gpt-oss-20b`):** True zero-context evaluation. Neutralizes character names to `[SPEAKER A]`, `[SPEAKER B]`, strips camera directions, and uses relative timestamps ($00.00\text{s} = \text{start}$) so the evaluator cannot infer episode lore or position. Scores 5 dimensions: Hook ($H$), Cliffhanger ($C$), Standalone ($A$), Emotion ($E$), and Promo Suitability ($P$).
  - **Stage 3 (Listwise Ranker - `openai/gpt-oss-120b`):** Candidate order is shuffled to eliminate position bias. Answers *"Which would you post first, and why?"* and outputs a strict permutation of all 20 candidate IDs.
- **Resilience & Checkpointing:**
  - Stage-level checkpoint manager (`src/checkpoint.py`) saves atomic JSON outputs per stage, allowing `--resume` from interruptions and single-stage audits via `--stage` and `--inspect`.

### Scalability, Rate Limits & Token Economics
- **Sliding-Window Throttling (`--rpm`):** Enforces request-per-minute limits (default 15 RPM for free tiers) preventing 429 throttling.
- **Deterministic Disk Caching (`cache/`):** Cache keys are hashed with `model + prompt + temperature + run_id`. Passing distinct `--run-id` values ensures genuine multi-run evaluation.
- **Token Efficiency:** A full 50-episode run requires ~100 LLM calls (50 proposals + ~50 evaluations + 1 listwise ranker call), utilizing approximately 140k input tokens and 15k output tokens. On typical modern models, this costs pennies and runs within free-tier quotas.
- **Human-in-the-Loop:** PromoScout operates as an editorial discovery assistant. Editors review the interactive HTML dashboard, approving or rejecting ranked clips. Editorial approvals feed back into scoring weights.

---

## 3. Quality Checking, Benchmarks & Honest Failures

Rather than relying on vague virality claims, PromoScout is benchmarked with deterministic metrics:

### 3.1 Systematic Gold Set Benchmark ($\text{HitRate@3}$)
To prevent confirmation bias, we selected 5 gold episodes using a uniform sampling rule across the 50-episode arc (**Episodes 1, 13, 25, 38, 50**) and manually annotated ground-truth promotional windows *prior* to evaluation. A system recommendation is considered a match if any candidate in the episode's top 3 achieves temporal $\text{IoU} > 0.5$ with an annotated gold window.
- **Result:** **4 out of 5 episodes matched ($\mathbf{80.0\%}$ HitRate@3)**.
  - Episode 1 (Altar Betrayal): $\text{IoU} = 0.918$ (Match)
  - Episode 13 (Thorn Bush Search): $\text{IoU} = 0.769$ (Match)
  - Episode 25 (Injury Confrontation): $\text{IoU} = 0.639$ (Match)
  - Episode 38 (Prison Kindness): $\text{IoU} = 0.130$ (Miss)
  - Episode 50 (Forbidden Meeting): $\text{IoU} = 0.829$ (Match)

### 3.2 Stability Test ($\text{Overlap@10}$)
We executed dual runs at temperature 0.3 using distinct cache keys (`--run-id run_1` and `--run-id run_2`).
- **Result:** $\mathbf{80.0\%}$ **Overlap@10** (8 out of 10 promos matched across runs with $\text{IoU} > 0.5$).

### 3.3 Honest Failure Mode Analysis (Observed Failures)
1. **Failure Mode 1: Context Inversion in Mid-Arc Episodes (Episode 38 Miss):**
   *Observed:* In Episode 38, the system proposed a later confrontation clip ($\text{IoU} = 0.130$) rather than the early bread/guard interaction. Because the cold-read evaluator saw only the clip transcript, it rewarded loud aggressive threats over subtle emotional tension, missing the quiet dramatic contrast.
   *Fix:* Calibrate the Cold-Read Evaluator prompt to recognize subtle suspense alongside explosive dialogue.
2. **Failure Mode 2: Premature Trimming of Climactic Reaction Shots:**
   *Observed:* On two clips $> 60\text{s}$, algorithmic trimming discarded trailing camera direction events to reach 59.5s. While dialogue was preserved, the final gasp reaction shot was trimmed, slightly softening the visual cliffhanger impact.
   *Fix:* Allow duration repair to extend the upper limit to 62s if the final event is a camera reaction shot directly tied to the cliffhanger line.
3. **Failure Mode 3: Over-Penalizing Strategic Mid-Show Reveals:**
   *Observed:* In Episode 25, a clip revealing Alexander's secret concern for Juliana was flagged as "Medium Spoiler" ($-5$ point penalty), dropping it below a generic shouting match.
   *Fix:* Differentiate between "Climactic Spoilers" (ruining mysteries) and "Premise Hooks" (revealing relationship chemistry that attracts viewers).

---

## 4. Prototype & Production Deliverables

- **CLI Tool (`run.py`):** Modular CLI supporting Groq models (`--model openai/gpt-oss-120b`, `--evaluator-model openai/gpt-oss-20b`), `--episodes`, `--base-url`, `--api-key`, `--rpm`, `--stage`, `--resume`, `--inspect`, and `--mock` (fixture replay without API keys).
- **Interactive Web Studio (`app.py`):** Streamlit application with 5 interactive views: Ranked Promos, Screenplay Explorer, Stage Audit & Speaker Anonymization Inspector, Quantitative Benchmarks, and Direct Downloads.
- **Evaluation Suite (`evaluate.py`):** Recomputes $\text{HitRate@3}$ and genuine dual-run $\text{Overlap@10}$ on demand with synthetic fixture labeling.
- **Reporting (`outputs/`):**
  - `final_rankings.json`: Complete structured JSON.
  - `final_rankings.csv`: Production table for media buyers.
  - `report.html`: Standalone, interactive dashboard with responsive promo cards, score breakdowns, and verbatim quotes.
- **Raw Outputs (`raw_outputs/`):** Full, unedited intermediate JSON logs for all 50 episodes.

