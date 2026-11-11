# PromoScout — Automated Promotional Clip Discovery & Ranking

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)]()
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-orange.svg)](https://docs.pydantic.dev/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-PromoScout-blue?logo=github)](https://github.com/Raman1645/PromoScout)

> **Kuku AI Generalist Assignment — Problem 1: Find the Promos (Ads)**  
> **Repository:** [https://github.com/Raman1645/PromoScout](https://github.com/Raman1645/PromoScout)  
> An AI-assisted editorial tool that ingests microdrama screenplays, extracts timestamped events to 0.01s resolution, discovers high-converting 20–60s promotional clips, validates boundaries deterministically, filters through quality gates, and outputs a listwise-ranked Top 10 promo package for social platforms (TikTok, Instagram Reels, YouTube Shorts).

---

## 1. System Architecture

PromoScout operates on a fundamental division of labor:
> **"Python owns timestamps, boundaries, and deterministic validation; LLMs own creative judgment, candidate discovery, and comparative ranking."**

```
+-------------------------------------------------------------+
|               Screenplay Files (50 Episodes)                |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  1. Regex Screenplay Parser (src/parser.py)                 |
|     - Extracts scenes, dialogues, actions, cameras, sounds  |
|     - Canonical event IDs (E01_001), 0.01s timestamp floats |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  2. LLM Candidate Proposer (src/proposer.py)               |
|     - Proposes 1-3 candidates per episode                   |
|     - Verbatim quotes, spoiler risk (low/medium/high)       |
|     - Single feedback retry if proposed duration < 20s      |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  3. Boundary Validator & Duration Repair (src/validator.py) |
|     - Algorithmic trimming of non-dialogue buffer if > 60s  |
|     - Post-repair validation: verbatim quote match,         |
|       hook in first 3.0s, cliffhanger in final 2.0-5.0s     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  4. Candidate Deduplication (src/deduplication.py)          |
|     - Intra-episode IoU > 0.5 pruning                       |
|     - Retains candidate with lower spoiler / stronger hook  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  5. Cold-Read LLM Evaluator (src/evaluator.py)              |
|     - True zero-context evaluation of clip dialogue         |
|     - Speaker anonymization: [SPEAKER A], [SPEAKER B]       |
|     - Production CAMERA directions filtered out             |
|     - Relative timestamps (00.00s = start), IDs stripped    |
|     - Powered by Groq openai/gpt-oss-20b                    |
|     - Scores 1-10 on Hook, Cliffhanger, Standalone,         |
|       Emotion, Promo Suitability                            |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  6. Quality Gates & Weighted Scoring (src/scoring.py)       |
|     - Gates: Hook >= 7, Cliffhanger >= 7, 20s <= Dur <= 60s |
|     - S = 0.25H + 0.30C + 0.20A + 0.15E + 0.10N            |
|     - Spoiler penalty: Med (-5), High (-10)                 |
|     - Establishes per-episode ranking & Top 20 shortlist    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  7. Final Listwise Ranker (src/ranker.py)                   |
|     - Shuffles candidates to eliminate position bias        |
|     - Asks: "Which would you post first, and why?"          |
|     - Validates full 20-candidate permutation in Python     |
|     - Diversity constraints: max 2/episode, IoU <= 0.5      |
|     - Generates Top 10 with 'post_first_because' rationale  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  8. Reporting Engine (src/reporter.py)                      |
|     - outputs/final_rankings.json                           |
|     - outputs/final_rankings.csv                            |
|     - outputs/report.html (Interactive Visual Dashboard)    |
|     - raw_outputs/ (Unedited intermediate execution logs)   |
+-------------------------------------------------------------+
```

---

## 2. Target File Structure

```
kukuFM/
├── .venv/                                       # Dedicated virtual environment
├── .gitignore                                   # Confidentiality: ignores pack, cache, pdf, plan
├── requirements.txt                             # Pydantic & Pytest
├── README.md                                    # This documentation
├── solution_doc.md                              # 2-page solution document
├── run.py                                       # Main CLI orchestrator
├── evaluate.py                                  # Evaluation suite (Gold Set & Stability)
├── src/
│   ├── models.py                                # Pydantic data schemas
│   ├── parser.py                                # Regex screenplay parser
│   ├── llm_client.py                            # Cached, RPM-limited LLM client + mock replay
│   ├── proposer.py                              # Stage 1: Candidate proposer + retry for <20s
│   ├── validator.py                             # Stage 2: Boundary check, trimming, quote/position checks
│   ├── deduplication.py                         # Stage 3: Intra-episode IoU deduplication
│   ├── evaluator.py                             # Stage 4: Relative-time zero-context cold-read evaluator
│   ├── scoring.py                               # Stage 5: Quality gates, weighted scoring, per-episode Top 3
│   ├── ranker.py                                # Stage 6: Listwise ranking & constraints
│   └── reporter.py                              # Stage 7: JSON, CSV, HTML report generation
├── tests/
│   ├── test_parser.py                           # 5 tests: 50 episodes + malformed handling
│   ├── test_validator.py                        # 4 tests: boundaries, trimming, quote/position checks
│   ├── test_deduplication.py                    # 2 tests: IoU calculation & deduplication
│   └── test_scoring.py                          # 4 tests: quality gates, weighted formulas, per-episode ranking
├── fixtures/                                    # Deterministic replay fixtures for --mock mode
│   ├── mock_proposals.json
│   ├── mock_evaluations.json
│   └── mock_ranking.json
├── gold_set/
│   ├── gold_annotations.json                    # 5 systematically sampled gold episodes (1, 13, 25, 38, 50)
│   └── benchmark_results.json                   # Output metrics: HitRate@3 & Overlap@10
├── raw_outputs/                                 # Unedited pipeline logs (shared via link-only Drive)
│   ├── parsed_screenplays.json
│   ├── proposed_candidates.json
│   ├── validated_candidates.json
│   ├── cold_read_evaluations.json
│   ├── scored_shortlist.json
│   └── listwise_ranking_raw.json
├── outputs/                                     # Human deliverables
│   ├── final_rankings.json
│   ├── final_rankings.csv
│   └── report.html                              # Standalone visual dashboard
└── docs/
    └── walkthrough_script.md                    # 3-minute video presentation script
```

---

## 3. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Raman1645/PromoScout.git
   cd PromoScout
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install minimal dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Unit Tests:**
   ```bash
   pytest
   ```
   *(All 20 tests pass in < 0.4s)*

---

## 4. Execution Guide

### Option A: Interactive Web Application (Streamlit Studio)
Launch the graphical editorial studio with real-time transparency across all 5 pipeline tabs:
```bash
streamlit run app.py
```
Tabs included:
1. **🏆 Top Ranked Promos:** Ranked cards with 3s hook, unresolved cliffhanger, and editorial *Why Post First* rationales.
2. **🎬 Screenplay Explorer:** Filterable event-level viewer across all 50 microdrama episodes.
3. **🔍 Pipeline Stages & Audit Log:** Raw JSON stage inspections plus live side-by-side **Speaker Anonymization Inspector** (raw vs. `[SPEAKER A/B]`).
4. **📊 Quality Benchmark:** Quantitative HitRate@3 breakdowns and failure mode analysis.
5. **💾 Export & Deliverables:** Direct download buttons for `final_rankings.json`, `final_rankings.csv`, `report.html`, and `solution_doc.pdf`.

### Option B: Immediate Offline Mock Mode (Zero API Key Needed)
PromoScout includes built-in fixture replay allowing full end-to-end execution without external network calls or API keys:
```bash
# Process a 3-episode sample
python run.py --episodes 1,2,3 --mock

# Process all 50 episodes
python run.py --episodes all --mock
```

### Option C: Live LLM Execution (Groq Primary / Gemini / OpenAI)

**Primary Recommendation: Groq Free Tier (GPT-OSS Models)**
Groq provides fast, free-tier rate limits for the `openai/gpt-oss-120b` and `openai/gpt-oss-20b` models:
```bash
python run.py \
  --provider groq \
  --api-key "YOUR_GROQ_API_KEY" \
  --model "openai/gpt-oss-120b" \
  --evaluator-model "openai/gpt-oss-20b" \
  --rpm 15 \
  --episodes all
```

**Using Google Gemini (Google AI Studio):**
```bash
python run.py \
  --provider gemini \
  --api-key "YOUR_GEMINI_API_KEY" \
  --model "gemini-2.5-flash" \
  --rpm 15 \
  --episodes all
```

**Using OpenAI:**
```bash
python run.py \
  --provider openai \
  --api-key "YOUR_OPENAI_API_KEY" \
  --model "gpt-4o-mini" \
  --episodes all
```

### Pipeline Checkpointing & Stage-by-Stage Audits
PromoScout features stage-level checkpointing (`src/checkpoint.py`) enabling interruption resumption and discrete audits:
```bash
# Run only parsing, proposal, and validation
python run.py --stage validate --mock

# Resume a previous run from the last completed stage
python run.py --resume --mock

# Inspect stage candidates with detailed quotes and boundaries
python run.py --stage validate --inspect --mock --episodes 1,2
```

**Available CLI Flags:**
- `--provider`: `groq` (default), `openai`, `gemini`, or `custom`.
- `--model`: Primary LLM for Proposer & Listwise Ranker (default: `openai/gpt-oss-120b`).
- `--evaluator-model`: Evaluator LLM for Cold-Read judging (default: `openai/gpt-oss-20b`).
- `--episodes`: Comma-separated list or ranges (`1,2,3`, `1-10`, or `all`).
- `--stage`: Stop execution after target stage (`parse`, `propose`, `validate`, `dedup`, `evaluate`, `rank`).
- `--resume`: Resume execution from the last persisted checkpoint.
- `--inspect`: Print detailed inspection summary for a stage and exit.
- `--temperature`: Default `0.3` (non-zero for realistic stability benchmarking).
- `--run-id`: Cache partition identifier (`default`, `run_1`, `run_2`).
- `--rpm`: Sliding-window requests-per-minute throttle (default: `15`).
- `--no-cache`: Force bypass of disk cache.
- `--mock`: Enable static fixture replay (strict: fails if API key missing without `--mock`).

---

## 5. Benchmarking & Quality Evaluation

Run the evaluation script to compute quantitative metrics against the ground truth and test run stability:
```bash
python evaluate.py --mock
```

### Benchmark Results:
- **HitRate@3 on Systematic Gold Set:** **80.0% (4/5 episodes matched)**
  - Gold episodes selected by objective narrative rule: **Episodes 1, 13, 25, 38, 50**.
  - Ep 1 (Altar Betrayal): $\text{IoU} = 0.918$ (Match)
  - Ep 13 (Thorn Bush Search): $\text{IoU} = 0.769$ (Match)
  - Ep 25 (Injury Confrontation): $\text{IoU} = 0.639$ (Match)
  - Ep 38 (Prison Kindness): $\text{IoU} = 0.130$ (Miss)
  - Ep 50 (Forbidden Meeting): $\text{IoU} = 0.829$ (Match)
- **Stability Overlap@10:** **80.0% (8/10 promos matched across independent runs at temperature 0.3)**.

---

## 6. Real Observed Failure Modes

1. **Overt Shouting Favored Over Subtle Tension (Ep 38 Miss):**
   The cold-read evaluator receives only dialogue/action text with zero backstory. In Episode 38, it scored a loud argument higher than an intimate, quiet scene with a sympathetic prison guard, missing subtle subtext.
2. **Buffer Trimming Softening Final Visual Reaction:**
   On clips exceeding 60s, trimming trailing non-dialogue buffer achieved duration compliance but occasionally cut the final visual reaction shot following a cliffhanger line.
3. **Mid-Arc Chemical Hooks Misclassified as Spoilers:**
   In Episode 25, a clip showing early romantic chemistry between Juliana and Alexander was penalized as a "Medium Spoiler", even though displaying romantic chemistry is an ad hook rather than a plot ruin.

---

## 7. Deliverables & Outputs

- **Solution Document:** [solution_doc.md](file:///c:/Users/Dell/Desktop/kukuFM/solution_doc.md) (2-page concise brief addressing all 5 assignment lenses).
- **Interactive HTML Report:** [report.html](file:///c:/Users/Dell/Desktop/kukuFM/outputs/report.html) (Visual cards, scores, quotes, and filterable table).
- **JSON Rankings:** `outputs/final_rankings.json`.
- **CSV Rankings:** `outputs/final_rankings.csv`.
- **Raw Outputs:** `raw_outputs/` (Preserved intermediate outputs for all 50 episodes; shared via link-only Google Drive folder to uphold screenplay confidentiality).
- **Walkthrough Script:** [walkthrough_script.md](file:///c:/Users/Dell/Desktop/kukuFM/docs/walkthrough_script.md) (Up-to-3-minute video presentation guide).

