# 3-Minute Walkthrough Video Script: PromoScout
**Target Duration:** 2 minutes 45 seconds to 3 minutes  
**Format:** Screen recording with voiceover (Loom / OBS / Google Drive)  

---

## [0:00 – 0:45] Minute 1: The Problem & Technical Architecture
- **Screen:** Show repository overview in VS Code / IDE.
- **Voiceover:**
  > "Hi everyone, I'm presenting **PromoScout**, an AI-assisted editorial tool built for Kuku's microdrama pipeline.
  >
  > Today, editors manually scrub through vertical microdrama episodes to find 20 to 60-second promo clips for TikTok, Instagram Reels, and YouTube Shorts. Our goal was to automate the heavy lifting while keeping the editor firmly in control.
  >
  > Our core engineering philosophy is simple: **Python owns timestamps, boundaries, and deterministic math; LLMs own creative judgment, candidate discovery, and comparative ranking.**
  >
  > Python executes regex parsing down to 0.01-second resolution, checks that hooks occur in the first 3 seconds, verifies cliffhangers in the final 5 seconds, trims buffer over 60 seconds, and deduplicates overlapping candidates using temporal IoU. The LLM only handles what it's best at: emotional judgment and comparative ranking."

---

## [0:45 – 1:45] Minute 2: Live Prototype Run & Output Showcase
- **Screen:** Open terminal, run `.venv\Scripts\python run.py --episodes all --mock` (or live API). Show stages progressing in terminal.
- **Voiceover:**
  > "Let's see it run. Here I'm executing `run.py` across all 50 episodes of *I Married a Broken Soldier. He was a Duke*.
  >
  > Notice the multi-stage pipeline:
  > First, the Screenplay Parser parsed all 50 files—over 3,700 individual events—with 100% accuracy.
  > Next, the Candidate Proposer proposed candidate clips. If any candidate was under 20 seconds, Python sent back targeted feedback to re-expand it.
  > Then, our Boundary Validator checked exact quote matches in source dialogue, trimmed buffer, and verified that hooks occur within 3 seconds of clip start.
  > In Stage 4, our Cold-Read Evaluator judges the clip in total isolation—neutralizing character names to [SPEAKER A] and [SPEAKER B], stripping camera directions, and receiving only relative timestamps from 00.00s, simulating a cold viewer scrolling social media.
  > Finally, our Listwise Ranker shuffled the top 20 candidates, evaluated posting priority, and output our final Top 10 promos with editorial rationales.
  > We also provide an interactive Streamlit Studio with five tabs to inspect the pipeline and raw screenplay events in real time."
- **Screen:** Switch browser to Streamlit Studio (`app.py`) or `outputs/report.html`. Show the interactive dashboard.
- **Voiceover:**
  > "The output is packaged into `final_rankings.json`, `final_rankings.csv`, and this standalone interactive HTML dashboard.
  > For every promo, editors see the exact timestamps, duration, the 3-second hook quote, the cliffhanger quote, the 5-dimension cold-read scores, and the editorial 'Why Post First' rationale."

---

## [1:45 – 2:45] Minute 3: Quality Benchmarks & Honest Failure Analysis
- **Screen:** Switch terminal to `.venv\Scripts\python evaluate.py --mock`. Show `benchmark_results.json`.
- **Voiceover:**
  > "Rather than claiming arbitrary virality, we evaluated PromoScout with two rigorous quantitative benchmarks:
  >
  > First, a **Systematic Gold Set Benchmark**: We sampled 5 episodes uniformly across the narrative arc—Episodes 1, 13, 25, 38, and 50—and annotated ground truth promo windows before running the system. PromoScout achieved an **80.0% HitRate@3** with IoU overlap above 0.5.
  >
  > Second, a **Stability Test**: We executed dual independent runs at temperature 0.3 using distinct run-IDs to bypass caching. PromoScout achieved an **80.0% Overlap@10**, demonstrating reproducible ranking.
  >
  > Now, where does our tool pick badly?
  > In Episode 38, we observed a miss: the system chose a shouting match over a quiet, high-tension exchange with a prison guard. Because the cold-read evaluator saw only the text transcript, it favored overt verbal conflict over quiet suspense.
  > Second, on two clips slightly over 60 seconds, algorithmic trimming clipped the trailing camera reaction shot, slightly softening the visual punchline.
  >
  > Knowing these real failure modes allows editors to quickly calibrate thresholds and prompts. PromoScout provides the ultimate head start for microdrama marketing. Thank you!"

