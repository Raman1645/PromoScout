# PromoScout — Complete Architecture & Risk Mitigation Plan

## 1. Objective

Build a reliable, repeatable system that identifies and ranks compelling 20–60-second promotional clips from a 50-episode screenplay pack.

A good promo should:
- Establish an interesting hook within the first 3 seconds.
- Be understandable without requiring prior knowledge of the show.
- End on an unresolved cliffhanger or compelling moment.
- Preserve the original screenplay's meaning and dialogue.
- Fit the 20–60-second duration constraint.
- Be suitable for attracting new viewers.

The system must produce traceable outputs: every proposed clip should map back to specific screenplay events and timestamps.

## 2. End-to-End Architecture

**Pipeline:**

1. Input ingestion
2. Screenplay parsing and normalization
3. Episode event timeline construction
4. Candidate promo generation
5. Timestamp mapping and duration repair
6. Candidate validation and evidence checking
7. Deduplication
8. Cold-read quality evaluation
9. Scoring and listwise ranking
10. Reporting and evaluation

The architecture separates deterministic Python processing from LLM-based creative judgment. The LLM proposes and evaluates content; Python controls timestamps, constraints, validation, reproducibility, and metrics.

## 3. Component Design and Mitigations

### A. Input Ingestion and Screenplay Parsing

**Purpose:** Convert screenplay files into a structured representation that downstream components can process consistently.

**Proposed implementation:**
- Extract all episode files from the input archive.
- Identify episode boundaries, scene headings, dialogue, action, camera directions, and sound effects.
- Parse timestamps and convert them into numeric seconds.
- Assign every parsed event a stable, episode-specific event ID.
- Preserve the original text alongside normalized fields.

**Event representation:**

- `event_id`
- `episode_id`
- `timestamp_seconds`
- `event_type`
- `speaker`
- `text`
- `scene_id`

**Risks and mitigations:**

| Risk | Mitigation |
|---|---|
| Inconsistent screenplay formatting | Use tolerant parsing rules instead of relying on one exact regex format. |
| Missing or malformed timestamps | Record parsing errors and flag affected episodes rather than silently inventing timestamps. |
| Dialogue split across lines | Normalize dialogue blocks while retaining their original text. |
| Duplicate event IDs | Generate IDs deterministically within each episode. |
| Parser works only on supplied files | Test against deliberately modified formatting and additional screenplay examples. |

**Acceptance criteria:**
- Every parsed event has a valid episode ID and stable event ID.
- Timestamps are numeric and chronologically ordered.
- Unparsed content and parsing warnings are recorded.
- A parser test suite covers common formatting variations.

### B. Episode Timeline Construction

**Purpose:** Give the candidate generator a compact but complete representation of each episode.

Each episode is converted into an ordered event timeline. Every event includes its timestamp, type, and text.

Example:

`[E012 | 11.3s | DIALOGUE | CHARACTER: "You have no idea what I've done."]`

**Important design decision:** Include elapsed seconds in the event representation, but do not ask the LLM to calculate or invent clip timestamps.

The LLM should select event IDs. Python should map those IDs to actual timestamps.

**Mitigations:**
- Preserve chronological order.
- Keep dialogue and meaningful action visible to the model.
- Distinguish visible actions from production-only camera instructions.
- Keep event IDs unique within an episode.
- Record token usage and prompt size.

### C. Candidate Promo Generation

**Purpose:** Identify promising sections of an episode that could work as standalone promotional clips.

**Approach:**

Send the complete compact episode timeline to the proposer LLM. Ask it to return candidate segments using event IDs.

Expected output fields:

- `episode_id`
- `start_event_id`
- `end_event_id`
- `hook_quote`
- `cliffhanger_quote`
- `reason`
- `spoiler_risk`

The proposer sees the complete episode and can therefore identify narrative moments and potential cliffhangers.

**Important constraint:** The proposer must not generate timestamps. It only selects event boundaries.

**Candidate generation rules:**
- Prefer openings with dialogue or meaningful action.
- Avoid starting on an uninformative camera shot.
- Prefer endings containing a revelation, threat, question, interruption, or unresolved action.
- Include an episode-ending candidate as a fallback.
- Generate multiple candidates where the episode contains distinct promotional moments.

**Risks and mitigations:**

| Risk | Mitigation |
|---|---|
| LLM invents timestamps | Accept only event IDs and derive timestamps in Python. |
| Weak or generic candidates | Require explicit hook and cliffhanger evidence. |
| Missing episode-ending moments | Always consider an episode-ending fallback. |
| Excessive candidate generation | Apply deterministic validation and deduplication before expensive evaluation. |
| Narrative spoilers | Ask the proposer to classify spoiler risk. |

### D. Timestamp Mapping and Duration Repair

**Purpose:** Convert proposed event boundaries into valid clip timestamps.

Python looks up the selected event IDs in the episode timeline.

For a candidate:

`duration = end_timestamp - start_timestamp`

The candidate must be between 20 and 60 seconds.

**Duration repair strategy:**
- If the duration is too short, extend the boundary using adjacent meaningful events.
- If too long, trim adjacent events without removing the hook or cliffhanger.
- Do not arbitrarily modify timestamps.
- Reject the candidate if a safe repair is not possible.

Record:
- Original duration
- Repaired duration
- Whether repair was applied
- Reason for rejection, if applicable

**Key safeguard:** Duration repair must preserve the hook and cliffhanger evidence. A candidate should not become valid merely because Python trimmed away its strongest ending.

### E. Candidate Validation and Evidence Checking

**Purpose:** Reject structurally invalid candidates before sending them for expensive LLM evaluation.

Validation checks:

1. Episode exists.
2. Start and end event IDs exist.
3. Start event occurs before end event.
4. Duration is between 20 and 60 seconds after safe repair.
5. Opening event is meaningful.
6. Ending event is meaningful.
7. Hook quote exists in the selected opening evidence.
8. Cliffhanger quote exists in the selected ending evidence.
9. Quotes are not fabricated or unrelated to the selected boundaries.

**Evidence validation:**
- Normalize whitespace and punctuation before matching.
- Require meaningful exact quote matching rather than accepting a single shared word.
- Hook evidence should occur at the beginning of the segment, ideally within the first 3 seconds.
- Cliffhanger evidence should occur near the end of the segment, ideally within the final 2 seconds.
- Allow an adjacent reaction event only when explicitly supported by the candidate representation.

**Failure handling:** Reject invalid candidates with a machine-readable reason. Do not silently pass them to the evaluator.

### F. Deduplication

**Purpose:** Avoid evaluating or recommending multiple versions of essentially the same clip.

Use temporal Intersection over Union (IoU):

`IoU = overlap_duration / union_duration`

Candidates from the same episode with temporal IoU greater than 0.5 are treated as duplicates.

**Mitigations:**
- Deduplicate before expensive evaluation.
- Retain one candidate using deterministic criteria.
- Do not merge boundaries from different candidates.
- After evaluation, prefer the candidate with stronger quality evidence.
- Apply a maximum of two final promos per episode to maintain episode coverage.

Avoid using arbitrary properties such as the length of an LLM-generated explanation as a proxy for quality.

### G. Cold-Read Quality Evaluation

**Purpose:** Determine whether a candidate works for someone who has never watched the show.

This is a separate LLM call from the proposer.

The evaluator receives only the selected clip content, not the complete episode, show synopsis, or future narrative context.

**Critical design decision:** Neutralize character identities.

Replace character names with labels such as `[SPEAKER A]`, `[SPEAKER B]`, etc. This prevents the evaluator from relying on names, established relationships, or knowledge of the larger story.

The evaluator should assess:

| Dimension | What it measures |
|---|---|
| Hook strength | Does the opening grab attention within 3 seconds? |
| Standalone clarity | Can a new viewer understand the situation? |
| Cliffhanger strength | Does the ending create curiosity or suspense? |
| Emotional intensity | Does the segment create an emotional reaction? |
| Promo suitability | Would the segment encourage someone to watch the show? |

Each dimension receives a score from 1–10 with a short evidence-based explanation.

**Evaluation safeguards:**
- Include only content a viewer could see or hear.
- Do not treat production-only camera directions as visible story events.
- Do not assume relationships based on speaker labels.
- Require explanations grounded in actual clip content.
- Avoid exposing the evaluator to information outside the candidate.

### H. Spoiler Risk and Quality Gates

The proposer has access to the entire episode, so it can identify whether a candidate reveals a major development.

Use a separate `spoiler_risk` field:

- Low: Does not reveal a major plot development.
- Medium: Reveals meaningful information but preserves curiosity.
- High: Reveals a major twist or resolves an important mystery.

A possible scoring adjustment:

- Low: 0 penalty
- Medium: −5 points
- High: −10 points

Treat this as an initial heuristic, not a validated measure.

**Hard quality gates:**
- Duration: 20–60 seconds.
- Hook score: at least 7/10.
- Cliffhanger score: at least 7/10.
- Evidence checks must pass.

Report the number of candidates rejected at each gate. If too few candidates survive, inspect the reasons and calibrate the thresholds rather than silently lowering them.

### I. Scoring and Final Ranking

**Purpose:** Select the strongest promotional clips from the surviving candidates.

Use a weighted score as an initial filtering mechanism.

Suggested weights:

| Dimension | Weight |
|---|---:|
| Hook strength | 25% |
| Cliffhanger strength | 25% |
| Standalone clarity | 20% |
| Emotional intensity | 15% |
| Promo suitability | 15% |

Apply the spoiler penalty after calculating the weighted score.

The weighted score should shortlist approximately the top 20 candidates.

Then use a separate listwise ranking call to compare shortlisted candidates and determine the final order.

**Ranking safeguards:**
- Randomize candidate presentation order using a recorded seed.
- Do not always present candidates in score order.
- Keep the original scores for comparison.
- Validate that the ranking response contains each candidate exactly once.
- If listwise ranking fails, fall back to the deterministic score-based ranking and record the fallback.
- Limit the final result to two promos per episode.

## 4. LLM Integration and Provider Strategy

The system requires three independent LLM tasks:

1. Candidate proposer: understands the full episode and proposes event boundaries.
2. Cold-read evaluator: assesses the clip without wider narrative context.
3. Listwise ranker: compares shortlisted clips.

### Recommended provider

Use Groq as the primary hosted inference provider, subject to approval for processing the assignment material.

Suggested initial configuration:
- Proposer: `openai/gpt-oss-120b`
- Evaluator: `openai/gpt-oss-20b`
- Ranker: `openai/gpt-oss-120b`
- API base URL: `https://api.groq.com/openai/v1`

These are starting configurations, not guaranteed optimal choices. Compare output quality and latency before finalizing.

Groq supports OpenAI-compatible requests and structured outputs for supported models. Use strict JSON Schema output where supported, followed by Pydantic validation.

### Critical mock-mode correction

The existing client automatically enters mock mode when no API key is found. This is dangerous because fixture responses can look like real inference results.

Change the behavior:
- Real inference is the default.
- Mock mode is enabled only through an explicit `--mock` flag.
- Missing API credentials must produce a clear error.
- Every run must record provider, model, and inference mode.
- Mock results must never be presented as live model results.

Support environment variables for `GROQ_API_KEY`, `GEMINI_API_KEY`, and `OPENAI_API_KEY`, with explicit provider selection.

### Reliability controls

- Retry transient HTTP failures and rate limits with exponential backoff.
- Set request timeouts.
- Validate every response against its expected schema.
- Cache responses to avoid unnecessary repeat calls.
- Include provider, model, prompt version, schema version, and input hash in cache keys.
- Track input/output tokens, latency, errors, and estimated cost.
- Do not silently switch to mock results after an API failure.

### Confidentiality

The screenplay pack is assignment material and should not be placed in a public repository.

Before sending it to any external provider, check the provider's current data-processing terms and the assignment's confidentiality requirements. Groq documents default inference data handling and temporary reliability/abuse-monitoring retention; verify applicable account settings and whether zero-data-retention arrangements are available.

Do not use Gemini's unpaid API tier for confidential screenplay material without explicit approval: its terms permit use of submitted content to improve products, and human review may occur.

## 5. Evaluation and Quality Measurement

A pipeline that produces plausible-looking clips is not necessarily a reliable pipeline. Evaluate it using independent evidence.

### A. Gold-set evaluation

Before inspecting system-generated candidates, manually annotate suitable promo moments from five episodes.

For each human-selected clip, record:
- Episode ID
- Start timestamp
- End timestamp
- Hook
- Cliffhanger
- Reason for selection

Match a system candidate to a human selection when:
- It belongs to the same episode.
- Temporal IoU is greater than 0.5.

Measure:

`HitRate@3 = episodes with at least one acceptable human selection in the top 3 / total gold-set episodes`

This measures whether the system can retrieve at least one human-approved moment near the top of its recommendations.

### B. Independent-run stability

Run the actual pipeline twice with independent inference calls.

Match candidates using the same-episode temporal IoU threshold of 0.5.

`Overlap@10 = matched clips between the two top-10 lists / 10`

Record:
- Both run IDs
- Model and prompt versions
- Randomization seeds
- Candidate overlap
- Ranking differences

Do not calculate stability using manually constructed dummy lists or fixture outputs and present it as live model stability.

### C. Failure analysis

Document at least 3–5 genuine failures, such as:
- A clip that needs prior context to make sense.
- An ending that resolves rather than creates suspense.
- A major spoiler.
- A weak opening or mid-dialogue start.
- A visually uninteresting moment.
- An interpretation that is not supported by the screenplay.

For each failure, explain the cause and the mitigation.

### D. Honest reporting

Clearly distinguish:
- Live LLM results
- Mock or fixture results
- Manually annotated gold-set data
- Estimated metrics
- Actual measured metrics

Never report a synthetic benchmark as a real evaluation result.

## 6. Operational Architecture

### CLI and execution

Support:
- `--episodes` for running selected episodes.
- `--resume` for continuing interrupted runs.
- Explicit `--mock` for fixture testing.
- Provider and model selection.
- Output directory selection.

Example:

`python run.py --episodes 1,2,3`

`python run.py --episodes 1,2,3 --resume`

`python run.py --mock --episodes 1,2,3`

### Caching and resumability

Cache results at each stage:
- Parsed episode data
- Candidate proposals
- Validation results
- Evaluation results
- Ranking results

Use stable input hashes and versioned prompts. If a run is interrupted, resume from the last completed stage instead of repeating every inference call.

### Output structure

A clean output directory should contain:

- `parsed_episodes.json`
- `proposals.json`
- `validated_candidates.json`
- `evaluations.json`
- `final_rankings.json`
- `final_rankings.csv`
- `run_metadata.json`
- `report.html`

Keep raw screenplay excerpts and confidential assignment material outside the public repository.

### Run metadata

Record:
- Run ID and timestamp
- Provider and model
- Prompt/schema versions
- Episode IDs
- Randomization seed
- Mock/live mode
- Token usage
- Latency
- Retry counts
- Candidate counts at every stage
- Rejection reasons
- Final output paths

## 7. Risk and Mitigation Matrix

| Risk | Impact | Mitigation |
|---|---|---|
| Parser fails on formatting variations | High | Tolerant parsing and altered-format tests |
| Incorrect timestamps | High | Event-ID-based mapping in Python |
| LLM invents event boundaries | High | Validate event IDs against parsed timeline |
| Invalid clip duration | High | Deterministic duration repair |
| Hook or cliffhanger quote is fabricated | High | Boundary-level exact evidence validation |
| Candidate requires prior context | High | Isolated cold-read evaluation |
| Character names leak narrative context | Medium | Speaker anonymization |
| Spoiler-heavy promo | High | Spoiler classification and penalty |
| Duplicate candidates | Medium | Temporal IoU deduplication |
| Ranking position bias | Medium | Shuffled listwise input and recorded seed |
| Provider rate limits | Medium | Backoff, caching, and resumability |
| Mock output mistaken for live inference | High | Explicit mock mode and run metadata |
| API failure produces incomplete results | High | Fail clearly or resume; never silently use fixtures |
| Synthetic metrics presented as real | High | Independent live runs and honest metric labels |
| Confidential screenplay exposure | High | Restricted storage and provider-policy checks |

## 8. Implementation Priorities

Given the limited assignment time, prioritize correctness over adding more features.

**P0 — Must fix**
1. Disable automatic mock mode.
2. Ensure real inference is used for all three LLM stages.
3. Replace timestamp generation with event-ID selection and Python mapping.
4. Fix candidate validation and duration repair.
5. Enforce evidence matching for hooks and cliffhangers.
6. Prevent evaluator access to wider narrative context.
7. Correct misleading benchmark and stability claims.

**P1 — Important**
1. Improve parser tolerance.
2. Add temporal deduplication.
3. Add deterministic quality scoring and listwise ranking.
4. Add caching, retries, and run metadata.
5. Remove confidential screenplay content from the public repository.

**P2 — Stretch goals**
1. Independent-run stability testing.
2. Gold-set HitRate@3 evaluation.
3. Model agreement analysis.
4. More advanced visual/audio suitability evaluation if actual video assets become available.

## 9. Acceptance Criteria

The system is ready for demonstration when:

- All 50 episodes can be parsed, with parsing failures reported.
- Candidate boundaries reference valid event IDs.
- Python derives all timestamps.
- Every final clip is between 20 and 60 seconds.
- Hook and cliffhanger evidence passes validation.
- Candidates are evaluated using isolated cold reads.
- Duplicate candidates are removed.
- Final ranking is reproducible and traceable.
- Live and mock runs are clearly distinguished.
- Output metrics are based on actual recorded results.
- The repository does not expose confidential screenplay material.

## 10. Final Design Principle

PromoScout should not be treated as a single LLM prompt that reads a screenplay and returns promotional clips.

It is a multi-stage decision pipeline:

- **LLMs handle semantic judgment:** finding hooks, identifying cliffhangers, evaluating standalone clarity, and comparing creative quality.
- **Python handles deterministic correctness:** parsing, event IDs, timestamp mapping, duration constraints, evidence checks, deduplication, caching, and metrics.
- **Human evaluation provides an external quality reference:** gold-set annotations and failure analysis.

This separation makes the system more reliable, explainable, testable, and reusable when KukuFM supplies a new screenplay pack.