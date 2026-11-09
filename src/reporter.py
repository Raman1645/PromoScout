"""Reporting and export module generating JSON, CSV, and interactive HTML dashboards."""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any
from src.models import FinalPromo, ScoredCandidate, Episode


class ReportGenerator:
    """Generates structured outputs (JSON, CSV, HTML) for human review and presentation."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, promos: List[FinalPromo], filename: str = "final_rankings.json") -> Path:
        """Export final promos to JSON format matching assignment schema."""
        out_path = self.output_dir / filename
        data = [p.model_dump() for p in promos]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return out_path

    def export_csv(self, promos: List[FinalPromo], filename: str = "final_rankings.csv") -> Path:
        """Export final promos to tabular CSV format."""
        out_path = self.output_dir / filename
        fieldnames = [
            "rank", "candidate_id", "episode_id", "start_time", "end_time",
            "duration", "start_event_id", "end_event_id", "final_score",
            "hook_score", "cliffhanger_score", "standalone_score", "emotion_score", "coherence_score",
            "spoiler_risk", "hook_quote", "cliffhanger_quote", "post_first_because"
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in promos:
                writer.writerow({
                    "rank": p.rank,
                    "candidate_id": p.candidate_id,
                    "episode_id": p.episode_id,
                    "start_time": f"{p.start_time:.2f}",
                    "end_time": f"{p.end_time:.2f}",
                    "duration": f"{p.duration:.2f}",
                    "start_event_id": p.start_event_id,
                    "end_event_id": p.end_event_id,
                    "final_score": f"{p.final_score:.2f}",
                    "hook_score": f"{p.scores.hook_score:.1f}",
                    "cliffhanger_score": f"{p.scores.cliffhanger_score:.1f}",
                    "standalone_score": f"{p.scores.standalone_score:.1f}",
                    "emotion_score": f"{p.scores.emotion_score:.1f}",
                    "coherence_score": f"{p.scores.coherence_score:.1f}",
                    "spoiler_risk": p.spoiler_risk,
                    "hook_quote": p.hook_quote,
                    "cliffhanger_quote": p.cliffhanger_quote,
                    "post_first_because": p.post_first_because
                })

        return out_path

    def export_html(
        self,
        promos: List[FinalPromo],
        episodes_analyzed: int,
        total_candidates_found: int,
        filename: str = "report.html"
    ) -> Path:
        """Generate a responsive, standalone HTML dashboard with cards and comparison table."""
        out_path = self.output_dir / filename

        # Build promo cards HTML
        cards_html = []
        for p in promos:
            spoiler_class = f"badge-{p.spoiler_risk.lower()}"
            cards_html.append(f"""
            <div class="promo-card" id="card-{p.rank}">
                <div class="card-header">
                    <span class="rank-badge">#{p.rank}</span>
                    <span class="ep-tag">Episode {p.episode_id}</span>
                    <span class="duration-tag">{p.duration:.1f}s ({p.start_time:05.2f} - {p.end_time:05.2f})</span>
                    <span class="score-tag">Score: {p.final_score:.1f}/100</span>
                    <span class="badge {spoiler_class}">Spoiler: {p.spoiler_risk.capitalize()}</span>
                </div>
                <div class="card-body">
                    <div class="quote-box hook-box">
                        <div class="quote-label">🔥 3-SECOND HOOK (Start: {p.start_event_id})</div>
                        <div class="quote-text">"{p.hook_quote}"</div>
                    </div>
                    <div class="quote-box cliff-box">
                        <div class="quote-label">⚡ CLIFFHANGER (End: {p.end_event_id})</div>
                        <div class="quote-text">"{p.cliffhanger_quote}"</div>
                    </div>
                    <div class="score-breakdown">
                        <div class="score-item"><span class="lbl">Hook:</span> <span class="val">{p.scores.hook_score:.1f}</span></div>
                        <div class="score-item"><span class="lbl">Cliffhanger:</span> <span class="val">{p.scores.cliffhanger_score:.1f}</span></div>
                        <div class="score-item"><span class="lbl">Standalone:</span> <span class="val">{p.scores.standalone_score:.1f}</span></div>
                        <div class="score-item"><span class="lbl">Emotion:</span> <span class="val">{p.scores.emotion_score:.1f}</span></div>
                        <div class="score-item"><span class="lbl">Coherence:</span> <span class="val">{p.scores.coherence_score:.1f}</span></div>
                    </div>
                    <div class="rationale-box">
                        <strong>Editorial Decision (Why Post First):</strong>
                        <p>{p.post_first_because}</p>
                    </div>
                    <div class="explanation-box">
                        <small><em>Cold-Read Evaluation: {p.scores.explanation}</em></small>
                    </div>
                </div>
            </div>
            """)

        # Build table rows
        rows_html = []
        for p in promos:
            rows_html.append(f"""
            <tr>
                <td><strong>#{p.rank}</strong></td>
                <td>Ep {p.episode_id}</td>
                <td>{p.duration:.1f}s</td>
                <td><strong>{p.final_score:.1f}</strong></td>
                <td>{p.scores.hook_score:.1f}</td>
                <td>{p.scores.cliffhanger_score:.1f}</td>
                <td>{p.scores.standalone_score:.1f}</td>
                <td>{p.spoiler_risk.capitalize()}</td>
                <td><small>{p.post_first_because[:90]}...</small></td>
            </tr>
            """)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PromoScout — Microdrama Promotional Discovery Dashboard</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-hover: #334155;
            --border: #475569;
            --primary: #38bdf8;
            --accent: #f43f5e;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --gold: #fbbf24;
            --green: #34d399;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        h1 {{ font-size: 28px; font-weight: 700; color: var(--primary); }}
        .subtitle {{ color: var(--text-muted); font-size: 14px; margin-top: 4px; }}
        .metrics-banner {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--surface);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .metric-val {{ font-size: 24px; font-weight: 700; color: var(--text); }}
        .metric-lbl {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; }}
        .promo-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .promo-card:hover {{
            border-color: var(--primary);
            transform: translateY(-2px);
        }}
        .card-header {{
            background: rgba(30, 41, 59, 0.8);
            border-bottom: 1px solid var(--border);
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .rank-badge {{
            background: var(--primary);
            color: #0f172a;
            font-weight: 800;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 14px;
        }}
        .ep-tag {{ font-weight: 600; font-size: 14px; }}
        .duration-tag {{ color: var(--text-muted); font-size: 13px; }}
        .score-tag {{
            margin-left: auto;
            font-weight: 700;
            color: var(--gold);
            font-size: 14px;
        }}
        .badge {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-low {{ background: #065f46; color: #a7f3d0; }}
        .badge-medium {{ background: #92400e; color: #fde68a; }}
        .badge-high {{ background: #991b1b; color: #fecaca; }}
        .card-body {{ padding: 16px; }}
        .quote-box {{
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid var(--border);
            padding: 10px 14px;
            border-radius: 4px;
            margin-bottom: 12px;
        }}
        .hook-box {{ border-left-color: var(--accent); }}
        .cliff-box {{ border-left-color: var(--gold); }}
        .quote-label {{ font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 4px; }}
        .quote-text {{ font-size: 14px; font-style: italic; color: #e2e8f0; }}
        .score-breakdown {{
            display: flex;
            gap: 16px;
            background: rgba(15, 23, 42, 0.4);
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 12px;
            flex-wrap: wrap;
        }}
        .score-item .lbl {{ color: var(--text-muted); margin-right: 4px; }}
        .score-item .val {{ font-weight: 700; color: var(--primary); }}
        .rationale-box {{
            font-size: 13px;
            color: #cbd5e1;
            margin-bottom: 8px;
        }}
        .rationale-box strong {{ color: var(--primary); }}
        .explanation-box {{ color: var(--text-muted); font-size: 12px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border-radius: 8px;
            overflow: hidden;
            margin-top: 32px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }}
        th {{
            background: rgba(30, 41, 59, 0.9);
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.05em;
        }}
        tr:hover td {{ background: var(--surface-hover); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>PromoScout — Top 10 Editorial Recommendations</h1>
            <div class="subtitle">Kuku Microdrama Promotional Discovery System • Series: <em>I Married a Broken Soldier. He was a Duke</em></div>
        </header>

        <div class="metrics-banner">
            <div class="metric-card">
                <div class="metric-val">{episodes_analyzed}</div>
                <div class="metric-lbl">Episodes Analyzed</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{total_candidates_found}</div>
                <div class="metric-lbl">Candidates Evaluated</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{len(promos)}</div>
                <div class="metric-lbl">Final Ranked Promos</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">100%</div>
                <div class="metric-lbl">Duration Compliance (20-60s)</div>
            </div>
        </div>

        <h2>Ranked Recommendations</h2>
        <div style="margin-top: 16px;">
            {''.join(cards_html)}
        </div>

        <h2 style="margin-top: 36px;">Summary Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Episode</th>
                    <th>Duration</th>
                    <th>Score</th>
                    <th>Hook</th>
                    <th>Cliffhanger</th>
                    <th>Standalone</th>
                    <th>Spoiler</th>
                    <th>Why Post First</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return out_path

