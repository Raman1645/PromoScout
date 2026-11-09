"""Regex-based screenplay parser for microdrama episodes."""

import re
from pathlib import Path
from typing import List, Tuple, Optional, Union
from src.models import ScreenplayEvent, Scene, Episode


def parse_timestamp(ts_str: str) -> float:
    """Parse a timestamp string like '01:23.45' or '(00:04.80)' into float seconds."""
    clean = ts_str.strip().strip("()")
    parts = clean.split(":")
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return round(hours * 3600 + minutes * 60 + seconds, 2)
    elif len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return round(minutes * 60 + seconds, 2)
    elif len(parts) == 1:
        return round(float(parts[0]), 2)
    raise ValueError(f"Invalid timestamp format: {ts_str}")


def format_timestamp(seconds: float) -> str:
    """Format float seconds to MM:SS.SS format."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


class ScreenplayParser:
    """Parses microdrama screenplay text into structured Episode and ScreenplayEvent models."""

    EVENT_REGEX = re.compile(
        r'\s*\*\*([^*]+?):?\*\*:?\s*\r?\n'
        r'\s*Start:\s*\(([0-9:.]+)\)\s*\|\s*End:\s*\(([0-9:.]+)\)\s*\r?\n'
        r'(.*?)(?=(?:\r?\n\s*\*\*|\r?\n\s*SCENE|\Z))',
        re.DOTALL
    )

    SCENE_REGEX = re.compile(
        r'\s*SCENE\s+(\d+)\s*\r?\n'
        r'\s*Scene heading:\s*(.*?)\r?\n'
        r'\s*Scene start:\s*([0-9:.]+)\s*\r?\n'
        r'\s*Scene end:\s*([0-9:.]+)',
        re.IGNORECASE
    )

    def parse_text(self, text: str, fallback_episode_id: int = 1) -> Episode:
        """Parse raw screenplay text into an Episode object."""
        # 1. Episode Number
        ep_match = re.search(r'EPISODE NUMBER:\s*Episode\s*(\d+)', text, re.IGNORECASE)
        episode_id = int(ep_match.group(1)) if ep_match else fallback_episode_id
        ep_raw = ep_match.group(0).strip() if ep_match else f"Episode {episode_id}"

        # 2. Logline & Continuity
        logline_match = re.search(r'EPISODE LOGLINE:\s*(.*?)(?=\n[A-Z ]+:|\n---|\Z)', text, re.DOTALL | re.IGNORECASE)
        logline = logline_match.group(1).strip() if logline_match else ""

        cont_match = re.search(r'CONTINUITY NOTES:\s*(.*?)(?=\n[A-Z ]+:|\n---|\Z)', text, re.DOTALL | re.IGNORECASE)
        continuity_notes = cont_match.group(1).strip() if cont_match else ""

        # 3. Scenes
        scenes: List[Scene] = []
        for s_match in self.SCENE_REGEX.finditer(text):
            scene_num = int(s_match.group(1))
            heading = s_match.group(2).strip()
            start_t = parse_timestamp(s_match.group(3))
            end_t = parse_timestamp(s_match.group(4))
            scenes.append(Scene(
                scene_num=scene_num,
                heading=heading,
                start_time=start_t,
                end_time=end_t
            ))

        # Fallback if no scenes matched explicitly
        if not scenes:
            scenes.append(Scene(scene_num=1, heading="INT. UNKNOWN", start_time=0.0, end_time=0.0))

        # 4. Events
        events: List[ScreenplayEvent] = []
        event_idx = 1

        for match in self.EVENT_REGEX.finditer(text):
            raw_tag = match.group(1).strip().rstrip(":")
            start_str = match.group(2)
            end_str = match.group(3)
            body_text = match.group(4).strip()

            start_time = parse_timestamp(start_str)
            end_time = parse_timestamp(end_str)

            # Determine scene number based on timestamp
            scene_num = scenes[0].scene_num
            for sc in scenes:
                if sc.start_time <= start_time <= sc.end_time:
                    scene_num = sc.scene_num
                    break

            # Classify event type
            upper_tag = raw_tag.upper()
            character: Optional[str] = None
            parenthetical: Optional[str] = None

            if upper_tag == "CAMERA":
                event_type = "camera"
            elif upper_tag == "ACTION":
                event_type = "action"
            elif "SOUND" in upper_tag or "FX" in upper_tag:
                event_type = "sound"
            else:
                event_type = "dialogue"
                if "(" in raw_tag and ")" in raw_tag:
                    paren_match = re.search(r'^(.*?)\s*\((.*?)\)', raw_tag)
                    if paren_match:
                        character = paren_match.group(1).strip()
                        parenthetical = paren_match.group(2).strip()
                    else:
                        character = raw_tag
                else:
                    character = raw_tag

            event_id = f"E{episode_id:02d}_{event_idx:03d}"
            event_idx += 1

            # Clean quotation marks if dialogue
            clean_text = body_text.strip()
            if event_type == "dialogue" and clean_text.startswith('"') and clean_text.endswith('"') and len(clean_text) > 1:
                clean_text = clean_text[1:-1].strip()

            events.append(ScreenplayEvent(
                event_id=event_id,
                episode_id=episode_id,
                scene_num=scene_num,
                event_type=event_type,
                character=character,
                parenthetical=parenthetical,
                start_time=start_time,
                end_time=end_time,
                text=clean_text
            ))

        return Episode(
            episode_id=episode_id,
            episode_number_raw=ep_raw,
            logline=logline,
            continuity_notes=continuity_notes,
            scenes=scenes,
            events=events
        )

    def parse_file(self, file_path: Union[str, Path]) -> Episode:
        """Parse a screenplay file from disk."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Try to infer episode id from filename if not in text
        ep_id_infer = 1
        m = re.search(r'episode-(\d+)', path.name, re.IGNORECASE)
        if m:
            ep_id_infer = int(m.group(1))

        return self.parse_text(text, fallback_episode_id=ep_id_infer)
