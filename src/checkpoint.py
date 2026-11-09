"""Stage-level checkpointing module enabling resumability (--resume) across pipeline runs."""

import json
import hashlib
from pathlib import Path
from typing import Optional, Any, Dict


class CheckpointManager:
    """Manages disk-backed stage checkpoints so interrupted runs can resume cleanly."""

    def __init__(self, checkpoint_dir: str = "cache/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, stage: str, item_id: str) -> Path:
        safe_id = hashlib.md5(item_id.encode("utf-8")).hexdigest()
        return self.checkpoint_dir / f"{stage}_{safe_id}.json"

    def save(self, stage: str, item_id: str, data: Any) -> None:
        """Save a completed stage output for an episode or candidate."""
        path = self._get_path(stage, item_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"item_id": item_id, "data": data}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def load(self, stage: str, item_id: str) -> Optional[Any]:
        """Load a completed stage output if available."""
        path = self._get_path(stage, item_id)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    return payload.get("data")
            except Exception:
                return None
        return None

    def clear(self) -> None:
        """Clear all stored checkpoints."""
        for p in self.checkpoint_dir.glob("*.json"):
            try:
                p.unlink()
            except Exception:
                pass
