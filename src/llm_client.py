"""Unified, cached, rate-limited LLM client supporting OpenAI-compatible APIs and fixture replay."""

import os
import json
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any, Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Provider-agnostic LLM client with disk caching, rate limiting, and fixture replay."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        run_id: str = "default",
        rpm_limit: int = 15,
        cache_dir: str = "cache",
        no_cache: bool = False,
        mock_mode: bool = False,
        fixtures_dir: str = "fixtures"
    ):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.model = model
        self.temperature = temperature
        self.run_id = run_id
        self.rpm_limit = rpm_limit
        self.cache_dir = Path(cache_dir)
        self.no_cache = no_cache
        self.mock_mode = mock_mode or (not self.api_key and not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"))
        self.fixtures_dir = Path(fixtures_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting state
        self._request_timestamps: list[float] = []

        # Token & Call Accounting
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.cache_hits = 0

    def _throttle(self) -> None:
        """Enforce requests-per-minute limit using a sliding window."""
        if self.rpm_limit <= 0 or self.mock_mode:
            return

        now = time.time()
        # Keep only timestamps within the last 60 seconds
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 60.0]

        if len(self._request_timestamps) >= self.rpm_limit:
            sleep_needed = 60.0 - (now - self._request_timestamps[0]) + 0.1
            if sleep_needed > 0:
                time.sleep(sleep_needed)

        self._request_timestamps.append(time.time())

    def _get_cache_key(self, prompt: str, schema_name: str) -> str:
        """Generate cache key incorporating model, prompt, temperature, schema, and run_id."""
        raw_key = f"{self.model}:{self.temperature}:{self.run_id}:{schema_name}:{prompt}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Read cached response if available."""
        if self.no_cache:
            return None
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.cache_hits += 1
                    return data
            except Exception:
                return None
        return None

    def _write_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save response to disk cache."""
        if self.no_cache:
            return
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _replay_from_fixture(self, fixture_name: str, fallback_key: str) -> Dict[str, Any]:
        """Replay response from a static fixture file."""
        fixture_file = self.fixtures_dir / f"{fixture_name}.json"
        if fixture_file.exists():
            with open(fixture_file, "r", encoding="utf-8") as f:
                fixtures = json.load(f)
                if isinstance(fixtures, dict) and fallback_key in fixtures:
                    return fixtures[fallback_key]
                elif isinstance(fixtures, dict) and "default" in fixtures:
                    return fixtures["default"]
                return fixtures
        return {}

    def call_json(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: str = "You are a professional microdrama video editor. Always return valid JSON conforming to the requested schema.",
        fixture_name: Optional[str] = None,
        fixture_key: str = "default",
        max_retries: int = 1
    ) -> T:
        """
        Execute an LLM call enforcing JSON output and validating against response_model.
        Supports caching, RPM throttling, and deterministic fixture replay.
        """
        schema_name = response_model.__name__
        cache_key = self._get_cache_key(prompt, schema_name)

        # Check Cache
        cached = self._read_cache(cache_key)
        if cached is not None and "content" in cached:
            try:
                validated = response_model.model_validate_json(cached["content"])
                return validated
            except ValidationError:
                pass

        # Handle Mock Mode
        if self.mock_mode:
            self.total_calls += 1
            if fixture_name:
                fixture_data = self._replay_from_fixture(fixture_name, fixture_key)
                if fixture_data:
                    validated = response_model.model_validate(fixture_data)
                    self._write_cache(cache_key, {
                        "content": validated.model_dump_json(),
                        "model": self.model,
                        "run_id": self.run_id,
                        "source": "fixture"
                    })
                    return validated

            raise RuntimeError(
                f"Mock mode active but fixture '{fixture_name}' key '{fixture_key}' not found. "
                "Provide API keys or register fixture before running."
            )

        # Live API Call with Retry
        last_error = None
        current_prompt = prompt

        for attempt in range(max_retries + 1):
            try:
                self._throttle()
                url = f"{self.base_url}/chat/completions"
                payload = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": current_prompt}
                    ]
                }

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    method="POST"
                )

                start_t = time.time()
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp_body = json.loads(resp.read().decode("utf-8"))

                self.total_calls += 1
                usage = resp_body.get("usage", {})
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)

                raw_content = resp_body["choices"][0]["message"]["content"]

                # Validate against Pydantic model
                validated = response_model.model_validate_json(raw_content)

                # Save to cache
                self._write_cache(cache_key, {
                    "content": raw_content,
                    "model": self.model,
                    "run_id": self.run_id,
                    "temperature": self.temperature,
                    "usage": usage,
                    "duration_seconds": round(time.time() - start_t, 2)
                })

                return validated

            except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, ValidationError) as e:
                last_error = e
                if attempt < max_retries:
                    # Retry with corrective prompt on validation/json error
                    if isinstance(e, (ValidationError, json.JSONDecodeError)):
                        current_prompt = (
                            f"{prompt}\n\n"
                            f"CRITICAL: Your previous response failed validation with error: {str(e)}.\n"
                            f"Ensure you return ONLY valid JSON matching schema: {response_model.model_json_schema()}"
                        )
                    time.sleep(2.0 * (attempt + 1))
                else:
                    break

        raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {last_error}")

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return token accounting and performance summary."""
        return {
            "model": self.model,
            "run_id": self.run_id,
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }

