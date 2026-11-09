"""Unified, cached, rate-limited LLM client supporting Groq, OpenAI, Gemini, and explicit mock replay."""

import os
import sys
import json
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any, Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ConfigurationError(Exception):
    """Raised when configuration or required credentials are missing."""
    pass


class LLMClient:
    """
    Provider-agnostic LLM client with Groq as default primary provider.
    Enforces strict mock-mode (no silent fallback), disk caching with run_id isolation,
    sliding-window rate limiting, and structured JSON validation.
    """

    def __init__(
        self,
        provider: str = "groq",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        run_id: str = "default",
        rpm_limit: int = 15,
        cache_dir: str = "cache",
        no_cache: bool = False,
        mock_mode: bool = False,
        fixtures_dir: str = "fixtures"
    ):
        self.provider = provider.lower()
        self.mock_mode = mock_mode
        self.temperature = temperature
        self.run_id = run_id
        self.rpm_limit = rpm_limit
        self.cache_dir = Path(cache_dir)
        self.no_cache = no_cache
        self.fixtures_dir = Path(fixtures_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)

        # 1. Base URL & Model resolution
        if self.provider == "groq":
            self.base_url = (base_url or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
            self.model = model or "openai/gpt-oss-120b"
            self.api_key = api_key or os.getenv("GROQ_API_KEY") or ""
        elif self.provider == "openai":
            self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
            self.model = model or "gpt-4o-mini"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        elif self.provider == "gemini":
            self.base_url = (base_url or os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/")
            self.model = model or "gemini-2.5-flash"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
            print("[PRIVACY CAUTION] Gemini unpaid API terms allow Google to use content for product improvement. Verify against confidentiality requirements.")
        else:
            self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
            self.model = model or "openai/gpt-oss-120b"
            self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or ""

        # 2. Strict Mock-Mode Enforcement (NO silent auto-fallback)
        if not self.mock_mode and not self.api_key:
            raise ConfigurationError(
                f"No API key provided for provider '{self.provider}'. "
                f"Please set {self.provider.upper()}_API_KEY in your environment, or pass --api-key. "
                "To run deterministic offline tests without API calls, explicitly provide the '--mock' flag."
            )

        self.inference_mode = "mock" if self.mock_mode else "live"

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
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 60.0]

        if len(self._request_timestamps) >= self.rpm_limit:
            sleep_needed = 60.0 - (now - self._request_timestamps[0]) + 0.1
            if sleep_needed > 0:
                time.sleep(sleep_needed)

        self._request_timestamps.append(time.time())

    def _get_cache_key(self, prompt: str, schema_name: str, model_override: Optional[str] = None) -> str:
        """Generate cache key incorporating provider, model, prompt, temp, schema, and run_id."""
        effective_model = model_override or self.model
        raw_key = f"{self.provider}:{effective_model}:{self.temperature}:{self.run_id}:{schema_name}:{prompt}"
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
        model_override: Optional[str] = None,
        fixture_name: Optional[str] = None,
        fixture_key: str = "default",
        max_retries: int = 1
    ) -> T:
        """
        Execute an LLM call enforcing JSON output and validating against response_model.
        Supports per-call model_override (e.g. for evaluator model).
        """
        effective_model = model_override or self.model
        schema_name = response_model.__name__
        cache_key = self._get_cache_key(prompt, schema_name, model_override=effective_model)

        # 1. Check Cache
        cached = self._read_cache(cache_key)
        if cached is not None and "content" in cached:
            try:
                validated = response_model.model_validate_json(cached["content"])
                return validated
            except ValidationError:
                pass

        # 2. Handle Explicit Mock Mode
        if self.mock_mode:
            self.total_calls += 1
            if fixture_name:
                fixture_data = self._replay_from_fixture(fixture_name, fixture_key)
                if fixture_data:
                    validated = response_model.model_validate(fixture_data)
                    self._write_cache(cache_key, {
                        "content": validated.model_dump_json(),
                        "provider": self.provider,
                        "model": effective_model,
                        "run_id": self.run_id,
                        "inference_mode": "mock",
                        "source": "fixture"
                    })
                    return validated

            raise RuntimeError(
                f"Mock mode active but fixture '{fixture_name}' key '{fixture_key}' not found."
            )

        # 3. Live API Call with Retry
        last_error = None
        current_prompt = prompt

        for attempt in range(max_retries + 1):
            try:
                self._throttle()
                url = f"{self.base_url}/chat/completions"
                payload = {
                    "model": effective_model,
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
                    "provider": self.provider,
                    "model": effective_model,
                    "run_id": self.run_id,
                    "temperature": self.temperature,
                    "inference_mode": "live",
                    "usage": usage,
                    "duration_seconds": round(time.time() - start_t, 2)
                })

                return validated

            except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, ValidationError) as e:
                last_error = e
                if attempt < max_retries:
                    if isinstance(e, (ValidationError, json.JSONDecodeError)):
                        current_prompt = (
                            f"{prompt}\n\n"
                            f"CRITICAL: Your previous response failed validation with error: {str(e)}.\n"
                            f"Ensure you return ONLY valid JSON matching schema: {response_model.model_json_schema()}"
                        )
                    time.sleep(2.0 * (attempt + 1))
                else:
                    break

        raise RuntimeError(f"LLM call to {self.provider} ({effective_model}) failed after {max_retries + 1} attempts: {last_error}")

    def get_usage_summary(self) -> Dict[str, Any]:
        """Return token accounting and performance summary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "inference_mode": self.inference_mode,
            "run_id": self.run_id,
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }
