"""
models/ollama_client.py
-----------------------
# * Low-level Ollama API client for the Loan Dashboard project.
# * Single entry point for ALL model calls — no other file calls Ollama directly.
# * Handles connection, retries, timeouts, streaming, and error logging.

# ? Why a dedicated client instead of calling requests.post() everywhere?
# ? If Ollama changes its API or we swap to a different local runner,
# ? we update ONE file instead of hunting through every model file.
# ? All retry logic, timeout handling, and error formatting lives here.

# ? Two models are used:
# ?   CODER_MODEL (qwen2.5-coder:14b) — intent, clarifying questions, DAX
# ?   FAST_MODEL  (qwen2.5:7b)        — chart type decisions (lightweight)

Exports:
    - OllamaClient          : Main client class
    - get_client()          : Singleton accessor
    - ModelResponse         : Dataclass for structured model output
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_URL,
    OLLAMA_TAGS_URL,
    OLLAMA_TIMEOUT,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_RETRIES,
    CODER_MODEL,
    FAST_MODEL,
)
from utils.logger import get_logger, get_model_logger
from utils.benchmark import benchmark
from utils.helpers import parse_json_safely, estimate_tokens, truncate_string

# * Module level loggers
log       = get_logger(__name__)
model_log = get_model_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * MODEL RESPONSE DATACLASS
# * Wraps every model response in a consistent structure.
# * Callers always get the same shape regardless of success or failure.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResponse:
    """
    # * Structured response from any Ollama model call.
    # * Always check .success before using .content or .parsed.

    Attributes:
        success       : True if model returned a valid response
        content       : Raw text response from the model
        parsed        : Parsed dict/list if JSON was requested, else None
        model         : Which model was called
        step          : Which pipeline step this was (e.g. "dax_generation")
        latency_ms    : Time taken for the model call in milliseconds
        prompt_tokens : Estimated input token count
        error         : Error message if success=False, else None
    """
    success:        bool
    content:        str                   = ""
    parsed:         Optional[dict | list] = None
    model:          str                   = ""
    step:           str                   = ""
    latency_ms:     int                   = 0
    prompt_tokens:  int                   = 0
    error:          Optional[str]         = None

    @property
    def failed(self) -> bool:
        """# * Convenience — True if the call failed."""
        return not self.success


# ──────────────────────────────────────────────────────────────────────────────
# * OLLAMA CLIENT
# ──────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    # * Handles all communication with the locally running Ollama instance.
    # * Used by analyzer.py, clarifier.py, dax_generator.py, chart_decider.py.

    # ? Why not use the official ollama Python library?
    # ? The requests-based approach gives us full control over:
    # ?   - Custom timeout per call
    # ?   - Retry logic with backoff
    # ?   - Structured logging of every request/response
    # ?   - Easy mock injection for testing without Ollama running
    """

    def __init__(self):
        self._session = requests.Session()
        # * Set default headers once — reused across all requests
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept":       "application/json",
        })
        log.info(f"[ollama] Client initialised | base_url={OLLAMA_BASE_URL}")

    # ── Health check ──────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """
        # * Check if Ollama is running and reachable.
        # * Used by app.py on startup and sidebar status indicator.

        Returns:
            True if Ollama responds, False otherwise.
        """
        try:
            resp = self._session.get(
                OLLAMA_BASE_URL,
                timeout=3
            )
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            log.warning("[ollama] is_running check failed — Ollama not reachable")
            return False
        except Exception as e:
            log.error(f"[ollama] is_running check error | {e}")
            return False

    def is_model_loaded(self, model: str) -> bool:
        """
        # * Check if a specific model is available in Ollama.
        # * Does NOT check if it is currently in VRAM — just that it was pulled.

        Args:
            model : Model name e.g. "qwen2.5-coder:14b"

        Returns:
            True if model is in Ollama's model list.
        """
        try:
            resp = self._session.get(OLLAMA_TAGS_URL, timeout=5)
            if resp.status_code != 200:
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            # * Check exact match or prefix match (e.g. "qwen2.5-coder:14b" in list)
            return any(model in m or m in model for m in models)
        except Exception as e:
            log.error(f"[ollama] is_model_loaded check failed | model={model} | error={e}")
            return False

    def get_loaded_models(self) -> list[str]:
        """
        # * Returns list of all model names available in Ollama.
        # * Used for sidebar model status display.
        """
        try:
            resp = self._session.get(OLLAMA_TAGS_URL, timeout=5)
            if resp.status_code == 200:
                return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    # ── Core chat call ────────────────────────────────────────────────────────

    def chat(
        self,
        model:        str,
        system:       str,
        user:         str,
        step:         str         = "unknown",
        temperature:  float       = OLLAMA_TEMPERATURE,
        expect_json:  bool        = False,
        timeout:      int         = OLLAMA_TIMEOUT,
    ) -> ModelResponse:
        """
        # * Send a chat request to Ollama and return a structured ModelResponse.
        # * Retries up to OLLAMA_MAX_RETRIES times on timeout or connection error.
        # * All prompt/response details are logged to model.log.

        Args:
            model       : Model name (use CODER_MODEL or FAST_MODEL from config)
            system      : System prompt string
            user        : User message string
            step        : Pipeline step name for logging (e.g. "dax_generation")
            temperature : Sampling temperature — keep low for DAX/JSON
            expect_json : If True, attempts to parse response as JSON
            timeout     : Request timeout in seconds

        Returns:
            ModelResponse dataclass — always check .success first.

        Example:
            response = client.chat(
                model      = CODER_MODEL,
                system     = system_prompt,
                user       = user_message,
                step       = "dax_generation",
                expect_json= False,
            )
            if response.success:
                dax = response.content
        """
        # * Estimate prompt size for logging
        prompt_tokens = estimate_tokens(system) + estimate_tokens(user)

        model_log.info(
            f"[{step}] CALL START | "
            f"model={model} | "
            f"~{prompt_tokens} prompt tokens | "
            f"expect_json={expect_json}"
        )

        payload = {
            "model":    model,
            "options":  {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
        }

        last_error = ""
        start_time = time.perf_counter()

        for attempt in range(1, OLLAMA_MAX_RETRIES + 2):  # +2 = initial + retries
            try:
                resp = self._session.post(
                    OLLAMA_CHAT_URL,
                    json    = payload,
                    timeout = timeout,
                )
                resp.raise_for_status()

                content    = resp.json()["message"]["content"].strip()
                latency_ms = int((time.perf_counter() - start_time) * 1000)

                # * Log response summary
                model_log.info(
                    f"[{step}] CALL OK | "
                    f"model={model} | "
                    f"latency={latency_ms}ms | "
                    f"~{estimate_tokens(content)} response tokens | "
                    f"preview=\"{truncate_string(content, 80)}\""
                )

                # * Attempt JSON parse if requested
                parsed = None
                if expect_json:
                    parsed = parse_json_safely(content)
                    if parsed is None:
                        # ! JSON parse failed — log but still return content
                        model_log.warning(
                            f"[{step}] JSON parse failed | "
                            f"model={model} | "
                            f"raw_preview=\"{truncate_string(content, 120)}\""
                        )

                return ModelResponse(
                    success       = True,
                    content       = content,
                    parsed        = parsed,
                    model         = model,
                    step          = step,
                    latency_ms    = latency_ms,
                    prompt_tokens = prompt_tokens,
                )

            except requests.exceptions.ConnectionError:
                last_error = "Cannot connect to Ollama — is it running? Run: ollama serve"
                model_log.error(f"[{step}] CONNECTION ERROR | attempt={attempt} | {last_error}")
                break  # ! No point retrying a connection error

            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {timeout}s"
                model_log.warning(
                    f"[{step}] TIMEOUT | "
                    f"attempt={attempt}/{OLLAMA_MAX_RETRIES + 1} | "
                    f"model={model}"
                )
                if attempt <= OLLAMA_MAX_RETRIES:
                    wait = attempt * 2   # * Exponential backoff: 2s, 4s
                    log.info(f"[ollama] Retrying in {wait}s...")
                    time.sleep(wait)

            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                model_log.error(f"[{step}] HTTP ERROR | model={model} | {last_error}")
                break  # ! HTTP errors are not retryable

            except KeyError:
                last_error = "Unexpected response format from Ollama"
                model_log.error(f"[{step}] PARSE ERROR | model={model} | {last_error}")
                break

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                model_log.error(f"[{step}] UNEXPECTED ERROR | model={model} | {last_error}")
                break

        # * All attempts exhausted — return failure response
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        model_log.error(
            f"[{step}] CALL FAILED | "
            f"model={model} | "
            f"latency={latency_ms}ms | "
            f"error={last_error}"
        )
        return ModelResponse(
            success    = False,
            model      = model,
            step       = step,
            latency_ms = latency_ms,
            error      = last_error,
        )

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def call_coder(
        self,
        system:      str,
        user:        str,
        step:        str   = "coder_call",
        expect_json: bool  = False,
    ) -> ModelResponse:
        """
        # * Shorthand for calling CODER_MODEL (qwen2.5-coder:14b).
        # * Use for: intent analysis, clarifying questions, DAX generation.

        Args:
            system      : System prompt
            user        : User message
            step        : Step label for logging
            expect_json : Parse response as JSON if True

        Returns:
            ModelResponse
        """
        with benchmark(step):
            return self.chat(
                model       = CODER_MODEL,
                system      = system,
                user        = user,
                step        = step,
                expect_json = expect_json,
            )

    def call_fast(
        self,
        system:      str,
        user:        str,
        step:        str   = "fast_call",
        expect_json: bool  = True,
    ) -> ModelResponse:
        """
        # * Shorthand for calling FAST_MODEL (qwen2.5:7b).
        # * Use for: chart type decisions, lightweight classification.
        # ? Default expect_json=True because 7B is only used for structured decisions.

        Args:
            system      : System prompt
            user        : User message
            step        : Step label for logging
            expect_json : Parse response as JSON if True (default True)

        Returns:
            ModelResponse
        """
        with benchmark(step):
            return self.chat(
                model       = FAST_MODEL,
                system      = system,
                user        = user,
                step        = step,
                expect_json = expect_json,
            )

    # ── Status summary ────────────────────────────────────────────────────────

    def status_summary(self) -> dict:
        """
        # * Returns a summary dict of Ollama status for sidebar display.
        # * Checks if Ollama is running and both models are available.

        Returns:
            {
                "ollama_running"  : bool,
                "coder_available" : bool,
                "fast_available"  : bool,
                "loaded_models"   : list[str],
            }
        """
        running = self.is_running()
        return {
            "ollama_running":   running,
            "coder_available":  self.is_model_loaded(CODER_MODEL) if running else False,
            "fast_available":   self.is_model_loaded(FAST_MODEL)  if running else False,
            "loaded_models":    self.get_loaded_models()           if running else [],
        }


# ──────────────────────────────────────────────────────────────────────────────
# * SINGLETON ACCESSOR
# ──────────────────────────────────────────────────────────────────────────────

_client_instance: Optional[OllamaClient] = None


def get_client() -> OllamaClient:
    """
    # * Returns the shared OllamaClient instance.
    # * Creates it on first call — reused on every subsequent call.

    # ? Why a singleton?
    # ? requests.Session() maintains a connection pool.
    # ? Creating a new session on every model call is wasteful.
    # ? One shared session = faster requests through connection reuse.

    Usage in any model file:
        from models.ollama_client import get_client
        client   = get_client()
        response = client.call_coder(system, user, step="dax_generation")
        if response.success:
            dax = response.content
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = OllamaClient()
    return _client_instance