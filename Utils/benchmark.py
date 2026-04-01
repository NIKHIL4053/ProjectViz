"""
utils/benchmark.py
------------------
# * Dedicated benchmarking utilities for the Loan Dashboard project.
# * Builds on top of utils/logger.py — always import logger first.

# ? Why a separate file from logger.py?
# ? logger.py handles WHERE logs go (files, rotation, format).
# ? benchmark.py handles HOW we measure and report performance.
# ? Keeping them separate makes each file single-responsibility.

Exports:
    - StepTimer        : Measure individual named steps with start/stop
    - QueryBenchmark   : Full end-to-end query timing with summary report
    - benchmark        : Context manager for quick block timing
    - benchmark_decorator : Function decorator for automatic timing
"""

import time
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_benchmark_logger, get_logger

# * Module level loggers
log       = get_logger(__name__)
bench_log = get_benchmark_logger()


# ──────────────────────────────────────────────────────────────────────────────
# * STEP TIMER
# * Measures a single named step — start / stop / elapsed
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepTimer:
    """
    # * Tracks timing for a single named step.
    # * Used internally by QueryBenchmark to store per-step results.

    Attributes:
        name     : Human readable step name (e.g. "dax_generation")
        start_ts : Timestamp when step started (perf_counter)
        end_ts   : Timestamp when step ended (perf_counter), None if not ended
        error    : Error message if step failed, None if successful
    """

    name:     str
    start_ts: float             = field(default_factory=time.perf_counter)
    end_ts:   Optional[float]   = None
    error:    Optional[str]     = None

    def stop(self, error: Optional[str] = None):
        # * Record the end time and optional error message
        self.end_ts = time.perf_counter()
        self.error  = error

    @property
    def elapsed_ms(self) -> int:
        # ! Returns 0 if stop() was never called — always call stop()
        if self.end_ts is None:
            return 0
        return int((self.end_ts - self.start_ts) * 1000)

    @property
    def is_complete(self) -> bool:
        return self.end_ts is not None

    @property
    def status(self) -> str:
        if not self.is_complete:
            return "INCOMPLETE"
        if self.error:
            return f"ERROR: {self.error}"
        return "OK"


# ──────────────────────────────────────────────────────────────────────────────
# * QUERY BENCHMARK
# * Full end-to-end timing for one user query across all pipeline steps.
# * This is the main class you will use in the pipeline orchestrator.
# ──────────────────────────────────────────────────────────────────────────────

class QueryBenchmark:
    """
    # * Tracks and reports full end-to-end timing for a single user query.
    # * Each step in the pipeline calls start() then end() with a step name.
    # * call report() at the end to write the full summary to benchmark.log.

    # ? Pipeline steps in order:
    # ?   1. chromadb_fetch        - Fetch relevant columns from ChromaDB
    # ?   2. clarifying_questions  - Qwen Coder generates clarifying questions
    # ?   3. dax_generation        - Qwen Coder generates DAX query
    # ?   4. powerbi_fetch         - Power BI REST API executes DAX
    # ?   5. chart_decision        - Qwen 7B decides chart type and axes
    # ?   6. chart_render          - Seaborn renders the final chart(s)

    Usage:
        qb = QueryBenchmark(user_question="show bounced loans by branch")

        qb.start("chromadb_fetch")
        results = chroma.query(...)
        qb.end("chromadb_fetch")

        qb.start("dax_generation")
        dax = qwen_coder.generate(...)
        qb.end("dax_generation")

        qb.report()
    """

    # * Canonical step names — used for consistent log output
    STEPS = [
        "chromadb_fetch",
        "clarifying_questions",
        "sql_generation",
        "db_fetch",
        "chart_decision",
        "chart_render",
    ]

    def __init__(self, user_question: str = ""):
        self.user_question  = user_question
        self._steps: dict[str, StepTimer] = {}
        self._order: list[str]            = []   # preserve insertion order

    # ── Start a step ──────────────────────────────────────────────────────────

    def start(self, step: str):
        """
        # * Begin timing a named step.
        # ! If you call start() twice for the same step, it resets the timer.

        Args:
            step : Step name (use STEPS constants above for consistency)
        """
        self._steps[step] = StepTimer(name=step)
        if step not in self._order:
            self._order.append(step)
        bench_log.debug(f"[START] {step:<30} | query=\"{self.user_question[:60]}\"")

    # ── End a step ────────────────────────────────────────────────────────────

    def end(self, step: str, error: Optional[str] = None):
        """
        # * Stop timing a named step and log the result immediately.

        Args:
            step  : Same step name used in start()
            error : Optional error message if the step failed
        """
        if step not in self._steps:
            # ! Called end() without a matching start() — log warning and skip
            log.warning(f"benchmark.end() called for '{step}' but start() was never called")
            return

        timer = self._steps[step]
        timer.stop(error=error)

        status_str = f"status={timer.status}"
        bench_log.debug(
            f"[END]   {step:<30} | {timer.elapsed_ms:>6}ms | {status_str}"
        )

    # ── Convenience: start + end around a callable ────────────────────────────

    def measure(self, step: str, fn, *args, **kwargs):
        """
        # * Shorthand — start step, call fn(*args, **kwargs), end step.
        # * Automatically captures errors.

        Usage:
            result = qb.measure("dax_generation", qwen.generate, question, filters)
        """
        self.start(step)
        try:
            result = fn(*args, **kwargs)
            self.end(step)
            return result
        except Exception as e:
            self.end(step, error=f"{type(e).__name__}: {str(e)[:80]}")
            raise

    # ── Report ────────────────────────────────────────────────────────────────

    def report(self):
        """
        # * Write a formatted summary table to benchmark.log.
        # * Call this at the very end of the pipeline after all steps complete.

        Output format in benchmark.log:
            ══════════════════════════════════════════════════
            QUERY : "show me bounced loans by branch"
            ├── chromadb_fetch             :     42ms  OK
            ├── clarifying_questions       :   3241ms  OK
            ├── dax_generation             :   4876ms  OK
            ├── powerbi_fetch              :   1203ms  OK
            ├── chart_decision             :    891ms  OK
            └── chart_render               :    312ms  OK
            ══════════════════════════════════════════════════
            TOTAL                          :  10565ms
            ══════════════════════════════════════════════════
        """
        divider = "═" * 52
        lines   = [divider]
        lines.append(f"QUERY : \"{self.user_question[:72]}\"")

        total_ms    = 0
        step_count  = len(self._order)

        for i, step in enumerate(self._order):
            timer    = self._steps.get(step)
            is_last  = (i == step_count - 1)
            prefix   = "└──" if is_last else "├──"

            if timer and timer.is_complete:
                total_ms   += timer.elapsed_ms
                status_tag  = "✗ " + (timer.error or "") if timer.error else "✓"
                lines.append(
                    f"  {prefix} {step:<28}: {timer.elapsed_ms:>6}ms  {status_tag}"
                )
            else:
                lines.append(f"  {prefix} {step:<28}:   ????ms  INCOMPLETE")

        lines.append(divider)
        lines.append(f"  {'TOTAL':<30}: {total_ms:>6}ms")
        lines.append(divider)

        for line in lines:
            bench_log.debug(line)

    # ── Summary dict — useful for Streamlit debug panel ───────────────────────

    def to_dict(self) -> dict:
        """
        # * Returns timing results as a plain dict.
        # * Use this to display benchmark results in the Streamlit UI debug panel.

        Returns:
            {
                "query": "...",
                "steps": {"chromadb_fetch": 42, "dax_generation": 3241, ...},
                "total_ms": 10565
            }
        """
        steps_dict = {
            step: self._steps[step].elapsed_ms
            for step in self._order
            if step in self._steps and self._steps[step].is_complete
        }
        return {
            "query":    self.user_question,
            "steps":    steps_dict,
            "total_ms": sum(steps_dict.values()),
        }


# ──────────────────────────────────────────────────────────────────────────────
# * CONTEXT MANAGER — benchmark()
# * Quick way to time any block of code without a full QueryBenchmark.
# * Useful for one-off measurements inside a function.
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def benchmark(step_name: str, query: str = ""):
    """
    # * Context manager for quick block-level benchmarking.
    # * Writes directly to benchmark.log — no QueryBenchmark object needed.

    Args:
        step_name : Label for this block (e.g. "chart_render")
        query     : Optional query string for context in the log

    Usage:
        with benchmark("chart_render", query=user_question):
            fig = renderer.render(df, config)
    """
    q_str = f" | query=\"{query[:60]}\"" if query else ""
    bench_log.debug(f"[START] {step_name}{q_str}")
    start  = time.perf_counter()
    status = "OK"

    try:
        yield
    except Exception as e:
        status = f"ERROR: {type(e).__name__}: {str(e)[:80]}"
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        bench_log.debug(
            f"[END]   {step_name:<30} | {elapsed_ms:>6}ms | status={status}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# * DECORATOR — benchmark_decorator()
# * Wraps an entire function — logs start and end automatically.
# * Use when you always want to time a specific function.
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_decorator(step_name: str):
    """
    # * Decorator that times every call to the decorated function.
    # * Logs to benchmark.log automatically.

    Usage:
        @benchmark_decorator("dax_generation")
        def generate_dax(question: str, filters: dict) -> str:
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bench_log.debug(f"[START] {step_name:<30} | fn={func.__name__}")
            start  = time.perf_counter()
            status = "OK"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = f"ERROR: {type(e).__name__}: {str(e)[:80]}"
                raise
            finally:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                bench_log.debug(
                    f"[END]   {step_name:<30} | {elapsed_ms:>6}ms | fn={func.__name__} | status={status}"
                )
        return wrapper
    return decorator