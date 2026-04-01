"""
utils/logger.py
---------------
Central logging setup for the entire Loan Dashboard project.
Every other file imports from here — never configure logging elsewhere.

Usage in any file:
    from utils.logger import get_logger, benchmark
    log = get_logger(__name__)

    log.info("Something happened")
    log.warning("Something looks wrong")
    log.error("Something failed")
    log.debug("Detailed debug info")

    # As a decorator
    @benchmark("dax_generation")
    def generate_dax(...): ...

    # As a context manager
    with benchmark("chart_render"):
        fig = render(df, config)
"""

import sys
import time
import functools
from pathlib import Path
from contextlib import contextmanager
from loguru import logger


# ── Log directory ─────────────────────────────────────────────────────────────
# Resolves to D:\Loan_Dashboard\Logs\ regardless of where script is run from
LOG_DIR = Path(__file__).resolve().parent.parent / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Remove default Loguru handler ─────────────────────────────────────────────
logger.remove()


# ── Console handler (development) ─────────────────────────────────────────────
logger.add(
    sys.stdout,
    level="DEBUG",
    colorize=True,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    ),
)


# ── app.log — General application events ──────────────────────────────────────
logger.add(
    LOG_DIR / "app.log",
    level="INFO",
    rotation="10 MB",
    retention=5,
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    filter=lambda record: record["extra"].get("log_type", "app") == "app"
                          or "log_type" not in record["extra"],
)


# ── model.log — All Qwen/Ollama model calls ───────────────────────────────────
logger.add(
    LOG_DIR / "model.log",
    level="DEBUG",
    rotation="10 MB",
    retention=5,
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    filter=lambda record: record["extra"].get("log_type") == "model",
)


 # ── db.log — PostgreSQL database interactions ─────────────────────────────────
logger.add(
    LOG_DIR / "db.log",
    level="DEBUG",
    rotation="10 MB",
    retention=5,
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    filter=lambda record: record["extra"].get("log_type") == "db",
)


# ── charts.log — Chart rendering events ───────────────────────────────────────
logger.add(
    LOG_DIR / "charts.log",
    level="DEBUG",
    rotation="10 MB",
    retention=5,
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    filter=lambda record: record["extra"].get("log_type") == "charts",
)


# ── benchmark.log — Step-by-step timing per user query ───────────────────────
logger.add(
    LOG_DIR / "benchmark.log",
    level="DEBUG",
    rotation="10 MB",
    retention=5,
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    filter=lambda record: record["extra"].get("log_type") == "benchmark",
)


# ── Named loggers per module ──────────────────────────────────────────────────

def get_logger(name: str):
    """
    Returns a named logger bound to the calling module.
    Logs go to app.log + console by default.

    Args:
        name: Pass __name__ from calling module.

    Example:
        log = get_logger(__name__)
        log.info("File loaded successfully")
    """
    return logger.bind(name=name)


def get_model_logger(name: str):
    """
    Returns a logger that writes to model.log.
    Use in models/ folder files.

    Example:
        log = get_model_logger(__name__)
        log.info("DAX generated | tokens=342 | latency=3241ms")
    """
    return logger.bind(name=name, log_type="model")


def get_db_logger(name: str):
    """
    Returns a logger that writes to db.log.
    Use in the Database/ folder files.
    """
    return logger.bind(name=name, log_type="db")

def get_charts_logger(name: str):
    """
    Returns a logger that writes to charts.log.
    Use in charts/ folder files.
    """
    return logger.bind(name=name, log_type="charts")


def get_benchmark_logger():
    """
    Returns a logger bound to benchmark.log.
    Used internally by the benchmark context manager and decorator.
    """
    return logger.bind(log_type="benchmark")


# ── Benchmark context manager ─────────────────────────────────────────────────

_bench_log = get_benchmark_logger()


@contextmanager
def benchmark(step_name: str, query: str = ""):
    """
    Context manager that times a code block and logs to benchmark.log.

    Args:
        step_name: Label for this step (e.g., "dax_generation", "chart_render")
        query:     Optional user query string for context

    Usage:
        with benchmark("powerbi_fetch", query=user_question):
            df = client.execute_dax(dax)

    Output in benchmark.log:
        2024-02-01 10:23:41 | [START] dax_generation | query="show bounced loans"
        2024-02-01 10:23:45 | [END]   dax_generation | duration=3241ms | status=OK
    """
    start = time.perf_counter()
    q_str = f' | query="{query[:80]}"' if query else ""
    _bench_log.debug(f"[START] {step_name}{q_str}")
    status = "OK"
    try:
        yield
    except Exception as e:
        status = f"ERROR: {type(e).__name__}: {str(e)[:100]}"
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        _bench_log.debug(
            f"[END]   {step_name} | duration={elapsed_ms}ms | status={status}"
        )


def benchmark_decorator(step_name: str):
    """
    Decorator version of benchmark. Times the entire function call.

    Usage:
        @benchmark_decorator("dax_generation")
        def generate_dax(question, filters):
            ...

    Output in benchmark.log:
        [START] dax_generation | fn=generate_dax
        [END]   dax_generation | duration=3241ms | status=OK
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start  = time.perf_counter()
            fn_str = f" | fn={func.__name__}"
            _bench_log.debug(f"[START] {step_name}{fn_str}")
            status = "OK"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = f"ERROR: {type(e).__name__}: {str(e)[:100]}"
                raise
            finally:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                _bench_log.debug(
                    f"[END]   {step_name}{fn_str} | duration={elapsed_ms}ms | status={status}"
                )
        return wrapper
    return decorator


# ── Full query benchmark logger ───────────────────────────────────────────────

class QueryBenchmark:
    """
    Tracks and logs the full end-to-end timing of a single user query,
    broken into named steps.

    Usage:
        qb = QueryBenchmark(user_question)
        qb.start("chromadb_fetch")
        # ... do chromadb fetch ...
        qb.end("chromadb_fetch")
        qb.start("clarifying_questions")
        # ... call Qwen ...
        qb.end("clarifying_questions")
        qb.report()  # writes full summary to benchmark.log

    Output:
        ══════════════════════════════════════════
        QUERY: "show me bounced loans by branch"
          ├── chromadb_fetch          :    42ms
          ├── clarifying_questions    :  3241ms
          ├── dax_generation          :  4876ms
          ├── powerbi_fetch           :  1203ms
          ├── chart_decision          :   891ms
          └── chart_render            :   312ms
        TOTAL                         : 10565ms
        ══════════════════════════════════════════
    """

    def __init__(self, query: str):
        self.query    = query
        self.steps    = {}   # step_name -> {"start": float, "end": float}
        self.order    = []   # preserve insertion order

    def start(self, step: str):
        self.steps[step] = {"start": time.perf_counter(), "end": None}
        if step not in self.order:
            self.order.append(step)

    def end(self, step: str):
        if step in self.steps:
            self.steps[step]["end"] = time.perf_counter()

    def report(self):
        lines   = ["═" * 50]
        lines.append(f'QUERY: "{self.query[:80]}"')
        total   = 0

        for i, step in enumerate(self.order):
            s = self.steps.get(step, {})
            t_start = s.get("start")
            t_end   = s.get("end")

            if t_start and t_end:
                ms    = int((t_end - t_start) * 1000)
                total += ms
                is_last = i == len(self.order) - 1
                prefix  = "└──" if is_last else "├──"
                lines.append(f"  {prefix} {step:<28}: {ms:>6}ms")
            else:
                lines.append(f"  ├── {step:<28}: incomplete")

        lines.append(f"{'TOTAL':<32}: {total:>6}ms")
        lines.append("═" * 50)

        for line in lines:
            _bench_log.debug(line)