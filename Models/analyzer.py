"""
models/analyzer.py
------------------
# * Pipeline Step 1 — Intent Analysis.
# * Takes the raw user question and maps it to:
# *   - The correct business metric (Resolution %, Bounce %, Coverage % etc.)
# *   - The relevant columns from the data dictionary
# *   - The columns that will be needed as slicers/filters
# *   - Whether time period is involved
# *   - The DAX aggregation type needed (COUNT, SUM, AVERAGE etc.)

# ? Why is this the first step and not DAX generation directly?
# ? Users say things like "show me bad loans in Pune" — no column names, no DAX.
# ? The analyzer bridges natural language → structured intent.
# ? The structured intent then drives every other step (clarifier, DAX gen, viz).

# ? Which model: Qwen Coder 14B
# ? Why: Needs strong instruction following and domain reasoning.
# ?      7B doesn't reliably produce structured JSON for complex loan domain queries.

Exports:
    - IntentResult     : Dataclass for structured analysis output
    - Analyzer         : Main class
    - get_analyzer()   : Singleton accessor
"""

from dataclasses import dataclass, field
from typing import Optional

from config import CODER_MODEL
from core.dictionary import get_dictionary
from models.ollama_client import get_client, ModelResponse
from utils.logger import get_logger, get_model_logger
from utils.benchmark import benchmark
from utils.helpers import truncate_string, estimate_tokens

# * Module level loggers
log       = get_logger(__name__)
model_log = get_model_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * INTENT RESULT DATACLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    """
    # * Structured output from the analyzer step.
    # * Everything the rest of the pipeline needs to proceed.

    Attributes:
        success             : True if analysis completed successfully
        raw_question        : Original user question unchanged
        intent_summary      : One sentence describing what the user wants
        metric              : Primary metric identified (e.g. "Bounce %")
        metric_key          : Internal metric key (e.g. "Bounce_Percent")
        columns_needed      : Columns required to answer the question
        slicer_candidates   : Columns that could be used as filters
        aggregation         : DAX aggregation needed (COUNT/SUM/AVERAGE/DISTINCTCOUNT)
        group_by            : Column to group results by (for charts)
        time_involved       : True if user asked about a time period
        granularity         : "customer" or "loan" level analysis
        dax_pattern_hint    : Key from 05_dax_patterns.json to use as DAX template
        confidence          : High / Medium / Low — how confident the model is
        error               : Error message if success=False
    """
    success:            bool
    raw_question:       str                = ""
    intent_summary:     str                = ""
    metric:             str                = ""
    metric_key:         str                = ""
    columns_needed:     list[str]          = field(default_factory=list)
    slicer_candidates:  list[str]          = field(default_factory=list)
    aggregation:        str                = "COUNT"
    group_by:           Optional[str]      = None
    time_involved:      bool               = False
    granularity:        str                = "customer"
    dax_pattern_hint:   Optional[str]      = None
    confidence:         str                = "High"
    error:              Optional[str]      = None

    @property
    def failed(self) -> bool:
        return not self.success

    def to_dict(self) -> dict:
        """# * Convert to plain dict for session storage and prompt injection."""
        return {
            "intent_summary":    self.intent_summary,
            "metric":            self.metric,
            "metric_key":        self.metric_key,
            "columns_needed":    self.columns_needed,
            "slicer_candidates": self.slicer_candidates,
            "aggregation":       self.aggregation,
            "group_by":          self.group_by,
            "time_involved":     self.time_involved,
            "granularity":       self.granularity,
            "dax_pattern_hint":  self.dax_pattern_hint,
            "confidence":        self.confidence,
        }


# ──────────────────────────────────────────────────────────────────────────────
# * SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

# * This prompt is injected with the relevant context chunks from ChromaDB
# * before being sent to Qwen Coder.
_SYSTEM_PROMPT_TEMPLATE = """
You are an expert data analyst for a loan collection portfolio system at an NBFC (Non-Banking Financial Company).
You deeply understand loan collection domain — DPD, buckets, bounce, resolution, NPA, coverage, and field collection operations.

Your job is to analyze what the user is asking and map it to the correct technical fields and metrics.

IMPORTANT RULES:
1. Users are NON-TECHNICAL — they will never use column names or technical terms
2. Map their words to the correct fields using the language mapping context below
3. Always use DISTINCTCOUNT for customer-level metrics — never plain COUNT
4. Use COUNT or COUNTROWS for loan-level metrics
5. The primary key is loanappno — the customer key is Cust ID
6. Bounce means Bounce status IN ('Tech', 'Non Tech') — NEVER include PAID
7. Resolved / Norm means Cust wise status = 'Norm'
8. Op bucket = opening/start-of-month risk bucket — this is the PRIMARY bucket field
9. When unsure between customer-level and loan-level, default to customer-level

AVAILABLE METRICS (use these exact metric_key values):
- Bounce_Count         : Count of bounced customers
- Bounce_Percent       : Bounce rate as %
- Resolution           : Count of Norm customers
- Resolution_Percent   : Resolution rate as %
- Coverage             : Count of visited allocated customers
- Coverage_Percent     : Coverage rate as %
- Intensity            : Average visits per customer
- Portfolio_Outstanding: SUM of bal_prin
- Total_Overdue        : SUM of TOD
- Bucket_Distribution  : Count/outstanding by Op bucket
- Bucket_Movement      : Opening vs Closing bucket matrix
- NPA_Count            : Count of NPA accounts
- Add_NPA_Count        : Count of new NPA additions (Add NPA = 1)
- Payment_Analysis     : Payment mode / Digital vs Cash breakdown
- FE_Scorecard         : Field executive performance (Resolution + Coverage + Intensity)
- DPD_Distribution     : Distribution of dpd_casewise values
- Custom               : Use when none of the above match exactly

AGGREGATION OPTIONS:
- DISTINCTCOUNT : Customer-level counts (Resolution, Bounce, Coverage)
- COUNT         : Loan-level counts (NPA accounts, Add NPA)
- SUM           : Financial amounts (bal_prin, TOD, Bounce charges)
- AVERAGE       : Averages (Intensity, average DPD)
- DIVIDE        : Percentages (always SUM or COUNT in numerator/denominator)

GRANULARITY:
- customer : Use when question is about customers (resolution, bounce, coverage)
- loan     : Use when question is about loan accounts/applications (NPA accounts, DPD)

RELEVANT CONTEXT FROM DATA DICTIONARY:
{context}

Return ONLY valid JSON — no explanation, no markdown, no extra text:
{{
  "intent_summary":    "one sentence of what the user wants to see",
  "metric":            "human readable metric name e.g. Bounce Rate",
  "metric_key":        "exact key from AVAILABLE METRICS list above",
  "columns_needed":    ["list", "of", "exact", "column", "names", "from", "data", "dictionary"],
  "slicer_candidates": ["columns", "user", "might", "want", "to", "filter", "by"],
  "aggregation":       "DISTINCTCOUNT | COUNT | SUM | AVERAGE | DIVIDE",
  "group_by":          "column to group results by for the chart, or null",
  "time_involved":     true or false,
  "granularity":       "customer or loan",
  "dax_pattern_hint":  "closest key from dax_patterns e.g. bounce_by_branch, or null",
  "confidence":        "High | Medium | Low"
}}
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# * ANALYZER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class Analyzer:
    """
    # * Step 1 of the pipeline — analyzes user question and returns IntentResult.
    # * Uses ChromaDB to fetch relevant context then calls Qwen Coder 14B.

    # ? Call flow:
    # ?   1. Fetch relevant context chunks from ChromaDB for the question
    # ?   2. Build system prompt with context injected
    # ?   3. Call Qwen Coder 14B via OllamaClient
    # ?   4. Parse JSON response into IntentResult
    # ?   5. Validate and clean the result
    # ?   6. Return IntentResult to the pipeline orchestrator
    """

    def __init__(self):
        self._client     = get_client()
        self._dictionary = get_dictionary()
        log.info("[analyzer] Initialised")

    # ── Main entry point ──────────────────────────────────────────────────────

    def analyze(self, question: str) -> IntentResult:
        """
        # * Analyze a user question and return a structured IntentResult.
        # * This is the only public method — call this from the pipeline.

        Args:
            question : Raw user question string (e.g. "show bounced loans last month")

        Returns:
            IntentResult with success=True and all fields populated,
            or IntentResult with success=False and error message.

        Example:
            analyzer = get_analyzer()
            result   = analyzer.analyze("how many loans bounced in Pune last month?")

            if result.success:
                print(result.metric)          # "Bounce Rate"
                print(result.metric_key)      # "Bounce_Percent"
                print(result.columns_needed)  # ["Bounce status", "Cust ID", "Branch"]
                print(result.slicer_candidates)  # ["Branch", "Region", "Next Date"]
        """
        if not question or not question.strip():
            return IntentResult(
                success=False,
                error="Empty question received"
            )

        question = question.strip()
        log.info(f"[analyzer] Analyzing question | '{truncate_string(question, 80)}'")

        with benchmark("intent_analysis", query=question):

            # * Step 1 — Fetch relevant context from ChromaDB
            context = self._fetch_context(question)

            # * Step 2 — Build prompt with context
            system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(context=context)
            user_message  = f'User question: "{question}"\n\nAnalyze and return JSON now.'

            # * Log prompt size
            total_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_message)
            model_log.debug(
                f"[intent_analysis] Prompt built | "
                f"context_len={len(context)} chars | "
                f"~{total_tokens} tokens"
            )

            # * Step 3 — Call Qwen Coder
            response: ModelResponse = self._client.call_coder(
                system      = system_prompt,
                user        = user_message,
                step        = "intent_analysis",
                expect_json = True,
            )

            # * Step 4 — Handle failure
            if response.failed:
                return IntentResult(
                    success      = False,
                    raw_question = question,
                    error        = response.error or "Model call failed"
                )

            # * Step 5 — Parse and validate
            return self._parse_response(response, question)

    # ── Context fetching ──────────────────────────────────────────────────────

    def _fetch_context(self, question: str) -> str:
        """
        # * Fetch relevant context from ChromaDB for the given question.
        # * Returns formatted context string for prompt injection.
        # * Falls back to a brief hardcoded summary if ChromaDB is not ready.

        Args:
            question : User question

        Returns:
            Multi-section context string.
        """
        if not self._dictionary.is_ready:
            log.warning("[analyzer] Dictionary not ready — using fallback context")
            return _FALLBACK_CONTEXT

        context = self._dictionary.get_coder_context(question)

        if not context:
            log.warning("[analyzer] ChromaDB returned empty context — using fallback")
            return _FALLBACK_CONTEXT

        return context

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(
        self,
        response: ModelResponse,
        question: str,
    ) -> IntentResult:
        """
        # * Parse Qwen's JSON response into a validated IntentResult.
        # * Applies cleaning and defaults for any missing fields.

        Args:
            response : ModelResponse from OllamaClient
            question : Original user question

        Returns:
            IntentResult — always returns one even if parsing is partial.
        """
        parsed = response.parsed

        # ! If JSON parsing failed entirely, try to extract useful info from raw text
        if parsed is None:
            log.warning(
                f"[analyzer] JSON parse failed — attempting raw text fallback | "
                f"raw='{truncate_string(response.content, 100)}'"
            )
            return self._fallback_parse(response.content, question)

        if not isinstance(parsed, dict):
            log.error(f"[analyzer] Parsed result is not a dict | type={type(parsed)}")
            return IntentResult(
                success      = False,
                raw_question = question,
                error        = "Model returned non-dict JSON"
            )

        # * Extract fields with safe defaults
        intent_summary   = str(parsed.get("intent_summary",   "")).strip()
        metric           = str(parsed.get("metric",           "")).strip()
        metric_key       = str(parsed.get("metric_key",       "Custom")).strip()
        columns_needed   = self._clean_list(parsed.get("columns_needed",    []))
        slicer_candidates= self._clean_list(parsed.get("slicer_candidates", []))
        aggregation      = str(parsed.get("aggregation",      "COUNT")).upper().strip()
        group_by         = parsed.get("group_by")
        time_involved    = bool(parsed.get("time_involved",   False))
        granularity      = str(parsed.get("granularity",      "customer")).lower().strip()
        dax_hint         = parsed.get("dax_pattern_hint")
        confidence       = str(parsed.get("confidence",       "Medium")).strip()

        # * Validate aggregation value
        valid_aggs = {"DISTINCTCOUNT", "COUNT", "SUM", "AVERAGE", "DIVIDE"}
        if aggregation not in valid_aggs:
            log.warning(f"[analyzer] Invalid aggregation '{aggregation}' — defaulting to COUNT")
            aggregation = "COUNT"

        # * Validate granularity
        if granularity not in ("customer", "loan"):
            granularity = "customer"

        # * Validate group_by is a string or None
        if group_by and not isinstance(group_by, str):
            group_by = str(group_by)
        if group_by and group_by.lower() in ("null", "none", ""):
            group_by = None

        # * Validate dax_hint
        if dax_hint and not isinstance(dax_hint, str):
            dax_hint = str(dax_hint)
        if dax_hint and dax_hint.lower() in ("null", "none", ""):
            dax_hint = None

        result = IntentResult(
            success           = True,
            raw_question      = question,
            intent_summary    = intent_summary,
            metric            = metric,
            metric_key        = metric_key,
            columns_needed    = columns_needed,
            slicer_candidates = slicer_candidates,
            aggregation       = aggregation,
            group_by          = group_by,
            time_involved     = time_involved,
            granularity       = granularity,
            dax_pattern_hint  = dax_hint,
            confidence        = confidence,
        )

        model_log.info(
            f"[intent_analysis] PARSED OK | "
            f"metric='{metric}' | "
            f"metric_key='{metric_key}' | "
            f"columns={columns_needed} | "
            f"confidence={confidence}"
        )

        return result

    def _clean_list(self, raw: any) -> list[str]:
        """
        # * Safely convert a raw value to a clean list of strings.
        # * Handles None, strings, and lists from the model output.

        Args:
            raw : Any value from parsed JSON

        Returns:
            List of non-empty strings.
        """
        if raw is None:
            return []
        if isinstance(raw, str):
            # * Model sometimes returns comma-separated string instead of list
            return [s.strip() for s in raw.split(",") if s.strip()]
        if isinstance(raw, list):
            return [str(s).strip() for s in raw if str(s).strip()]
        return []

    def _fallback_parse(self, raw_text: str, question: str) -> IntentResult:
        """
        # * Last resort when JSON parsing completely fails.
        # * Tries to extract intent from raw model text using keywords.
        # * Returns a low-confidence result so the pipeline can still proceed.

        # ? This handles edge cases where Qwen adds explanation text
        # ? around the JSON or returns a malformed response.

        Args:
            raw_text : Raw model response string
            question : Original user question

        Returns:
            IntentResult with confidence=Low and best-guess fields.
        """
        log.warning("[analyzer] Using fallback parse — model output was not clean JSON")

        raw_lower = raw_text.lower()
        question_lower = question.lower()

        # * Keyword-based metric detection as fallback
        if any(w in raw_lower or w in question_lower
               for w in ["bounce", "bounced", "payment failure", "nach"]):
            metric = "Bounce Rate"
            metric_key = "Bounce_Percent"
            columns = ["Bounce status", "Cust ID"]

        elif any(w in raw_lower or w in question_lower
                 for w in ["resolution", "resolved", "norm", "recovered"]):
            metric = "Resolution %"
            metric_key = "Resolution_Percent"
            columns = ["Cust wise status", "Cust ID"]

        elif any(w in raw_lower or w in question_lower
                 for w in ["coverage", "visited", "visit", "field visit"]):
            metric = "Coverage %"
            metric_key = "Coverage_Percent"
            columns = ["Visit or not", "Allocated or not", "Cust ID"]

        elif any(w in raw_lower or w in question_lower
                 for w in ["npa", "non performing", "write off", "written off"]):
            metric = "NPA Count"
            metric_key = "NPA_Count"
            columns = ["Op bucket", "loanappno"]

        elif any(w in raw_lower or w in question_lower
                 for w in ["outstanding", "principal", "bal_prin", "portfolio"]):
            metric = "Portfolio Outstanding"
            metric_key = "Portfolio_Outstanding"
            columns = ["bal_prin", "Op bucket"]

        else:
            metric = "Custom"
            metric_key = "Custom"
            columns = ["loanappno"]

        model_log.warning(
            f"[intent_analysis] FALLBACK RESULT | "
            f"metric='{metric}' | confidence=Low"
        )

        return IntentResult(
            success           = True,
            raw_question      = question,
            intent_summary    = f"User wants to see {metric} data",
            metric            = metric,
            metric_key        = metric_key,
            columns_needed    = columns,
            slicer_candidates = ["Region", "Branch", "Op bucket"],
            aggregation       = "DISTINCTCOUNT",
            granularity       = "customer",
            confidence        = "Low",
        )


# ──────────────────────────────────────────────────────────────────────────────
# * FALLBACK CONTEXT
# * Used when ChromaDB is not initialised yet.
# * Bare minimum context to keep the model from hallucinating column names.
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_CONTEXT = """
KEY FIELDS:
- loanappno      : Loan Application Number (primary key)
- Cust ID        : Customer ID (use DISTINCTCOUNT for customer metrics)
- Op bucket      : Risk bucket — Current, Risk X, 1-29 DPD, 30-59 DPD, 60-89 DPD, NPA, Write-off
- Bounce status  : PAID | Tech | Non Tech (Tech + Non Tech = bounce)
- Cust wise status: Current | Norm | Flow | Stab | Roll Back | Risk NPA | NPA | Write-off
- bal_prin       : Balance Principal outstanding
- TOD            : Total Overdue amount
- Visit or not   : Visited | Not Visited
- Allocated or not: Allocated | Non Allocated
- Region         : Geographic region (slicer)
- Branch         : Service branch (slicer)
- Portfolio new  : Portfolio group (slicer)
- TL             : Team Leader (slicer)

KEY METRICS:
- Bounce %       : DISTINCTCOUNT(Cust ID) WHERE Bounce status IN (Tech, Non Tech) / Total
- Resolution %   : DISTINCTCOUNT(Cust ID) WHERE Cust wise status = Norm / Total
- Coverage %     : DISTINCTCOUNT(Cust ID) WHERE Visited AND Allocated / Total Allocated
- Intensity      : SUM(Visit Count) / DISTINCTCOUNT(Cust ID)
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# * SINGLETON ACCESSOR
# ──────────────────────────────────────────────────────────────────────────────

_analyzer_instance: Optional[Analyzer] = None


def get_analyzer() -> Analyzer:
    """
    # * Returns the shared Analyzer instance.
    # * Creates it on first call — reused on every subsequent call.

    Usage:
        from models.analyzer import get_analyzer
        analyzer = get_analyzer()
        result   = analyzer.analyze("show bounced loans by branch")

        if result.success:
            print(result.metric_key)         # Bounce_Percent
            print(result.slicer_candidates)  # ["Branch", "Region", "Next Date"]
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = Analyzer()
    return _analyzer_instance