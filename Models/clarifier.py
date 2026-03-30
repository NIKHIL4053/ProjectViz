"""
models/clarifier.py
-------------------
# * Pipeline Step 2 — Clarifying Questions.
# * Takes the IntentResult from analyzer.py and generates 2-4 targeted
# * clarifying questions to collect the slicer values the user wants.

# ? Why ask clarifying questions instead of going straight to DAX?
# ? "Show me bounced loans" is ambiguous — which branch? which month?
# ? Asking 2-4 targeted questions makes the DAX precise and the chart meaningful.
# ? Questions are generated DYNAMICALLY based on what slicers are relevant
# ? to the specific question — not a fixed set of dropdowns every time.

# ? Which model: Qwen Coder 14B
# ? Why: Needs domain awareness to generate sensible questions.
# ?      "Which team leader?" only makes sense for coverage/intensity questions.
# ?      "Which portfolio?" only makes sense for portfolio-level analysis.
# ?      The model decides which questions are relevant based on context.

# ? User experience:
# ?   Questions are shown as Streamlit selectbox/multiselect widgets.
# ?   User picks answers → answers stored as slicer dict → passed to DAX generator.
# ?   If user selects "All" for any question → no filter applied for that slicer.

Exports:
    - ClarifyingQuestion   : Dataclass for one question + its options
    - ClarifierResult      : Dataclass wrapping all generated questions
    - Clarifier            : Main class
    - get_clarifier()      : Singleton accessor
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import streamlit as st

from config import CODER_MODEL
from core.dictionary import get_dictionary
from models.analyzer import IntentResult
from models.ollama_client import get_client, ModelResponse
from utils.logger import get_logger, get_model_logger
from utils.benchmark import benchmark
from utils.helpers import truncate_string, estimate_tokens, parse_json_safely

# * Module level loggers
log       = get_logger(__name__)
model_log = get_model_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClarifyingQuestion:
    """
    # * Represents one clarifying question shown to the user.

    Attributes:
        question      : Human-readable question text shown in the UI
        slicer_field  : Exact column name this answer maps to (e.g. "Branch")
        options       : List of selectable options including "All"
        widget_type   : "selectbox" or "multiselect"
        default       : Default selection (always "All")
        category      : Optional grouping label for the UI
    """
    question:     str
    slicer_field: str
    options:      list[str]              = field(default_factory=list)
    widget_type:  str                    = "selectbox"
    default:      str                    = "All"
    category:     str                    = ""


@dataclass
class ClarifierResult:
    """
    # * Output from the clarifier step.

    Attributes:
        success    : True if questions were generated successfully
        questions  : List of ClarifyingQuestion objects
        answers    : Dict of {slicer_field: selected_value} after user answers
                     Populated by collect_answers() — empty until user interacts
        error      : Error message if success=False
    """
    success:   bool
    questions: list[ClarifyingQuestion]  = field(default_factory=list)
    answers:   dict                      = field(default_factory=dict)
    error:     Optional[str]             = None

    @property
    def has_active_filters(self) -> bool:
        """# * True if user selected at least one non-All answer."""
        return any(
            v and str(v).lower() not in ("all", "")
            for v in self.answers.values()
        )

    @property
    def active_filters(self) -> dict:
        """# * Returns only the answers where user picked something specific (not All)."""
        return {
            k: v for k, v in self.answers.items()
            if v and str(v).lower() not in ("all", "")
        }

    @property
    def failed(self) -> bool:
        return not self.success


# ──────────────────────────────────────────────────────────────────────────────
# * SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """
You are a data analyst assistant for a loan collection portfolio system.
A user has asked a question and you need to generate 2-4 clarifying questions
to collect the filter/slicer values needed to answer their question precisely.

RULES FOR GENERATING QUESTIONS:
1. ONLY ask about slicers that are RELEVANT to this specific question
   - Bounce question     → ask about Branch, Region, Month, Portfolio
   - Coverage question   → ask about TL, SH, Branch, Month
   - NPA question        → ask about Region, Portfolio, Op bucket, Month
   - Resolution question → ask about Branch, Region, Portfolio, Month
   - Never ask about ALL possible slicers — only the relevant ones
2. Always include a time period question if the metric is time-sensitive
3. ALWAYS include "All" as the first option — user can skip any filter
4. Keep question language simple — user is NON-TECHNICAL
5. Do not ask about loanappno, Cust ID, or other identifier fields
6. Maximum 4 questions — minimum 2 questions
7. For Op bucket questions, always list all bucket values in order

SLICER FIELD NAMES (use EXACTLY these — case sensitive):
Geography    : "Region", "Branch"
Bucket/Status: "Op bucket", "Cust wise status", "Bounce status"
Portfolio    : "Portfolio new", "Product", "Type of Arrangement"
Collection   : "TL", "SH", "Coll/Sales", "Allocated or not"
Time         : "Next Date" (use month options like "Jan 2026", "Feb 2026")
Loan         : "MOB Bucket"
Payment      : "Digital/cash", "Payment mode", "Payment Day bucket"
Risk         : "Add NPA"

WIDGET TYPE:
- Use "selectbox"   for single-value fields (Branch, Region, Portfolio, Month)
- Use "multiselect" for fields where multiple values make sense (Op bucket, Bounce status)

RELEVANT CONTEXT:
{context}

QUESTION BEING ANALYZED:
Intent: {intent_summary}
Metric: {metric}
Slicer candidates identified: {slicer_candidates}
Time involved: {time_involved}

Return ONLY a valid JSON array — no explanation, no markdown:
[
  {{
    "question":     "Which region would you like to see? Or show all regions?",
    "slicer_field": "Region",
    "options":      ["All", "Pune", "NCR", "Rajasthan", "MP", "Karnataka", "Telangana"],
    "widget_type":  "selectbox",
    "category":     "Geography"
  }}
]
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# * KNOWN OPTION SETS
# * Pre-defined options for fields with fixed value sets.
# * Used to populate question options without scanning data.
# ──────────────────────────────────────────────────────────────────────────────

_KNOWN_OPTIONS: dict[str, list[str]] = {
    "Op bucket":        ["All", "Current", "Risk X", "1-29 DPD", "30-59 DPD", "60-89 DPD", "NPA", "Write-off"],
    "Cust wise status": ["All", "Current", "Norm", "Flow", "Stab", "Roll Back", "Risk NPA", "NPA", "Write-off"],
    "Bounce status":    ["All", "Tech", "Non Tech", "PAID"],
    "Digital/cash":     ["All", "Digital", "Cash"],
    "Allocated or not": ["All", "Allocated", "Non Allocated"],
    "Visit or not":     ["All", "Visited", "Not Visited"],
    "Coll/Sales":       ["All", "Collection", "Sales"],
    "Add NPA":          ["All", "1 (New NPA)", "0 (Not NPA)"],
    "MOB Bucket":       ["All", "1-6 MOB", "7-12 MOB", "13-18 MOB", "19-24 MOB", "24+ MOB"],
    "Payment mode":     ["All", "Cash", "Cheque", "Demand Draft", "NEFT", "RTGS", "Blank"],
    "Payment Day bucket": ["All", "1-5", "6-10", "11-15", "16-20", "21-25", "26-27", "28-31", "NA"],
}

# * Default widget types per field
_WIDGET_TYPES: dict[str, str] = {
    "Op bucket":     "multiselect",
    "Bounce status": "multiselect",
    "Cust wise status": "multiselect",
}


# ──────────────────────────────────────────────────────────────────────────────
# * CLARIFIER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class Clarifier:
    """
    # * Step 2 of the pipeline — generates clarifying questions from IntentResult.
    # ? Two modes of operation:
    # ? Mode A — With PostgreSQL data available:
    # ?   Options are dynamically populated from actual column values in the data.
    # ?   e.g. Branch options = actual branches in the returned dataset.
    # ? Mode B — Without data (first question, before any SQL query runs):
    # ?   Options come from _KNOWN_OPTIONS dict or are left generic.
    # ?   Qwen generates the options based on domain knowledge.
    """

    def __init__(self):
        self._client     = get_client()
        self._dictionary = get_dictionary()
        log.info("[clarifier] Initialised")

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate(
        self,
        intent:    IntentResult,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> ClarifierResult:
        """
        # * Generate clarifying questions based on the IntentResult.
        # * Optionally enriches options from a reference DataFrame.

        Args:
            intent       : IntentResult from analyzer.py
            reference_df : Optional DataFrame to extract real option values from.
                           If None, uses _KNOWN_OPTIONS and generic values.

        Returns:
            ClarifierResult with questions populated.
            Call collect_answers() next to render widgets and get user input.

        Example:
            clarifier = get_clarifier()
            result    = clarifier.generate(intent_result)
            answers   = clarifier.collect_answers(result)
        """
        if intent.failed:
            return ClarifierResult(
                success=False,
                error="Cannot generate questions — intent analysis failed"
            )

        log.info(
            f"[clarifier] Generating questions | "
            f"metric='{intent.metric}' | "
            f"slicer_candidates={intent.slicer_candidates}"
        )

        with benchmark("clarifying_questions", query=intent.raw_question):

            # * Step 1 — Fetch relevant context
            context = self._fetch_context(intent)

            # * Step 2 — Build prompt
            system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                context           = context,
                intent_summary    = intent.intent_summary,
                metric            = intent.metric,
                slicer_candidates = intent.slicer_candidates,
                time_involved     = intent.time_involved,
            )
            user_message = (
                f'Original question: "{intent.raw_question}"\n\n'
                f"Generate the clarifying questions JSON array now."
            )

            model_log.debug(
                f"[clarifying_questions] Prompt built | "
                f"~{estimate_tokens(system_prompt)} tokens"
            )

            # * Step 3 — Call Qwen Coder
            response: ModelResponse = self._client.call_coder(
                system      = system_prompt,
                user        = user_message,
                step        = "clarifying_questions",
                expect_json = True,
            )

            if response.failed:
                # ! Model call failed — fall back to rule-based questions
                log.warning(
                    "[clarifier] Model call failed — using rule-based fallback | "
                    f"error={response.error}"
                )
                questions = self._rule_based_questions(intent)
            else:
                questions = self._parse_questions(response, intent)

            # * Step 4 — Enrich options from reference data if available
            if reference_df is not None and not reference_df.empty:
                questions = self._enrich_from_data(questions, reference_df)
            else:
                questions = self._enrich_from_known_options(questions)

            log.info(f"[clarifier] Generated {len(questions)} questions")
            return ClarifierResult(success=True, questions=questions)

    # ── Render widgets and collect answers ────────────────────────────────────

    def collect_answers(
        self,
        result:     ClarifierResult,
        session_key_prefix: str = "clarifier",
    ) -> dict:
        """
        # * Render clarifying questions as Streamlit widgets and return answers.
        # * Must be called inside a Streamlit context (app is running).

        # ? Each widget gets a stable session_state key so selections persist
        # ? across reruns without resetting to default.

        Args:
            result             : ClarifierResult from generate()
            session_key_prefix : Prefix for Streamlit widget keys

        Returns:
            Dict of {slicer_field: selected_value}
            "All" values are included — DAX generator filters them out.

        Example:
            answers = clarifier.collect_answers(result)
            # → {"Region": "All", "Branch": "Pune", "Next Date": "Feb 2026"}
        """
        if not result.success or not result.questions:
            return {}

        answers = {}

        st.markdown("#### 🔍 Help me narrow this down:")

        # * Group questions by category
        categories: dict[str, list[ClarifyingQuestion]] = {}
        for q in result.questions:
            cat = q.category or "General"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(q)

        # * Render each question
        for cat, questions in categories.items():
            if len(categories) > 1:
                st.markdown(f"**{cat}**")

            cols = st.columns(min(len(questions), 2))

            for i, q in enumerate(questions):
                col         = cols[i % 2]
                widget_key  = f"{session_key_prefix}_{q.slicer_field.lower().replace(' ', '_')}"
                options     = q.options if q.options else ["All"]

                # * Ensure "All" is always first
                if "All" not in options:
                    options = ["All"] + options

                with col:
                    if q.widget_type == "multiselect":
                        selected = st.multiselect(
                            label   = q.question,
                            options = options[1:],  # * exclude "All" from multiselect options
                            default = [],
                            key     = widget_key,
                            help    = f"Leave empty to include all {q.slicer_field} values"
                        )
                        # * Empty multiselect = "All"
                        answers[q.slicer_field] = selected if selected else "All"

                    else:
                        selected = st.selectbox(
                            label   = q.question,
                            options = options,
                            index   = 0,
                            key     = widget_key,
                        )
                        answers[q.slicer_field] = selected

        result.answers = answers
        active = {k: v for k, v in answers.items()
                  if v and str(v).lower() != "all"}

        log.info(
            f"[clarifier] Answers collected | "
            f"total={len(answers)} | "
            f"active={len(active)} | "
            f"active_filters={active}"
        )
        return answers

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_questions(
        self,
        response: ModelResponse,
        intent:   IntentResult,
    ) -> list[ClarifyingQuestion]:
        """
        # * Parse Qwen's JSON array into ClarifyingQuestion objects.
        # * Falls back to rule-based if parsing fails.

        Args:
            response : ModelResponse from OllamaClient
            intent   : IntentResult for context in fallback

        Returns:
            List of ClarifyingQuestion objects.
        """
        parsed = response.parsed

        # * parsed should be a list from the model
        if parsed is None:
            log.warning("[clarifier] JSON parse failed — using rule-based fallback")
            return self._rule_based_questions(intent)

        if not isinstance(parsed, list):
            # * Sometimes model wraps the array in an object
            if isinstance(parsed, dict):
                for key in ("questions", "clarifying_questions", "items"):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
            if not isinstance(parsed, list):
                log.warning("[clarifier] Model returned non-list JSON — using fallback")
                return self._rule_based_questions(intent)

        questions = []
        for item in parsed[:4]:  # * Max 4 questions
            if not isinstance(item, dict):
                continue
            try:
                q = ClarifyingQuestion(
                    question     = str(item.get("question",     "")).strip(),
                    slicer_field = str(item.get("slicer_field", "")).strip(),
                    options      = self._clean_options(item.get("options", [])),
                    widget_type  = str(item.get("widget_type",  "selectbox")).strip(),
                    category     = str(item.get("category",     "")).strip(),
                )
                # * Skip if essential fields are missing
                if not q.question or not q.slicer_field:
                    log.warning(f"[clarifier] Skipping question — missing question or slicer_field | {item}")
                    continue

                # * Override widget type if we have a known preference
                if q.slicer_field in _WIDGET_TYPES:
                    q.widget_type = _WIDGET_TYPES[q.slicer_field]

                questions.append(q)

            except Exception as e:
                log.error(f"[clarifier] Failed to parse question item | error={e} | item={item}")
                continue

        if not questions:
            log.warning("[clarifier] No valid questions parsed — using rule-based fallback")
            return self._rule_based_questions(intent)

        model_log.info(
            f"[clarifying_questions] PARSED OK | "
            f"count={len(questions)} | "
            f"fields={[q.slicer_field for q in questions]}"
        )
        return questions

    def _clean_options(self, raw: any) -> list[str]:
        """# * Convert raw options value to a clean list of strings."""
        if raw is None:
            return ["All"]
        if isinstance(raw, list):
            cleaned = [str(v).strip() for v in raw if str(v).strip()]
            if "All" not in cleaned:
                cleaned = ["All"] + cleaned
            return cleaned
        return ["All"]

    # ── Option enrichment ─────────────────────────────────────────────────────

    def _enrich_from_data(
        self,
        questions:    list[ClarifyingQuestion],
        reference_df: pd.DataFrame,
    ) -> list[ClarifyingQuestion]:
        """
        # * Replace generic options with actual values from the reference DataFrame.
        # * e.g. Branch options become the actual branch names in the data.
        # * Fields with _KNOWN_OPTIONS are not overridden — they have fixed value sets.

        Args:
            questions    : List of ClarifyingQuestion from parsing
            reference_df : DataFrame with actual data values

        Returns:
            Questions with enriched options.
        """
        for q in questions:
            # * Skip if this field has known fixed values
            if q.slicer_field in _KNOWN_OPTIONS:
                continue

            # * Skip if field not in DataFrame
            if q.slicer_field not in reference_df.columns:
                continue

            try:
                unique_vals = (
                    reference_df[q.slicer_field]
                    .dropna()
                    .unique()
                    .tolist()
                )
                # * Only replace if we got meaningful values (2 or more)
                if len(unique_vals) >= 2:
                    sorted_vals = sorted([str(v) for v in unique_vals])
                    q.options   = ["All"] + sorted_vals
                    log.debug(
                        f"[clarifier] Enriched options for '{q.slicer_field}' | "
                        f"{len(sorted_vals)} values from data"
                    )
            except Exception as e:
                log.error(f"[clarifier] Option enrichment failed for '{q.slicer_field}' | error={e}")

        return questions

    def _enrich_from_known_options(
        self,
        questions: list[ClarifyingQuestion],
    ) -> list[ClarifyingQuestion]:
        """
        # * Replace empty or generic options with known option sets.
        # * Used when no reference DataFrame is available.

        Args:
            questions : List of ClarifyingQuestion from parsing

        Returns:
            Questions with known options applied.
        """
        for q in questions:
            if q.slicer_field in _KNOWN_OPTIONS:
                q.options = _KNOWN_OPTIONS[q.slicer_field]
            elif not q.options or q.options == ["All"]:
                # * Generic fallback for unknown fields
                q.options = ["All"]

        return questions

    # ── Context fetch ─────────────────────────────────────────────────────────

    def _fetch_context(self, intent: IntentResult) -> str:
        """
        # * Fetch relevant context from ChromaDB focused on slicer/filter knowledge.

        Args:
            intent : IntentResult from analyzer

        Returns:
            Context string for prompt injection.
        """
        if not self._dictionary.is_ready:
            return ""

        # * Build a targeted query combining metric + slicer candidates
        query = (
            f"{intent.raw_question} "
            f"{intent.metric} "
            f"{' '.join(intent.slicer_candidates[:5])}"
        )

        # * Focus on language mapping and schema — most relevant for question generation
        lang_chunks   = self._dictionary.search(query, top_k=4, source="language_mapping")
        schema_chunks = self._dictionary.search(query, top_k=3, source="schema_context")

        sections = []
        if lang_chunks:
            sections.append("SLICER CONTEXT:\n" + "\n\n".join(lang_chunks))
        if schema_chunks:
            sections.append("FIELD CONTEXT:\n" + "\n\n".join(schema_chunks))

        return "\n\n".join(sections)

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_based_questions(self, intent: IntentResult) -> list[ClarifyingQuestion]:
        """
        # * Generate questions using rules instead of Qwen.
        # * Used when the model call fails or JSON parsing fails entirely.
        # * Produces sensible questions based on the metric_key.

        Args:
            intent : IntentResult from analyzer

        Returns:
            List of ClarifyingQuestion objects — always returns at least 2.
        """
        log.info(f"[clarifier] Rule-based fallback | metric_key={intent.metric_key}")

        questions = []

        # * Geography questions — almost always relevant
        questions.append(ClarifyingQuestion(
            question     = "Which region would you like to see? Or show all regions?",
            slicer_field = "Region",
            options      = ["All", "Pune", "NCR", "Rajasthan", "MP", "Karnataka", "Telangana"],
            widget_type  = "selectbox",
            category     = "📍 Geography",
        ))
        questions.append(ClarifyingQuestion(
            question     = "Any specific branch? Or show all branches?",
            slicer_field = "Branch",
            options      = ["All"],
            widget_type  = "selectbox",
            category     = "📍 Geography",
        ))

        # * Metric-specific additional questions
        metric = intent.metric_key

        if metric in ("Bounce_Count", "Bounce_Percent"):
            questions.append(ClarifyingQuestion(
                question     = "Which type of bounce? Tech, Non Tech, or both?",
                slicer_field = "Bounce status",
                options      = _KNOWN_OPTIONS["Bounce status"],
                widget_type  = "multiselect",
                category     = "💳 Payment",
            ))

        elif metric in ("Coverage_Percent", "Coverage"):
            questions.append(ClarifyingQuestion(
                question     = "Any specific Team Leader? Or show all TLs?",
                slicer_field = "TL",
                options      = ["All"],
                widget_type  = "selectbox",
                category     = "👥 Collection Team",
            ))

        elif metric in ("Bucket_Distribution", "Bucket_Movement"):
            questions.append(ClarifyingQuestion(
                question     = "Which opening bucket(s)? Or show all?",
                slicer_field = "Op bucket",
                options      = _KNOWN_OPTIONS["Op bucket"],
                widget_type  = "multiselect",
                category     = "📊 Bucket",
            ))

        elif metric in ("NPA_Count", "Add_NPA_Count"):
            questions.append(ClarifyingQuestion(
                question     = "Any specific portfolio? Or show all portfolios?",
                slicer_field = "Portfolio new",
                options      = ["All", "VASTU", "State Bank of India DA 1-3", "Tata Capital DA01", "Other"],
                widget_type  = "selectbox",
                category     = "🗂️ Portfolio",
            ))

        elif metric in ("Resolution_Percent", "Resolution"):
            questions.append(ClarifyingQuestion(
                question     = "Any specific portfolio? Or show all?",
                slicer_field = "Portfolio new",
                options      = ["All", "VASTU", "State Bank of India DA 1-3", "Tata Capital DA01", "Other"],
                widget_type  = "selectbox",
                category     = "🗂️ Portfolio",
            ))

        # * Time question — add if time is involved
        if intent.time_involved:
            questions.append(ClarifyingQuestion(
                question     = "Which month are you looking at?",
                slicer_field = "Next Date",
                options      = ["All", "Feb 2026", "Jan 2026", "Dec 2025", "Nov 2025", "Oct 2025"],
                widget_type  = "selectbox",
                category     = "📅 Time Period",
            ))

        return questions[:4]  # * Never exceed 4 questions


# ──────────────────────────────────────────────────────────────────────────────
# * SINGLETON ACCESSOR
# ──────────────────────────────────────────────────────────────────────────────

_clarifier_instance: Optional[Clarifier] = None


def get_clarifier() -> Clarifier:
    """
    # * Returns the shared Clarifier instance.
    # * Creates it on first call — reused on every subsequent call.

    Usage:
        from models.clarifier import get_clarifier
        clarifier = get_clarifier()
        result    = clarifier.generate(intent_result)
        answers   = clarifier.collect_answers(result)
        # answers → {"Region": "Pune", "Branch": "All", "Next Date": "Feb 2026"}
    """
    global _clarifier_instance
    if _clarifier_instance is None:
        _clarifier_instance = Clarifier()
    return _clarifier_instance