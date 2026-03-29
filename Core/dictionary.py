"""
core/dictionary.py
------------------
# * Manages the data dictionary for the Loan Dashboard project.
# * Loads all 5 context JSONs into ChromaDB on first run.
# * Fetches semantically relevant context chunks per user query.
# * Provides formatted context strings ready for prompt injection.

# ? Why ChromaDB instead of just dumping all JSONs into every prompt?
# ? The 5 context JSONs combined are ~15,000 tokens.
# ? Qwen 14B context window is 128K but we want to keep prompts lean and fast.
# ? ChromaDB lets us fetch only the TOP-K most relevant chunks per question.
# ? e.g. "show bounce rate" → only bounce-related fields + metric formulas fetched
# ? e.g. "coverage by TL"  → only collection team fields + coverage metric fetched

# ! IMPORTANT: Run this module once before starting the app.
# ! If ChromaDB collection is empty, context injection will be empty and Qwen will hallucinate.

Exports:
    - DataDictionary          : Main class — load, search, retrieve context
    - get_dictionary()        : Singleton accessor — returns shared instance
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMADB_PATH,
    CHROMA_COLLECTION,
    CHROMA_TOP_K,
    EMBEDDING_MODEL,
    SCHEMA_CONTEXT_PATH,
    BUSINESS_LOGIC_PATH,
    LANGUAGE_MAPPING_PATH,
    VISUALIZATION_PATH,
    DAX_PATTERNS_PATH,
)
from utils.logger import get_logger
from utils.benchmark import benchmark
from utils.helpers import chunk_list, estimate_tokens, truncate_string

# * Module level logger
log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# * DOCUMENT BUILDERS
# * Each function reads one context JSON and converts it into a list of
# * plain-text "documents" that ChromaDB will embed and store.
# * Each document = one searchable chunk with a unique ID and metadata.
# ──────────────────────────────────────────────────────────────────────────────

def _build_schema_documents(data: dict) -> list[dict]:
    """
    # * Converts 01_schema_context.json into embeddable documents.
    # * One document per field — each includes field name, type, description,
    # * consumer language, DAX hints, and slicer status.

    Args:
        data : Parsed JSON from 01_schema_context.json

    Returns:
        List of {"id": str, "text": str, "metadata": dict}
    """
    docs = []
    for field_def in data.get("fields", []):
        field_name = field_def.get("field", "unknown")

        # * Build a rich text block for each field — more context = better retrieval
        parts = [
            f"FIELD: {field_name}",
            f"TYPE: {field_def.get('data_type', '')}",
            f"CATEGORY: {field_def.get('category', '')}",
            f"SLICER: {'Yes' if field_def.get('slicer') else 'No'}",
            f"DESCRIPTION: {field_def.get('description', '')}",
        ]

        if field_def.get("values"):
            parts.append(f"VALID VALUES: {', '.join(str(v) for v in field_def['values'])}")

        if field_def.get("use_in_dax"):
            parts.append(f"DAX USAGE: {field_def['use_in_dax']}")

        if field_def.get("consumer_language"):
            parts.append(f"USER MAY SAY: {', '.join(field_def['consumer_language'])}")

        if field_def.get("warning"):
            parts.append(f"WARNING: {field_def['warning']}")

        docs.append({
            "id":       f"schema_{field_name.lower().replace(' ', '_')}",
            "text":     "\n".join(parts),
            "metadata": {
                "source":   "schema_context",
                "field":    field_name,
                "category": field_def.get("category", ""),
                "slicer":   str(field_def.get("slicer", False)),
                "type":     field_def.get("data_type", ""),
            }
        })

    log.info(f"[dictionary] Built {len(docs)} schema documents")
    return docs


def _build_business_logic_documents(data: dict) -> list[dict]:
    """
    # * Converts 02_business_logic.json into embeddable documents.
    # * Produces documents for bucket rules, movement logic, metrics, and null handling.

    Args:
        data : Parsed JSON from 02_business_logic.json

    Returns:
        List of {"id": str, "text": str, "metadata": dict}
    """
    docs = []

    # * Bucket classification rules — one doc per bucket
    for rule in data.get("bucket_classification", {}).get("rules", []):
        bucket = rule.get("bucket", "unknown")
        docs.append({
            "id":   f"bucket_rule_{bucket.lower().replace(' ', '_').replace('-', '_')}",
            "text": (
                f"BUCKET RULE: {bucket}\n"
                f"CONDITION: {rule.get('condition', '')}\n"
                f"DESCRIPTION: {rule.get('description', '')}\n"
                f"DAX FILTER: {rule.get('dax_filter', '')}"
            ),
            "metadata": {"source": "business_logic", "topic": "bucket_classification", "bucket": bucket}
        })

    # * Bucket movement logic — one doc per movement row
    for i, rule in enumerate(data.get("bucket_movement_logic", {}).get("rules", [])):
        op  = rule.get("op_bucket", "")
        cls = rule.get("closing_bucket", [])
        st  = rule.get("status", "")
        docs.append({
            "id":   f"movement_{i}_{op.lower().replace(' ', '_').replace('-', '_')}",
            "text": (
                f"BUCKET MOVEMENT: Opening={op} → Closing={cls}\n"
                f"RESULT STATUS: {st}\n"
                f"MEANING: {rule.get('meaning', '')}"
            ),
            "metadata": {"source": "business_logic", "topic": "bucket_movement", "status": st}
        })

    # * Metrics — one doc per metric
    for metric_key, metric in data.get("metrics", {}).items():
        docs.append({
            "id":   f"metric_{metric_key.lower()}",
            "text": (
                f"METRIC: {metric.get('label', metric_key)}\n"
                f"DESCRIPTION: {metric.get('description', '')}\n"
                f"FORMULA: {metric.get('formula', '')}\n"
                f"DAX: {metric.get('dax_pattern', '')}\n"
                f"USER MAY SAY: {', '.join(metric.get('consumer_language', []))}\n"
                f"IMPORTANT: {metric.get('important', '')}"
            ),
            "metadata": {"source": "business_logic", "topic": "metric", "metric": metric_key}
        })

    # * Granularity rules — one doc
    gran = data.get("granularity_rules", {})
    if gran:
        docs.append({
            "id":   "granularity_rules",
            "text": (
                f"GRANULARITY RULES:\n"
                f"Customer level: {gran.get('customer_level_metrics', {}).get('rule', '')}\n"
                f"Applies to: {gran.get('customer_level_metrics', {}).get('applies_to', '')}\n"
                f"Loan level: {gran.get('loan_level_metrics', {}).get('rule', '')}\n"
                f"Reason: {gran.get('customer_level_metrics', {}).get('reason', '')}"
            ),
            "metadata": {"source": "business_logic", "topic": "granularity"}
        })

    # * Null handling — one doc
    null_rules = data.get("null_handling", {})
    if null_rules:
        null_text = "NULL HANDLING RULES:\n" + "\n".join(
            f"{k}: {v}" for k, v in null_rules.items()
        )
        docs.append({
            "id":   "null_handling",
            "text": null_text,
            "metadata": {"source": "business_logic", "topic": "null_handling"}
        })

    log.info(f"[dictionary] Built {len(docs)} business logic documents")
    return docs


def _build_language_mapping_documents(data: dict) -> list[dict]:
    """
    # * Converts 03_language_mapping.json into embeddable documents.
    # * One doc per language mapping entry — each maps consumer phrases to fields/metrics.

    Args:
        data : Parsed JSON from 03_language_mapping.json

    Returns:
        List of {"id": str, "text": str, "metadata": dict}
    """
    docs = []

    for i, mapping in enumerate(data.get("language_to_field_mapping", [])):
        phrases = mapping.get("consumer_says", [])
        field   = mapping.get("maps_to_field", "")
        value   = mapping.get("maps_to_value", "")
        metric  = mapping.get("metric", "")
        dax     = mapping.get("dax_hint", "")
        warning = mapping.get("warning", "")

        docs.append({
            "id":   f"lang_map_{i}_{field.lower().replace(' ', '_').replace('/', '_')}",
            "text": (
                f"LANGUAGE MAPPING:\n"
                f"USER SAYS: {', '.join(phrases)}\n"
                f"MAPS TO FIELD: {field}\n"
                f"MAPS TO VALUE: {value}\n"
                f"METRIC: {metric}\n"
                f"DAX HINT: {dax}\n"
                f"WARNING: {warning}"
            ),
            "metadata": {
                "source": "language_mapping",
                "field":  field,
                "metric": metric,
            }
        })

    # * Intent-to-DAX examples — one doc per example
    for i, example in enumerate(data.get("intent_to_dax_examples", [])):
        docs.append({
            "id":   f"intent_example_{i}",
            "text": (
                f"EXAMPLE QUERY:\n"
                f"USER SAYS: {example.get('user_says', '')}\n"
                f"METRIC: {example.get('identified_metric', '')}\n"
                f"SLICERS NEEDED: {', '.join(example.get('identified_slicers', []))}\n"
                f"CLARIFYING QUESTIONS: {', '.join(example.get('clarifying_questions', []))}\n"
                f"DAX SKELETON: {example.get('dax_skeleton', '')}"
            ),
            "metadata": {"source": "language_mapping", "topic": "intent_example"}
        })

    log.info(f"[dictionary] Built {len(docs)} language mapping documents")
    return docs


def _build_visualization_documents(data: dict) -> list[dict]:
    """
    # * Converts 04_visualization_context.json into embeddable documents.
    # * One doc per chart selection rule and metric-to-chart mapping.

    Args:
        data : Parsed JSON from 04_visualization_context.json

    Returns:
        List of {"id": str, "text": str, "metadata": dict}
    """
    docs = []

    # * Chart selection rules
    for i, rule in enumerate(data.get("chart_selection_rules", {}).get("rules", [])):
        docs.append({
            "id":   f"chart_rule_{i}",
            "text": (
                f"CHART SELECTION RULE:\n"
                f"CONDITION: {rule.get('condition', '')}\n"
                f"CHART TYPE: {rule.get('chart_type', '')}\n"
                f"REASON: {rule.get('reason', '')}\n"
                f"EXAMPLE: {rule.get('example_question', '')}"
            ),
            "metadata": {
                "source":     "visualization",
                "topic":      "chart_rule",
                "chart_type": rule.get("chart_type", ""),
            }
        })

    # * Metric to chart mappings
    for metric, config in data.get("metric_to_chart_mapping", {}).items():
        docs.append({
            "id":   f"viz_metric_{metric.lower().replace(' ', '_').replace('%', 'pct')}",
            "text": (
                f"VISUALIZATION FOR METRIC: {metric}\n"
                f"CHART TYPE: {config.get('chart_type', '')}\n"
                f"X AXIS: {config.get('x_axis', '')}\n"
                f"Y AXIS: {config.get('y_axis', '')}\n"
                f"HUE: {config.get('hue', '')}\n"
                f"COLOR PALETTE: {config.get('color_palette', '')}\n"
                f"TITLE TEMPLATE: {config.get('title_template', '')}"
            ),
            "metadata": {
                "source":     "visualization",
                "topic":      "metric_chart_map",
                "metric":     metric,
                "chart_type": config.get("chart_type", ""),
            }
        })

    # * Color palette guide — one doc
    palette_guide = data.get("color_palette_guide", {})
    if palette_guide:
        docs.append({
            "id":   "color_palette_guide",
            "text": "COLOR PALETTE GUIDE:\n" + "\n".join(
                f"{k}: {v}" for k, v in palette_guide.items()
            ),
            "metadata": {"source": "visualization", "topic": "palette"}
        })

    log.info(f"[dictionary] Built {len(docs)} visualization documents")
    return docs


def _build_dax_pattern_documents(data: dict) -> list[dict]:
    """
    # * Converts 05_dax_patterns.json into embeddable documents.
    # * One doc per DAX pattern — includes description and the full DAX query.

    Args:
        data : Parsed JSON from 05_dax_patterns.json

    Returns:
        List of {"id": str, "text": str, "metadata": dict}
    """
    docs = []

    for pattern_key, pattern in data.get("dax_patterns", {}).items():
        docs.append({
            "id":   f"dax_{pattern_key}",
            "text": (
                f"DAX PATTERN: {pattern_key}\n"
                f"DESCRIPTION: {pattern.get('description', '')}\n"
                f"DAX QUERY:\n{pattern.get('dax', pattern.get('with_branch_filter', ''))}"
            ),
            "metadata": {
                "source":  "dax_patterns",
                "pattern": pattern_key,
            }
        })

    # * Filter snippets as one combined doc
    snippets = data.get("filter_snippets", {})
    if snippets:
        docs.append({
            "id":   "dax_filter_snippets",
            "text": "DAX FILTER SNIPPETS:\n" + "\n".join(
                f"{k}: {v}" for k, v in snippets.items()
                if not k.startswith("_")
            ),
            "metadata": {"source": "dax_patterns", "topic": "filter_snippets"}
        })

    log.info(f"[dictionary] Built {len(docs)} DAX pattern documents")
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# * DATA DICTIONARY CLASS
# ──────────────────────────────────────────────────────────────────────────────

class DataDictionary:
    """
    # * Manages all context knowledge for the Loan Dashboard AI pipeline.
    # * Loads 5 context JSONs into ChromaDB on first run.
    # * Provides semantic search to fetch only relevant context per user query.
    # * Provides formatted prompt strings for Qwen Coder and Qwen 7B.

    # ? Why a class and not just module-level functions?
    # ? The ChromaDB client and collection are stateful — keeping them as
    # ? instance attributes avoids reconnecting on every call.
    # ? The singleton pattern (get_dictionary()) ensures only one instance
    # ? exists across the entire app session.
    """

    def __init__(self):
        self._client:     Optional[chromadb.ClientAPI]       = None
        self._collection: Optional[chromadb.Collection]      = None
        self._embedding_fn = None
        self._loaded:     bool                               = False
        self._total_docs: int                                = 0

    # ── Initialise ────────────────────────────────────────────────────────────

    def initialise(self) -> bool:
        """
        # * Connect to ChromaDB, create collection, load context JSONs.
        # * Safe to call multiple times — skips loading if already done.
        # * Returns True if successful, False if ChromaDB failed.

        # ? Called once at app startup from app.py
        # ? Subsequent calls are no-ops if already initialised
        """
        if self._loaded:
            log.debug("[dictionary] Already initialised — skipping")
            return True

        log.info("[dictionary] Initialising ChromaDB and loading context JSONs...")

        with benchmark("dictionary_initialise"):
            try:
                # * Connect to persistent ChromaDB at configured path
                self._client = chromadb.PersistentClient(
                    path=str(CHROMADB_PATH)
                )

                # * Set up sentence-transformer embedding function
                # ? all-MiniLM-L6-v2 runs locally — no API calls, no cost
                self._embedding_fn = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=EMBEDDING_MODEL
                    )
                )

                # * Get or create the collection
                self._collection = self._client.get_or_create_collection(
                    name=CHROMA_COLLECTION,
                    embedding_function=self._embedding_fn,
                    metadata={"hnsw:space": "cosine"},   # cosine similarity
                )

                # * Load documents only if collection is empty
                existing = self._collection.count()
                if existing == 0:
                    log.info("[dictionary] Collection empty — loading all context JSONs...")
                    self._load_all_jsons()
                else:
                    log.info(f"[dictionary] Collection already has {existing} documents — skipping reload")
                    self._total_docs = existing

                self._loaded = True
                log.info(f"[dictionary] Initialised successfully | total_docs={self._total_docs}")
                return True

            except Exception as e:
                log.error(f"[dictionary] ChromaDB initialisation failed | error={e}")
                self._loaded = False
                return False

    # ── Load all JSONs ────────────────────────────────────────────────────────

    def _load_all_jsons(self):
        """
        # * Reads all 5 context JSONs, builds documents, inserts into ChromaDB.
        # ! Only called when the ChromaDB collection is empty (first run).
        """
        builders = [
            (SCHEMA_CONTEXT_PATH,   _build_schema_documents,        "schema"),
            (BUSINESS_LOGIC_PATH,   _build_business_logic_documents, "business_logic"),
            (LANGUAGE_MAPPING_PATH, _build_language_mapping_documents,"language_mapping"),
            (VISUALIZATION_PATH,    _build_visualization_documents,   "visualization"),
            (DAX_PATTERNS_PATH,     _build_dax_pattern_documents,     "dax_patterns"),
        ]

        all_docs = []

        for json_path, builder_fn, source_name in builders:
            if not json_path.exists():
                # ! If a context JSON is missing, log error but continue
                log.error(f"[dictionary] Context JSON not found: {json_path}")
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                docs = builder_fn(data)
                all_docs.extend(docs)
                log.info(f"[dictionary] Loaded {len(docs)} docs from {source_name}")
            except Exception as e:
                log.error(f"[dictionary] Failed to load {source_name} | error={e}")

        if not all_docs:
            # ! No documents loaded — ChromaDB will be empty
            log.error("[dictionary] No documents loaded — check Context/ folder")
            return

        # * Insert in chunks to avoid ChromaDB batch size limits
        # ? ChromaDB recommends batches of max 500 documents
        BATCH_SIZE = 200
        chunks     = chunk_list(all_docs, BATCH_SIZE)

        for i, batch in enumerate(chunks):
            try:
                self._collection.add(
                    ids       = [doc["id"]       for doc in batch],
                    documents = [doc["text"]      for doc in batch],
                    metadatas = [doc["metadata"]  for doc in batch],
                )
                log.debug(f"[dictionary] Inserted batch {i+1}/{len(chunks)} | {len(batch)} docs")
            except Exception as e:
                log.error(f"[dictionary] Failed to insert batch {i+1} | error={e}")

        self._total_docs = self._collection.count()
        log.info(f"[dictionary] All context JSONs loaded | total={self._total_docs} documents")

    # ── Semantic search ───────────────────────────────────────────────────────

    def search(
        self,
        query:    str,
        top_k:   int             = CHROMA_TOP_K,
        source:  Optional[str]   = None,
    ) -> list[str]:
        """
        # * Fetch the most relevant context documents for a user query.
        # * Returns list of plain text chunks ready for prompt injection.

        Args:
            query  : User's natural language question
            top_k  : Number of results to return (default from config)
            source : Optional filter by source JSON
                     e.g. "schema_context", "dax_patterns", "business_logic"

        Returns:
            List of text strings — most relevant first.

        Example:
            chunks = dictionary.search("how many bounced loans last month")
            # → ["FIELD: Bounce status ...", "METRIC: Bounce Count ...", ...]
        """
        if not self._loaded or self._collection is None:
            log.warning("[dictionary] search() called but not initialised — returning empty")
            return []

        if not query or not query.strip():
            log.warning("[dictionary] search() called with empty query")
            return []

        try:
            where = {"source": source} if source else None

            results = self._collection.query(
                query_texts = [query.strip()],
                n_results   = min(top_k, self._total_docs),
                where       = where,
            )

            docs = results.get("documents", [[]])[0]
            log.debug(f"[dictionary] search | query='{truncate_string(query, 50)}' | returned={len(docs)} chunks")
            return docs

        except Exception as e:
            log.error(f"[dictionary] search() failed | error={e}")
            return []

    # ── Prompt builders ───────────────────────────────────────────────────────

    def get_coder_context(self, query: str) -> str:
        """
        # * Build the full context string for Qwen Coder (14B).
        # * Fetches relevant schema + business logic + language mapping + DAX patterns.
        # * This is injected into the system prompt for intent analysis and DAX generation.

        Args:
            query : User's question

        Returns:
            Formatted multi-section context string for Qwen Coder system prompt.
        """
        with benchmark("dictionary_coder_context", query=query):

            # * Fetch chunks from each relevant source
            schema_chunks   = self.search(query, top_k=6,  source="schema_context")
            logic_chunks    = self.search(query, top_k=5,  source="business_logic")
            language_chunks = self.search(query, top_k=4,  source="language_mapping")
            dax_chunks      = self.search(query, top_k=4,  source="dax_patterns")

            sections = []

            if schema_chunks:
                sections.append(
                    "=== RELEVANT FIELDS ===\n" + "\n\n".join(schema_chunks)
                )
            if logic_chunks:
                sections.append(
                    "=== BUSINESS LOGIC & METRICS ===\n" + "\n\n".join(logic_chunks)
                )
            if language_chunks:
                sections.append(
                    "=== LANGUAGE MAPPING ===\n" + "\n\n".join(language_chunks)
                )
            if dax_chunks:
                sections.append(
                    "=== DAX PATTERNS ===\n" + "\n\n".join(dax_chunks)
                )

            context = "\n\n".join(sections)
            token_estimate = estimate_tokens(context)
            log.debug(f"[dictionary] coder_context | ~{token_estimate} tokens")
            return context

    def get_viz_context(self, query: str) -> str:
        """
        # * Build context string for Qwen 7B (chart decision model).
        # * Fetches only visualization rules and metric-to-chart mappings.
        # * Kept smaller since Qwen 7B has a smaller effective context window.

        Args:
            query : User's question or the identified metric name

        Returns:
            Formatted context string for Qwen 7B system prompt.
        """
        with benchmark("dictionary_viz_context", query=query):

            viz_chunks = self.search(query, top_k=6, source="visualization")

            if not viz_chunks:
                return ""

            context = "=== CHART SELECTION RULES ===\n" + "\n\n".join(viz_chunks)
            log.debug(f"[dictionary] viz_context | ~{estimate_tokens(context)} tokens")
            return context

    def get_all_slicer_fields(self) -> list[str]:
        """
        # * Returns all field names marked as slicer=Yes from the schema.
        # * Used by filters.py to build the sidebar filter options.

        Returns:
            List of field name strings.
        """
        if not self._loaded or self._collection is None:
            return []

        try:
            results = self._collection.get(
                where={"slicer": "True"}
            )
            fields = [
                m.get("field", "")
                for m in results.get("metadatas", [])
                if m.get("field")
            ]
            return list(dict.fromkeys(fields))  # deduplicate preserving order
        except Exception as e:
            log.error(f"[dictionary] get_all_slicer_fields failed | error={e}")
            return []

    # ── Reload ────────────────────────────────────────────────────────────────

    def reload(self) -> bool:
        """
        # * Force reload all context JSONs into ChromaDB.
        # * Use this after updating any of the 5 context JSON files.
        # ? Deletes and recreates the collection — takes ~30 seconds.

        Returns:
            True if reload succeeded, False if it failed.
        """
        log.info("[dictionary] Force reloading all context JSONs...")
        try:
            if self._client:
                self._client.delete_collection(CHROMA_COLLECTION)
            self._collection = None
            self._loaded     = False
            self._total_docs = 0
            return self.initialise()
        except Exception as e:
            log.error(f"[dictionary] reload() failed | error={e}")
            return False

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """# * True if ChromaDB is connected and documents are loaded."""
        return self._loaded and self._collection is not None

    @property
    def total_documents(self) -> int:
        """# * Total number of documents in the ChromaDB collection."""
        return self._total_docs


# ──────────────────────────────────────────────────────────────────────────────
# * SINGLETON ACCESSOR
# * One shared DataDictionary instance for the entire app session.
# * Import and call get_dictionary() anywhere in the project.
# ──────────────────────────────────────────────────────────────────────────────

_instance: Optional[DataDictionary] = None


def get_dictionary() -> DataDictionary:
    """
    # * Returns the shared DataDictionary instance.
    # * Creates and initialises it on first call.
    # * All subsequent calls return the same object.

    Usage in any file:
        from core.dictionary import get_dictionary
        dictionary = get_dictionary()
        context    = dictionary.get_coder_context(user_question)
    """
    global _instance
    if _instance is None:
        _instance = DataDictionary()
        _instance.initialise()
    return _instance