"""
embedding_mapper.py
-------------------
Semantic column mapping using intfloat/e5-large-v2 embeddings.

Architecture:
- Lazy loads the E5 model on first use (not at import time)
- Pre-embeds all canonical business names from SYNONYM_MAP once at startup
- Uses strict E5 format: 'query:' for column context, 'passage:' for glossaries
- Expands ERP abbreviations and detects regex patterns in sample values
- For each incoming column, embeds a rich context string and applies a hybrid
  ranking system (semantic similarity + pattern/abbrev heuristics)
- Returns top-3 candidates with similarity scores and reasoning metadata
- Caches results so repeated column contexts are never re-embedded
- Falls back silently to the rule engine if anything fails

This module is called by schema_mapper.py ONLY.
Nothing else in the application touches this file directly.
"""

from __future__ import annotations

import re
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── E5 Upgrades: Abbreviations & Patterns ─────────────────────────────────────

ERP_ABBREVIATIONS = {
    "inv": "invoice", "amt": "amount", "qty": "quantity", "gp": "gross profit",
    "nsv": "net sales", "sku": "product code", "no": "number",
    "cust": "customer", "cat": "category", "disc": "discount",
    "val": "value", "dt": "date", "loc": "location", "sp": "salesperson",
    "id": "identifier", "num": "number", "desc": "description",
    "org": "organization", "rev": "revenue", "cogs": "cost of goods sold"
}

def _expand_column_name(col_name: str) -> str:
    """Expands shorthand ERP column names into full semantic words."""
    if not col_name:
        return ""
    # Split by underscore or camelCase
    parts = re.split(r'[_]+|(?=[A-Z])', col_name.strip())
    expanded = []
    for part in parts:
        clean_part = part.lower().strip()
        if not clean_part:
            continue
        expanded.append(ERP_ABBREVIATIONS.get(clean_part, clean_part))
    return " ".join(expanded)

def _detect_patterns(sample_values: list) -> list:
    """Uses regex to detect common ERP data patterns in sample values."""
    patterns = []
    # Convert samples to a single string, ignoring NaNs
    samples_str = " ".join([str(v) for v in sample_values if pd.notna(v)])
    
    if re.search(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b', samples_str):
        patterns.append("date format")
    if re.search(r'\b[A-Z0-9]{5,15}\b', samples_str):
        patterns.append("alphanumeric id")
    if re.search(r'\b\d+\.\d{2}\b', samples_str):
        patterns.append("decimal currency")
        
    return patterns


# ── thresholds ────────────────────────────────────────────────────────────────
HIGH_CONFIDENCE   = 0.85
MEDIUM_CONFIDENCE = 0.65

# ── module-level singletons (populated lazily) ────────────────────────────────
_model        = None                 # SentenceTransformer instance
_model_failed = False                # set True if model load fails — triggers fallback
_canon_names  : list[str] = []       # ordered list of canonical business names
_canon_embeds : Optional[np.ndarray] = None  # shape (N, embed_dim)

# ── embedding cache ───────────────────────────────────────────────────────────
# Maps context_string → numpy embedding vector
_embed_cache: dict[str, np.ndarray] = {}


def _load_model() -> bool:
    """
    Lazy-load the E5 model and pre-embed all canonical business names.
    Returns True on success, False on failure.
    Called at most once — subsequent calls are no-ops.
    """
    global _model, _model_failed, _canon_names, _canon_embeds

    if _model is not None:
        return True                     # already loaded
    if _model_failed:
        return False                    # already failed — stop retrying

    try:
        from sentence_transformers import SentenceTransformer
        # Import SYNONYM_MAP here to avoid circular imports
        from schema_mapper import SYNONYM_MAP

        logger.info("Loading intfloat/e5-large-v2 model...")
        _model = SentenceTransformer("intfloat/e5-large-v2")

        _canon_names = list(SYNONYM_MAP.keys())
        
        # E5-Large-v2 REQUIRES the 'passage:' prefix for the target search space
        passages = [f"passage: Represent this business data column: {name}" for name in _canon_names]
        
        _canon_embeds = _model.encode(
            passages,
            normalize_embeddings=True,   # L2-normalise → cosine sim = dot product
            show_progress_bar=False,
            batch_size=32,
        )   

        logger.info(f"E5 model loaded. {len(_canon_names)} canonical embeddings ready.")
        return True

    except Exception as exc:
        logger.warning(f"E5 model load failed — falling back to rule engine. Reason: {exc}")
        _model_failed = True    
        return False


def _embed(text: str) -> Optional[np.ndarray]:
    """
    Embed a single text string with caching.
    Returns L2-normalised numpy vector, or None on failure.
    """
    if text in _embed_cache:
        return _embed_cache[text]
    try:
        # E5-Large-v2 REQUIRES the 'query:' prefix for the input search query
        query = f"query: {text}"
        vec = _model.encode( 
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        _embed_cache[text] = vec
        return vec
    except Exception as exc:
        logger.warning(f"Embedding failed for '{text}': {exc}")
        return None


def _build_context_string(column_info: dict, neighbors: list = None) -> str:
    """
    Build a rich context string from column metadata.
    This is what gets embedded for semantic matching.
    """
    col_name     = column_info.get("column_name", "")      
    data_type    = column_info.get("data_type", "")
    samples      = column_info.get("sample_values", [])[:5]
    null_count   = column_info.get("null_count", 0)
    unique_count = column_info.get("unique_count", 0)

    expanded_name = _expand_column_name(col_name)
    patterns      = _detect_patterns(samples)

    # Make data type human-readable
    type_map = {
        "object":  "text",
        "float64": "decimal number",
        "int64":   "integer number",
        "datetime64[ns]": "date",
        "bool":    "boolean",
    }
    readable_type = type_map.get(str(data_type), str(data_type))

    sample_str   = ", ".join(str(s) for s in samples) if samples else "no samples"
    pattern_str  = f" Detected patterns: {', '.join(patterns)}." if patterns else ""
    neighbor_str = f" Neighboring columns: {', '.join(neighbors)}." if neighbors else ""

    return (
        f"ERP column name: '{col_name}' (Expanded meaning: '{expanded_name}'). "
        f"Data type: {readable_type}. "
        f"Sample values: {sample_str}. "
        f"Unique values: {unique_count}. "
        f"Missing values: {null_count}."
        f"{pattern_str}{neighbor_str}"
    )


def get_top_candidates(
    column_info: dict,
    neighbors: list = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Main public function of this module.

    Embeds the column context, calculates similarities, applies hybrid heuristic
    boosts (abbreviations/patterns), and returns the top candidates.

    Return format:
    [
        {"business_name": "net sales value", "similarity": 0.94, "reasons": [...]},
        ...
    ]
    """
    if not _load_model():
        return []

    context = _build_context_string(column_info, neighbors)
    vec = _embed(context)
    if vec is None:
        return []

    # Calculate baseline cosine similarity
    similarities = _canon_embeds @ vec   # shape: (N,)

    expanded_col_name = _expand_column_name(column_info.get("column_name", ""))
    patterns = _detect_patterns(column_info.get("sample_values", []))

    candidates_pool = []
    
    # Apply hybrid ranking
    for idx, name in enumerate(_canon_names):
        sim_score = float(similarities[idx])
        final_score = sim_score
        reasons = [f"Semantic similarity: {sim_score:.2f}"]
        
        name_lower = name.lower()
        expanded_lower = expanded_col_name.lower()

        # Heuristic A: Exact / Expanded Name Match
        if name_lower in expanded_lower or expanded_lower in name_lower:
            final_score += 0.15
            reasons.append("Exact/Expanded name match (+0.15)")
            
        # Heuristic B: Datatype / Pattern Compatibility
        if "date" in name_lower and "date format" in patterns:
            final_score += 0.10
            reasons.append("Date pattern match (+0.10)")
        if any(w in name_lower for w in ["amount", "price", "sales", "revenue", "discount"]) and "decimal currency" in patterns:
            final_score += 0.10
            reasons.append("Currency pattern match (+0.10)")
            
        # Cap confidence at 1.0
        final_score = min(1.0, final_score)
        
        candidates_pool.append({
            "business_name": name,
            "similarity":    final_score,
            "reasons":       reasons
        })

    # Sort by the final hybrid score descending
    candidates_pool.sort(key=lambda x: x["similarity"], reverse=True)

    return candidates_pool[:top_k]


def combine_confidence(
    synonym_confidence: Optional[float],
    embedding_similarity: float,
) -> float:
    """
    Weighted combination of rule-engine confidence and embedding similarity.
    """
    if synonym_confidence is not None:
        return round(0.6 * synonym_confidence + 0.4 * embedding_similarity, 4)
    return round(embedding_similarity, 4)


def is_available() -> bool:
    """Check if the embedding model is loaded and ready."""
    return _model is not None and not _model_failed


def cache_stats() -> dict:
    """Return embedding cache statistics for monitoring."""
    return {
        "cached_embeddings": len(_embed_cache),
        "canonical_names":   len(_canon_names),
        "model_loaded":      _model is not None,
        "model_failed":      _model_failed,
    }