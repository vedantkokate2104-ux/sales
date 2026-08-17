"""
transaction_key.py
-------------------
Business-transaction identity for the ERP pipeline (Phase 1, Steps 1 & 2).

WHY THIS EXISTS
----------------
data_cleaner.clean() used to remove duplicates with

    df.drop_duplicates(subset=[id_cols[0]], keep="first")

where `id_cols[0]` is simply the FIRST column the schema mapper happened to
categorise as an "identifier". On the bundled dataset that resolves to the
invoice-number column, and the bundled dataset RECYCLES invoice numbers:
`INV1000` appears twice, on 2024-10-16 and on 2022-03-14, for different
quantities and different amounts. Those are two genuinely different
transactions that merely share an identifier string — not a duplicate
record. Deduplicating on that column alone destroyed 212 of 392 rows (54%
of the dataset, 57% of total revenue) and starved every downstream
forecast of the history it needed.

An identifier column being NON-UNIQUE is not evidence of duplication. It
is evidence that the column is not a primary key. This module makes that
distinction explicit and reusable, and deliberately separates three
different things that the old one-line call conflated:

  1. EXACT DUPLICATE ROWS
     Every field identical. Almost always a genuine double-import.
     Safe to remove.

  2. DUPLICATE BUSINESS TRANSACTIONS
     Identical on a composite BUSINESS KEY (invoice + date + product +
     customer + quantity + revenue) but possibly differing on incidental
     columns (a free-text note, a recomputed tax field, a status that has
     since moved on). The same real-world event recorded twice. Safe to
     remove, because the key covers what actually identifies the event.

  3. LEGITIMATE REPEATED IDENTIFIERS
     Rows sharing an invoice/customer/product identifier but differing on
     date, product, quantity, or amount. Distinct business events.
     MUST BE PRESERVED. This is the case the old logic got wrong.

DESIGN CONSTRAINTS
-------------------
  - Uniqueness is never invented. A business key is only used when the
    dataset actually contains enough identifying fields to support one
    (see `resolve_business_key` for the exact rule). When it does not,
    this module falls back to exact-row duplicate detection only, and
    says so in `basis` / `limitation` rather than guessing.
  - Nothing is ever removed silently. `analyze_duplicates()` returns full
    counts and a stated reason for every removal, which data_cleaner
    surfaces in its report.
  - Column resolution is STRICT (see `_strict_col`). analytics.get_col()
    has a deliberately loose second strategy that matches on any shared
    word, which is right for KPI lookup ("close enough" beats "no KPI at
    all") but wrong here: matching "invoice number" to the invoice DATE
    column would silently corrupt the identity key, so this module
    requires the full business-name phrase to appear.
"""

import warnings as _warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

from duplicate_detector import _normalize

# Business names (as produced by schema_mapper) whose combination
# identifies one real business transaction, in descending order of
# identifying power. Each entry is (business_name_phrase, role) — `role`
# is what the component contributes to the key, and drives the
# sufficiency rule in resolve_business_key().
_KEY_COMPONENTS: List[Tuple[str, str]] = [
    ("invoice number", "identifier"),
    ("invoice date",   "date"),
    ("product id",     "identifier"),
    ("customer code",  "identifier"),
    ("quantity sold",  "measure"),
    ("net sales",      "measure"),
]

# A composite key is only trusted when it has at least this many resolved
# components AND satisfies the role rule below. Three arbitrary columns
# are not enough to assert "these two rows are the same event".
MIN_KEY_COMPONENTS = 3


def _strict_col(df: pd.DataFrame, mappings: List[dict], business_name: str) -> Optional[str]:
    """
    Resolve a business name to an actual column, requiring the FULL phrase
    to appear in the mapped business name — unlike analytics.get_col(),
    whose any-shared-word fallback would happily resolve "invoice number"
    to the invoice DATE column. See the module docstring for why that
    looseness is unacceptable when building an identity key.

    Returns the first matching column present in `df`, or None.
    """
    if not mappings:
        return None
    search = business_name.lower().strip()
    for m in mappings:
        mapped = str(m.get("business_name", "")).lower()
        col = m.get("original_column", "")
        if search in mapped and col in df.columns:
            return col
    return None


def resolve_business_key(df: pd.DataFrame, mappings: List[dict]) -> Tuple[List[str], str, Optional[str]]:
    """
    Build the list of columns forming this dataset's business transaction
    key.

    A key is only considered reliable when it resolves at least
    MIN_KEY_COMPONENTS components AND includes both:
      - a DATE component (without a timestamp, two rows describing
        different events are indistinguishable from a re-import), and
      - at least one MEASURE (quantity or revenue — what actually
        differs between two separate line items on the same day).

    An identifier alone is deliberately NOT sufficient: an identifier that
    repeats is exactly the situation this module exists to handle, so
    treating it as a key would reintroduce the original bug.

    Returns (key_columns, basis, limitation):
      key_columns — [] when no reliable key exists
      basis        — human-readable description of what the key is made of
      limitation   — None when a key was built; otherwise a plain-language
                     statement of what could not be determined and what
                     the caller should fall back to
    """
    resolved: List[Tuple[str, str]] = []
    for business_name, role in _KEY_COMPONENTS:
        col = _strict_col(df, mappings, business_name)
        if col and col not in [c for c, _ in resolved]:
            resolved.append((col, role))

    key_cols = [c for c, _ in resolved]
    roles = {r for _, r in resolved}

    if len(key_cols) < MIN_KEY_COMPONENTS or "date" not in roles or "measure" not in roles:
        missing = []
        if "date" not in roles:
            missing.append("a transaction date column")
        if "measure" not in roles:
            missing.append("a quantity or revenue column")
        if len(key_cols) < MIN_KEY_COMPONENTS:
            missing.append(
                f"at least {MIN_KEY_COMPONENTS} identifying columns (found {len(key_cols)})"
            )
        limitation = (
            "No reliable business transaction key could be constructed — this dataset is missing "
            + ", ".join(missing)
            + ". Duplicate removal was therefore restricted to EXACT duplicate rows only, which is "
            "the conservative choice: some genuine double-entries that differ in an incidental "
            "field may remain in the data, but no legitimate transaction is discarded."
        )
        return [], "exact row match (no business key available)", limitation

    basis = " + ".join(key_cols)
    return key_cols, basis, None


def _normalised_key_frame(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """
    Normalise the key columns before comparison so that trivial
    representational differences (trailing whitespace, letter case,
    floating-point noise beyond 2dp, a date carried as a string in one row
    and a Timestamp in another) do not make two records of the same event
    look distinct. Reuses duplicate_detector._normalize for scalar
    normalisation so this module and the existing duplicate-review UI
    always agree on what "the same value" means.
    """
    out = {}
    for col in key_cols:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            out[col] = series.dt.normalize().astype(str)
            continue
        # Probe whether this column is a date at all. Pandas emits a
        # "could not infer format" UserWarning here on ambiguous columns;
        # that is expected noise for a probe (we only care whether it
        # parses, and the result is used solely to normalise a key for
        # equality comparison, never as an analysis date), so it is
        # suppressed rather than surfaced to the user.
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        # Treat as a date column only if essentially all of it parses —
        # otherwise a numeric/text column could be coerced into
        # meaningless dates and silently weaken the key.
        if parsed.notna().mean() > 0.95:
            out[col] = parsed.dt.normalize().astype(str)
        else:
            out[col] = series.map(_normalize).astype(str)
    return pd.DataFrame(out, index=df.index)


def analyze_duplicates(df: pd.DataFrame, mappings: List[dict]) -> Dict[str, object]:
    """
    Classify every kind of duplication present in `df` WITHOUT modifying
    it. Pure analysis — `deduplicate_transactions()` below is what
    actually removes anything, and it uses exactly these numbers.

    The three categories are reported separately (see module docstring):
    exact duplicate rows, business-key duplicate rows, and rows that
    merely share a repeated identifier and are therefore KEPT.
    """
    total_rows = int(len(df))
    key_cols, basis, limitation = resolve_business_key(df, mappings)

    exact_duplicate_rows = int(df.duplicated(keep="first").sum())

    if key_cols:
        key_frame = _normalised_key_frame(df, key_cols)
        business_key_duplicate_rows = int(key_frame.duplicated(keep="first").sum())
        strategy = "business_key"
        removal_reason = (
            f"Removed rows that are identical on the business transaction key ({basis}) — the same "
            f"real-world transaction recorded more than once. Rows sharing only an invoice number "
            f"but differing in date, product, quantity or value were KEPT, because a repeated "
            f"identifier does not make two transactions the same transaction."
        )
    else:
        business_key_duplicate_rows = exact_duplicate_rows
        strategy = "exact_row"
        removal_reason = (
            "Removed exact duplicate rows only (every field identical). No business transaction "
            "key could be constructed for this dataset, so no further deduplication was attempted."
        )

    # Rows that share a repeated identifier value but are NOT duplicates by
    # the rule above — i.e. precisely the rows the previous implementation
    # destroyed. Reported so the effect of the fix is visible and auditable.
    identifier_col = _strict_col(df, mappings, "invoice number")
    repeated_identifier_values = 0
    repeated_identifier_rows = 0
    preserved_repeated_identifier_rows = 0
    if identifier_col:
        id_dupe_mask = df.duplicated(subset=[identifier_col], keep="first")
        repeated_identifier_rows = int(id_dupe_mask.sum())
        repeated_identifier_values = int(
            df[df.duplicated(subset=[identifier_col], keep=False)][identifier_col].nunique()
        )
        preserved_repeated_identifier_rows = max(
            0, repeated_identifier_rows - business_key_duplicate_rows
        )

    return {
        "total_rows":                  total_rows,
        "exact_duplicate_rows":        exact_duplicate_rows,
        "business_key_duplicate_rows": business_key_duplicate_rows,
        "identifier_column":           identifier_col,
        "repeated_identifier_rows":    repeated_identifier_rows,
        "repeated_identifier_values":  repeated_identifier_values,
        "preserved_repeated_identifier_rows": preserved_repeated_identifier_rows,
        "business_key_columns":        key_cols,
        "business_key_basis":          basis,
        "strategy":                    strategy,
        "limitation":                  limitation,
        "rows_removed":                business_key_duplicate_rows,
        "rows_retained":               total_rows - business_key_duplicate_rows,
        "removal_reason":              removal_reason,
    }


def deduplicate_transactions(df: pd.DataFrame, mappings: List[dict]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Remove duplicate business transactions and return (deduplicated_df,
    report). The report is the full `analyze_duplicates()` dict, so the
    caller can log total rows, exact duplicates, business-key duplicates,
    retained rows, removed rows, and the reason for removal — nothing is
    dropped without an accompanying count and explanation.

    Never removes a row purely because an identifier repeats.
    """
    report = analyze_duplicates(df, mappings)
    key_cols = report["business_key_columns"]

    if key_cols:
        key_frame = _normalised_key_frame(df, key_cols)
        keep_mask = ~key_frame.duplicated(keep="first")
    else:
        keep_mask = ~df.duplicated(keep="first")

    deduped = df[keep_mask.values].copy()
    # Recomputed from the frame actually returned rather than trusting the
    # predicted count, so the report can never disagree with reality.
    report["rows_retained"] = int(len(deduped))
    report["rows_removed"] = int(len(df) - len(deduped))
    return deduped, report
