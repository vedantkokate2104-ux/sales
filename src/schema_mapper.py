"""
schema_mapper.py
----------------
Maps ERP column names to business meanings using a two-stage pipeline:

Stage 1 (fast, always runs):
    Rule-based synonym dictionary lookup with sample value hints.

Stage 2 (semantic, runs if model available):
    E5-Large-v2 embedding similarity + hybrid heuristics via embedding_mapper.py.

Final confidence is a weighted combination of both stages.
If the embedding model is unavailable, Stage 1 runs alone — identical
behaviour to the original implementation.

Public API is unchanged, but the return schema now includes 'reasons':
{
    "original_column": str,
    "business_name":   str,
    "category":        str,
    "description":     str,
    "confidence":      float,
    "human_corrected": bool,
    "source":          str,
    "reasons":         list[str]
}
"""

import re
import json
import os
import logging
from colorama import Fore, Style

logger = logging.getLogger(__name__)

# ── synonym dictionary ────────────────────────────────────────────────────────
# covers the most common ERP column naming patterns seen in Indian ERPs
# (Tally, Odoo, SAP, custom systems)

SYNONYM_MAP = {
    # dates
    "invoice date":         ["inv_dt", "inv_date", "invoice_dt", "bill_dt", "bill_date", "doc_dt", "trans_dt", "date"],
    "order date":           ["ord_dt", "order_dt", "ord_date", "po_dt"],
    "delivery date":        ["del_dt", "dlv_dt", "delivery_dt", "ship_dt"],
    "due date":             ["due_dt", "due_date", "pay_due_dt", "mat_dt"],

    # customer
    "customer code":        ["cust_cd", "cust_id", "customer_id", "cust_no", "client_cd", "party_cd"],
    "customer name":        ["cust_nm", "cust_name", "customer_nm", "client_nm", "party_nm", "buyer_nm"],
    "customer type":        ["cust_typ", "client_type", "party_type",
                             "cust_seg", "customer_segment", "cust_segment", "client_seg", "cust_tier"],

    # product
    "product id":           ["prod_id", "item_id", "sku", "product_cd", "item_cd", "mat_cd", "article_no"],
    "product name":         ["prod_nm", "item_nm", "product_name", "item_desc", "mat_desc", "article_nm"],
    "product category":     ["prod_desc", "cat_cd", "category", "prod_cat", "item_cat", "grp_cd",
                             "prod_grp", "product_group", "item_grp", "prod_group", "cat_nm", "prod_type"],

    # amounts
    "gross amount":         ["gross_amt", "gross_val", "gross", "bef_disc", "pre_disc_amt"],
    "net sales value":      ["net_val", "net_amt", "net_sales", "nett_val", "nsv", "net_revenue",
                             "rev", "revenue", "rev_val", "sales_val", "net_rev"],
    "total invoice value":  ["tot_inv_val", "total_amt", "tot_amt", "invoice_val", "bill_amt", "tot_val",
                             "trx_v", "trx_val", "transaction_val", "txn_val", "trxn_v"],
    "unit price":           ["unit_pr", "unit_price", "rate", "price", "mrp", "selling_price", "sp"],
    "cost price":           ["cost_pr", "cost_price", "cp", "purchase_price", "landed_cost"],

    # discount
    "discount percentage":  ["disc_pct", "disc_per", "discount_pct", "disc_%", "rebate_pct"],
    "discount amount":      ["disc_val", "disc_amt", "discount_amt", "rebate_amt", "disc_value"],

    # tax
    "gst amount":           ["gst_amt", "tax_amt", "gst", "igst", "cgst_sgst", "vat_amt"],
    "gst percentage":       ["gst_pct", "tax_pct", "gst_rate", "tax_rate"],

    # quantity
    "quantity sold":        ["qty_sold", "quantity", "qty", "units_sold", "vol", "pieces"],
    "quantity returned":    ["qty_ret", "return_qty", "ret_qty"],

    # geography
    "region":               ["reg_cd", "region_cd", "zone", "area_cd", "territory"],
    "city":                 ["city", "city_nm", "town"],
    "state":                ["state", "state_cd", "state_nm"],

    # invoice
    "invoice number":       ["inv_no", "invoice_no", "bill_no", "doc_no", "voucher_no", "inv_id"],

    # people
    "salesperson code":     ["sp_cd", "sales_rep", "salesman_cd", "rep_cd", "exec_cd", "emp_cd",
                             "slsp", "slsprsn", "sales_person", "salesperson_cd", "sp_nm", "salesman"],

    # margin
    "margin percentage":    ["mrgn_pct", "margin_pct", "margin_per", "gp_pct", "gross_margin_pct",
                             "margin_%", "net_margin", "profit_pct"],

    # delivery / city
    "delivery city":        ["dlv_st", "dlv_city", "delivery_city", "ship_city", "del_city",
                             "delivery_loc", "dlv_loc"],

    # status
    "payment status":       ["pmt_stat", "pay_status", "payment_stat", "inv_status", "bill_status"],
    "order status":         ["ord_stat", "order_status", "status"],
}

# build reverse lookup: column_pattern → business name
_REVERSE = {}
for biz_name, patterns in SYNONYM_MAP.items():
    for p in patterns:
        _REVERSE[p.lower()] = biz_name

# ── thresholds ────────────────────────────────────────────────────────────────
HIGH_CONFIDENCE   = 0.75  # Aligned with new E5 output range
MEDIUM_CONFIDENCE = 0.60  # Aligned with new E5 output range

# ── persistent memory (JSON file) ─────────────────────────────────────────────
_MEMORY_FILE = "./column_memory.json"

def _load_memory() -> dict:
    if os.path.exists(_MEMORY_FILE):
        with open(_MEMORY_FILE) as f:
            return json.load(f)
    return {}

def _save_memory(mem: dict):
    with open(_MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

_memory = _load_memory()


# ── internal helpers (unchanged from original) ────────────────────────────────

def _normalise(col: str) -> str:
    """Lowercase, strip, replace spaces/dashes with underscores."""
    return re.sub(r"[\s\-]+", "_", col.strip().lower())


def _rule_match(col_norm: str) -> tuple[str | None, float | None]:
    """
    Try synonym dictionary lookup.
    Returns (business_name, confidence) or (None, None).
    """
    if col_norm in _REVERSE:
        return _REVERSE[col_norm], 0.97
    for pattern, biz_name in _REVERSE.items():
        if pattern in col_norm or col_norm in pattern:
            return biz_name, 0.82
    return None, None


def _sample_hint(sample_values: list) -> str | None:
    """Infer column type from sample values."""
    if not sample_values:
        return None
    samples = [str(v).lower() for v in sample_values[:3]]
    if any(re.match(r"\d{4}-\d{2}-\d{2}", s) for s in samples):
        return "date"
    if all(s in ["paid", "pending", "overdue", "open", "closed"] for s in samples):
        return "status"
    if all(re.match(r"^[a-z]\d{3,4}$", s) for s in samples):
        return "code/id"
    try:
        nums = [float(v) for v in sample_values[:5]]
        if all(0 <= n <= 100 for n in nums) and all(isinstance(v, float) for v in sample_values[:5]):
            return "percentage"
        if all(n > 100 for n in nums):
            return "amount"
    except (ValueError, TypeError):
        pass
    return None


def _infer_category(biz_name: str, hint: str | None) -> str:
    biz_lower = biz_name.lower()
    if hint == "date" or "date" in biz_lower:              return "date"
    if hint == "percentage" or "percent" in biz_lower:     return "percentage"
    if hint == "status" or "status" in biz_lower:          return "status"
    if hint == "code/id" or any(w in biz_lower for w in ["code", " id", "number"]): return "identifier"
    if any(w in biz_lower for w in ["amount", "value", "revenue", "price", "cost", "gst", "tax"]): return "amount"
    if "quantity" in biz_lower:                            return "quantity"
    if any(w in biz_lower for w in ["customer", "client"]): return "customer"
    if any(w in biz_lower for w in ["product", "item", "sku"]): return "product"
    if any(w in biz_lower for w in ["region", "zone", "city", "state"]): return "geography"
    if any(w in biz_lower for w in ["salesperson", "rep", "employee"]): return "people"
    return "other"


# ── NEW: embedding integration ────────────────────────────────────────────────

def _try_import_embedding_mapper():
    """
    Attempt to import embedding_mapper.
    Returns the module or None if unavailable.
    """
    try:
        import embedding_mapper
        return embedding_mapper
    except ImportError:
        return None


def _map_with_embeddings(column_info: dict, rule_name: str | None, rule_conf: float | None, neighbors: list = None) -> tuple[str, float, str, list]:
    """
    Run the embedding-enhanced mapping for one column.

    Returns (business_name, final_confidence, source_label, reasons_list).
    """
    em = _try_import_embedding_mapper()
    if em is None:
        # sentence-transformers not installed — behave exactly as before
        return rule_name or "", rule_conf or 0.45, "rule engine", ["Used exact/partial synonym match"] if rule_name else []

    # Pass neighbors context into the embedding mapper
    candidates = em.get_top_candidates(column_info, neighbors=neighbors, top_k=3)

    if not candidates:
        return rule_name or "", rule_conf or 0.45, "rule engine (embedding unavailable)", ["Embedding failed; used rule match"] if rule_name else []

    top_candidate  = candidates[0]
    top_name       = top_candidate["business_name"]
    top_similarity = top_candidate["similarity"]
    top_reasons    = top_candidate.get("reasons", [])

    # ── Case 1: Rule engine found a match ─────────────────────────────────────
    if rule_name is not None:
        final_conf = em.combine_confidence(rule_conf, top_similarity)

        if top_name == rule_name:
            source = "hybrid (rule + embedding agreed)"
            reasons = ["Rule engine match"] + top_reasons
        else:
            logger.debug(
                f"Mapping disagreement for '{column_info.get('column_name')}':"
                f" rule='{rule_name}' ({rule_conf:.2f}),"
                f" embedding='{top_name}' ({top_similarity:.2f}). Using rule."
            )
            source = f"rule engine (embedding suggested '{top_name}')"
            reasons = [f"Rule engine override (Embedding mapped this to '{top_name}')"]

        return rule_name, final_conf, source, reasons

    # ── Case 2: Rule engine found nothing → use embedding alone ───────────────
    if top_similarity >= MEDIUM_CONFIDENCE:
        source = f"embedding only ({top_similarity:.0%} confidence)"
        return top_name, top_similarity, source, top_reasons

    # ── Case 3: Embedding confidence too low → readable fallback ─────────────
    readable = _normalise(column_info.get("column_name", "")).replace("_", " ").title()
    hint = _sample_hint(column_info.get("sample_values", []))
    fallback_conf = 0.55 if hint else 0.45
    return readable, fallback_conf, "fallback (low embedding confidence)", ["Similarity too low, used fallback name"]


# ── public API ───────────────────────────────────────────────────────────────

def map_column(column_info: dict, neighbors: list = None) -> dict:
    """
    Map a single column to its business meaning.
    """
    col_orig = column_info["column_name"]
    col_norm = _normalise(col_orig)

    # ── Memory override (unchanged) ───────────────────────────────────────────
    if col_orig in _memory and _memory[col_orig].get("human_corrected"):
        res = {**_memory[col_orig], "source": "memory (human confirmed)"}
        res["reasons"] = ["Loaded from human-corrected memory"]
        return res

    # ── Stage 1: Rule engine ──────────────────────────────────────────────────
    rule_name, rule_conf = _rule_match(col_norm)
    hint = _sample_hint(column_info.get("sample_values", []))

    # ── Stage 2: Embedding + combination ─────────────────────────────────────
    biz_name, confidence, source, reasons = _map_with_embeddings(
        column_info, rule_name, rule_conf, neighbors=neighbors
    )

    if not biz_name:
        biz_name   = col_norm.replace("_", " ").title()
        confidence = 0.55 if hint else 0.45
        source     = "rule engine fallback"
        reasons    = ["No strong semantic match found"]

    category = _infer_category(biz_name, hint)

    result = {
        "original_column": col_orig,
        "business_name":   biz_name.title(),
        "category":        category,
        "description":     f"Mapped from ERP column '{col_orig}'",
        "confidence":      round(confidence, 2),
        "human_corrected": False,
        "source":          source,
        "reasons":         reasons,
    }

    # Save to memory
    _memory[col_orig] = result
    _save_memory(_memory)
    return result


def map_all_columns(columns: list[dict]) -> list[dict]:
    """Map all columns and print a formatted result table."""
    mappings = []

    em = _try_import_embedding_mapper()
    if em is not None:
        em._load_model()
        if em.is_available():
            print(f"  {Fore.CYAN}Embedding model ready — semantic matching active{Style.RESET_ALL}")
        else:
            print(f"  {Fore.YELLOW}Embedding model unavailable — using rule engine only{Style.RESET_ALL}")

    print(f"\n  {'ERP Column':<18} {'Business Meaning':<28} {'Category':<14} {'Confidence':<12} {'Source'}")
    print("  " + "─" * 95)

    all_names = [c.get("column_name", "") for c in columns]

    for i, col in enumerate(columns):
        # Extract immediate neighbors for context
        neighbors = []
        if i > 0:
            neighbors.append(all_names[i-1])
        if i < len(all_names) - 1:
            neighbors.append(all_names[i+1])

        m = map_column(col, neighbors=neighbors)
        mappings.append(m)
        conf_pct = f"{m['confidence']*100:.0f}%"
        conf_color = (
            Fore.GREEN  if m["confidence"] >= HIGH_CONFIDENCE   else
            Fore.YELLOW if m["confidence"] >= MEDIUM_CONFIDENCE else
            Fore.RED
        )
        print(
            f"  {m['original_column']:<18} "
            f"{m['business_name']:<28} "
            f"{m['category']:<14} "
            f"{conf_color}{conf_pct:<12}{Style.RESET_ALL} "
            f"{Fore.CYAN}{m['source']}{Style.RESET_ALL}"
        )

    if em is not None and em.is_available():
        stats = em.cache_stats()
        print(f"\n  Embedding cache: {stats['cached_embeddings']} entries")

    return mappings


def correct_mapping(column_name: str, correct_business_name: str):
    """Human correction — saved permanently with confidence 1.0."""
    existing = _memory.get(column_name, {})
    old_name = existing.get("business_name", "unknown")

    _memory[column_name] = {
        "original_column": column_name,
        "business_name":   correct_business_name,
        "confidence":      1.0,
        "human_corrected": True,
        "source":          "human correction",
        "category":        existing.get("category", "other"),
        "description":     existing.get("description", ""),
        "reasons":         ["Human explicitly corrected this mapping"]
    }
    _save_memory(_memory)

    print(f"\n  {Fore.GREEN}✓ Correction saved{Style.RESET_ALL}")
    print(f"    Column : {column_name}")
    print(f"    Was    : {old_name}")
    print(f"    Now    : {correct_business_name} (confidence: 100%)")