"""
query_engine.py
---------------
Answers natural language questions about the sales data.
Uses a rule-based intent parser — no API key required.

Handles questions like:
  "What is total revenue?"
  "Which region has highest sales?"
  "Show top customers"
  "What is the average discount?"
  "How many unique customers?"
  "What are the overdue invoices?"
"""

import re
import pandas as pd
from colorama import Fore, Style
from analytics import get_col


def _clean_q(q: str) -> str:
    return q.lower().strip().rstrip("?")


def answer(question: str, df: pd.DataFrame, mappings: list[dict], kpis: dict) -> str:
    q = _clean_q(question)

    rev_col    = get_col(df, mappings, "net sales")
    qty_col    = get_col(df, mappings, "quantity")
    cust_col   = get_col(df, mappings, "customer code")
    region_col = get_col(df, mappings, "region")
    date_col   = get_col(df, mappings, "invoice date")
    disc_col   = get_col(df, mappings, "discount amount")
    disc_pct   = get_col(df, mappings, "discount percent")
    prod_col   = get_col(df, mappings, "product category")
    sp_col     = get_col(df, mappings, "salesperson")
    stat_col   = get_col(df, mappings, "payment status")
    tot_col    = get_col(df, mappings, "total invoice")

    # ── total revenue ─────────────────────────────────────────────────────────
    if any(w in q for w in ["total revenue", "total sales", "overall revenue", "how much revenue"]):
        val = kpis.get("total_revenue", df[rev_col].sum() if rev_col else 0)
        return f"Total revenue is ₹{val:,.2f}"

    # ── average order value ───────────────────────────────────────────────────
    if any(w in q for w in ["average order", "avg order", "mean order", "average sale"]):
        val = kpis.get("avg_order_value", 0)
        return f"Average order value is ₹{val:,.2f}"

    # ── best / highest region ─────────────────────────────────────────────────
    if any(w in q for w in ["best region", "highest region", "top region", "which region"]):
        if region_col and rev_col:
            top = df.groupby(region_col)[rev_col].sum().idxmax()
            val = df.groupby(region_col)[rev_col].sum().max()
            return f"Highest performing region is {Fore.GREEN}{top}{Style.RESET_ALL} with revenue ₹{val:,.2f}"

    # ── region breakdown ──────────────────────────────────────────────────────
    if "region" in q and any(w in q for w in ["breakdown", "performance", "all", "compare", "show"]):
        if "revenue_by_region" in kpis:
            lines = [f"Revenue by region:"]
            for reg, val in kpis["revenue_by_region"].items():
                lines.append(f"  {reg:<12} ₹{val:>15,.2f}")
            return "\n".join(lines)

    # ── top customers ─────────────────────────────────────────────────────────
    if any(w in q for w in ["top customer", "best customer", "highest customer"]):
        if "top_5_customers" in kpis:
            lines = ["Top 5 customers by revenue:"]
            for i, (cust, val) in enumerate(kpis["top_5_customers"].items(), 1):
                lines.append(f"  {i}. {cust:<12}  ₹{val:>15,.2f}")
            return "\n".join(lines)

    # ── unique customers ──────────────────────────────────────────────────────
    if any(w in q for w in ["how many customer", "unique customer", "number of customer"]):
        val = kpis.get("unique_customers", df[cust_col].nunique() if cust_col else 0)
        return f"There are {val} unique customers in the data"

    # ── total units ───────────────────────────────────────────────────────────
    if any(w in q for w in ["total unit", "units sold", "total quantity", "how many unit"]):
        val = kpis.get("total_units_sold", df[qty_col].sum() if qty_col else 0)
        return f"Total units sold: {val:,}"

    # ── discount ─────────────────────────────────────────────────────────────
    if any(w in q for w in ["total discount", "how much discount"]):
        val = kpis.get("total_discount_given", 0)
        pct = kpis.get("discount_as_pct_of_revenue", 0)
        return f"Total discount given: ₹{val:,.2f} ({pct}% of revenue)"

    if any(w in q for w in ["average discount", "avg discount", "mean discount"]):
        if disc_pct:
            val = round(df[disc_pct].mean(), 2)
            return f"Average discount percentage: {val}%"

    # ── overdue ───────────────────────────────────────────────────────────────
    if any(w in q for w in ["overdue", "pending payment", "unpaid"]):
        overdue = kpis.get("overdue_amount", 0)
        if stat_col:
            counts = df[stat_col].value_counts().to_dict()
            overdue_count = counts.get("OVERDUE", 0)
            pending_count = counts.get("PENDING", 0)
            return (f"Overdue invoices: {overdue_count} invoices totalling ₹{overdue:,.2f}\n"
                    f"Pending invoices: {pending_count}")

    # ── product / category breakdown ──────────────────────────────────────────
    if any(w in q for w in ["product", "category", "which product", "top product"]):
        if "revenue_by_category" in kpis:
            lines = ["Revenue by product category:"]
            for cat, val in kpis["revenue_by_category"].items():
                lines.append(f"  {cat:<15} ₹{val:>15,.2f}")
            return "\n".join(lines)

    # ── salesperson ───────────────────────────────────────────────────────────
    if any(w in q for w in ["salesperson", "sales rep", "who sold", "top salesperson"]):
        if "revenue_by_salesperson" in kpis:
            lines = ["Revenue by salesperson:"]
            for sp, val in kpis["revenue_by_salesperson"].items():
                lines.append(f"  {sp:<10} ₹{val:>15,.2f}")
            return "\n".join(lines)

    # ── monthly trend ─────────────────────────────────────────────────────────
    if any(w in q for w in ["monthly", "month by month", "trend", "over time"]):
        if "monthly_revenue" in kpis:
            lines = ["Monthly revenue:"]
            for month, val in list(kpis["monthly_revenue"].items())[-6:]:
                lines.append(f"  {month}   ₹{val:>15,.2f}")
            return "\n".join(lines)

    # ── total invoices ────────────────────────────────────────────────────────
    if any(w in q for w in ["how many invoice", "total invoice", "number of invoice"]):
        return f"Total invoices in dataset: {len(df):,}"

    # ── fallback ──────────────────────────────────────────────────────────────
    return (
        f"{Fore.YELLOW}I couldn't match that question to the data.{Style.RESET_ALL}\n"
        f"  Try asking: 'total revenue', 'top customers', 'region breakdown',\n"
        f"  'monthly trend', 'overdue invoices', 'revenue by product', 'total discount'"
    )
