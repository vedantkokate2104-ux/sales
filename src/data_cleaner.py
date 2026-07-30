"""
data_cleaner.py
---------------
Automatically cleans the ERP data:
- fixes missing values
- removes duplicates
- detects and flags outliers
- standardises date formats
"""

import pandas as pd
import numpy as np
from colorama import Fore, Style


def clean(df: pd.DataFrame, mappings: list[dict]) -> tuple[pd.DataFrame, dict]:
    """
    Run the full cleaning pipeline.
    Returns (cleaned_df, report).
    """
    report = {}
    original_rows = len(df)
    df = df.copy()

    # build a lookup: original_column → category
    col_category = {m["original_column"]: m["category"] for m in mappings}

    # ── 0. coerce numeric-looking columns to proper numeric types ─────────────
    # This prevents "unsupported operand type str - str" errors when a column
    # is categorised as 'amount' but pandas loaded it as object (string).
    for col in df.columns:
        if col_category.get(col) in ["amount", "quantity", "percentage"]:
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only replace if most values converted successfully (>80%)
            if converted.notna().sum() / max(len(converted), 1) >= 0.8:
                df[col] = converted

    # ── 1. remove duplicates ──────────────────────────────────────────────────
    id_cols = [c for c, cat in col_category.items() if cat == "identifier"]
    if id_cols:
        before = len(df)
        df = df.drop_duplicates(subset=[id_cols[0]], keep="first")
        dupes_removed = before - len(df)
    else:
        df = df.drop_duplicates()
        dupes_removed = original_rows - len(df)
    report["duplicates_removed"] = dupes_removed

    # ── 2. fix missing values ─────────────────────────────────────────────────
    nulls_fixed = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
        cat = col_category.get(col, "other")
        if cat == "amount":
            # Only apply numeric median if column is actually numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                nulls_fixed[col] = f"{null_count} nulls → filled with median"
            else:
                df[col] = df[col].fillna("UNKNOWN")
                nulls_fixed[col] = f"{null_count} nulls → filled with UNKNOWN"
        elif cat in ["customer", "product", "geography", "identifier", "people"]:
            df[col] = df[col].fillna("UNKNOWN")
            nulls_fixed[col] = f"{null_count} nulls → filled with UNKNOWN"
        elif cat == "date":
            df[col] = df[col].ffill()
            nulls_fixed[col] = f"{null_count} nulls → forward filled"
        elif cat == "percentage":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                nulls_fixed[col] = f"{null_count} nulls → filled with median"
            else:
                df[col] = df[col].fillna(0)
                nulls_fixed[col] = f"{null_count} nulls → filled with 0"
        else:
            df[col] = df[col].fillna("N/A")
            nulls_fixed[col] = f"{null_count} nulls → filled with N/A"
    report["nulls_fixed"] = nulls_fixed

    # ── 3. standardise dates ──────────────────────────────────────────────────
    date_cols = [c for c, cat in col_category.items() if cat == "date" and c in df.columns]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    report["date_cols_standardised"] = date_cols

    # ── 4. detect outliers in amount columns (numeric only) ───────────────────
    amount_cols = [
        c for c, cat in col_category.items()
        if cat == "amount"
        and c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    outliers_flagged = {}
    for col in amount_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 4 * iqr
        mask = df[col] > upper
        count = mask.sum()
        if count > 0:
            df[f"{col}_outlier_flag"] = mask.astype(int)
            outliers_flagged[col] = int(count)
    report["outliers_flagged"] = outliers_flagged

    report["rows_before"] = original_rows
    report["rows_after"]  = len(df)

    return df, report


def print_cleaning_report(report: dict):
    print(f"\n  {'─'*55}")
    print(f"  Rows before  : {report['rows_before']}")
    print(f"  Rows after   : {report['rows_after']}")
    print(f"  Duplicates   : {Fore.YELLOW}{report['duplicates_removed']} removed{Style.RESET_ALL}")

    if report["nulls_fixed"]:
        print(f"  Nulls fixed  :")
        for col, action in report["nulls_fixed"].items():
            print(f"    {Fore.CYAN}{col}{Style.RESET_ALL} → {action}")

    if report["outliers_flagged"]:
        print(f"  Outliers     :")
        for col, count in report["outliers_flagged"].items():
            print(f"    {Fore.RED}{col}{Style.RESET_ALL} → {count} rows flagged")

    if report["date_cols_standardised"]:
        print(f"  Dates parsed : {', '.join(report['date_cols_standardised'])}")
    print(f"  {'─'*55}")
