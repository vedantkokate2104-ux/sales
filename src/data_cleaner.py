"""
data_cleaner.py
---------------
Automatically cleans the ERP data:
- fixes missing values
- removes duplicates
- detects and flags outliers
- standardises date formats

Duplicate handling (Phase 1, Step 1) is delegated to transaction_key.py.
It used to be a single `drop_duplicates(subset=[id_cols[0]])` call on
whichever column the schema mapper listed as the first "identifier",
which on the bundled dataset is the invoice number — a column that
dataset RECYCLES across genuinely different transactions. That removed
212 of 392 rows (54% of the data, 57% of revenue) purely because an
identifier string repeated, which is not evidence of duplication. See
transaction_key.py for the three-way distinction (exact duplicate rows /
duplicate business transactions / legitimate repeated identifiers) that
replaced it.
"""

import pandas as pd
import numpy as np
from colorama import Fore, Style

from transaction_key import deduplicate_transactions
from forecast.preprocessing import parse_dates_deterministic


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
    # A repeated identifier is NOT a duplicate transaction — see
    # transaction_key.py. Rows are only removed when they are identical on
    # a composite business key (invoice + date + product + customer +
    # quantity + revenue), or, when no such key can be built from the
    # available columns, when the entire row is identical.
    df, dup_report = deduplicate_transactions(df, mappings)
    dupes_removed = dup_report["rows_removed"]
    # Existing scalar field, unchanged in name and meaning, so
    # ui/app.py and print_cleaning_report() keep working as-is.
    report["duplicates_removed"] = dupes_removed
    # Additive: the full breakdown behind that number, so no row is ever
    # dropped without a visible count and a stated reason.
    report["duplicate_detail"] = dup_report

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
    # Deterministic parsing: ONE explicit format, chosen by testing
    # candidates against the whole column, applied to every row.
    #
    # This previously called a bare pd.to_datetime(df[col], errors="coerce").
    # With no format and no dayfirst hint, pandas parses ambiguous values
    # row by row via dateutil, which resolves each row on its own: in this
    # project's own dataset "03/01/21" became 1 March while "21/01/21"
    # (day > 12, so month-first is impossible) became 21 January — two
    # different conventions inside a single column, silently. That
    # mis-dated 142 of 392 rows (36%), moving transactions between months
    # and corrupting every downstream aggregation, seasonality estimate
    # and forecast period. See forecast/preprocessing.py for the full
    # rationale; that module already used this parser, so the cleaner and
    # the forecasting pipeline now agree on what a date means.
    date_cols = [c for c, cat in col_category.items() if cat == "date" and c in df.columns]
    for col in date_cols:
        df[col] = parse_dates_deterministic(df[col])
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

    # Full duplicate breakdown (Phase 1, Step 1) — total rows, exact
    # duplicates, business-key duplicates, retained/removed rows and the
    # reason, so nothing is ever removed silently.
    dup = report.get("duplicate_detail")
    if dup:
        print(f"  Duplicate analysis:")
        print(f"    total rows              : {dup['total_rows']}")
        print(f"    exact duplicate rows    : {dup['exact_duplicate_rows']}")
        print(f"    business-key duplicates : {dup['business_key_duplicate_rows']}")
        if dup.get("identifier_column"):
            print(f"    repeated '{dup['identifier_column']}' values : "
                  f"{dup['repeated_identifier_values']} "
                  f"({dup['repeated_identifier_rows']} rows)")
            print(f"    {Fore.GREEN}preserved as legitimate : "
                  f"{dup['preserved_repeated_identifier_rows']} rows{Style.RESET_ALL}")
        print(f"    strategy                : {dup['strategy']} ({dup['business_key_basis']})")
        print(f"    retained / removed      : {dup['rows_retained']} / {dup['rows_removed']}")
        print(f"    reason                  : {dup['removal_reason']}")
        if dup.get("limitation"):
            print(f"    {Fore.YELLOW}limitation            : {dup['limitation']}{Style.RESET_ALL}")

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
