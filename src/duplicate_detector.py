"""
duplicate_detector.py

Confidence-based duplicate detection.

Produces:
[
    {
        "inv_no": "INV1001",
        "confidence": 96.2,
        "match_type": "EXACT",
        "rows": [...]
    }
]
"""

import pandas as pd


def _normalize(v):
    if pd.isna(v):
        return ""

    if isinstance(v, float):
        return round(v, 2)

    return str(v).strip().lower()


def compare_rows(row_a, row_b, columns):
    """
    Returns confidence score between two rows.
    """

    if not columns:
        return 0

    matches = 0
    total = 0

    for col in columns:
        total += 1

        a = _normalize(row_a[col])
        b = _normalize(row_b[col])

        if a == b:
            matches += 1

    return round((matches / total) * 100, 2)


def classify(score):

    if score >= 95:
        return "EXACT"

    if score >= 75:
        return "POSSIBLE"

    return "REVIEW"


def build_duplicate_groups(df, id_col):
    """
    Build duplicate groups with confidence.
    """

    dup_mask = df.duplicated(subset=[id_col], keep=False)

    dup_df = df[dup_mask].copy()

    results = []

    for inv_no, group in dup_df.groupby(id_col):

        rows = group.to_dict("records")

        if len(rows) < 2:
            continue

        comparison_cols = [
            c for c in group.columns
            if c != id_col
        ]

        scores = []

        base = rows[0]

        for r in rows[1:]:
            scores.append(
                compare_rows(
                    base,
                    r,
                    comparison_cols
                )
            )

        confidence = round(sum(scores) / len(scores), 2)

        results.append({
            "inv_no": str(inv_no),
            "confidence": confidence,
            "match_type": classify(confidence),
            "rows": rows
        })

    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return results