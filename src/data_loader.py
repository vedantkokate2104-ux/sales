"""
data_loader.py
--------------
Loads the ERP CSV export and extracts column metadata
for the schema mapping engine.
"""

import pandas as pd
from colorama import Fore, Style


def load_erp_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Loaded {Fore.CYAN}{len(df)}{Style.RESET_ALL} rows, "
          f"{Fore.CYAN}{len(df.columns)}{Style.RESET_ALL} columns from {filepath}")
    return df


def extract_column_info(df: pd.DataFrame) -> list[dict]:
    """
    For each column extract: name, dtype, sample values, null count.
    This context helps the schema mapper make better decisions.
    """
    columns = []
    for col in df.columns:
        sample_vals = df[col].dropna().head(5).tolist()
        columns.append({
            "column_name":   col,
            "data_type":     str(df[col].dtype),
            "sample_values": sample_vals,
            "null_count":    int(df[col].isnull().sum()),
            "unique_count":  int(df[col].nunique()),
        })
    return columns
