"""
main.py
-------
Sales Analysis Agent — Proof of Concept
Runs the full pipeline:
  1. Load ERP data
  2. Map column names to business meaning (Schema AI)
  3. Clean the data
  4. Calculate KPIs
  5. Detect anomalies
  6. Forecast next 3 months
  7. Interactive natural language query session

No Azure or OpenAI key required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from colorama import Fore, Style, init
init(autoreset=True)

from data_loader   import load_erp_data, extract_column_info
from schema_mapper import map_all_columns, correct_mapping
from data_cleaner  import clean, print_cleaning_report
from analytics     import calculate_kpis, detect_anomalies, run_forecast, print_kpis
from query_engine  import answer

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════╗
║       SALES ANALYSIS AGENT — PROOF OF CONCEPT       ║
║              Azure-Ready  |  No API Key Demo         ║
╚══════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""

SECTION = lambda title: print(f"\n{Fore.WHITE}{'─'*55}\n  {title}\n{'─'*55}{Style.RESET_ALL}")


def run():
    print(BANNER)

    # ── STEP 1: Load data ─────────────────────────────────────────────────────
    SECTION("STEP 1 — Loading ERP Data")
    df = load_erp_data("../data/erp_export.csv")

    # ── STEP 2: Schema mapping ────────────────────────────────────────────────
    SECTION("STEP 2 — Schema AI: Mapping Column Names")
    print(f"  {Fore.YELLOW}Analysing {len(df.columns)} ERP columns...{Style.RESET_ALL}\n")
    columns = extract_column_info(df)
    mappings = map_all_columns(columns)

    # show confidence summary
    high   = sum(1 for m in mappings if m["confidence"] >= 0.85)
    medium = sum(1 for m in mappings if 0.65 <= m["confidence"] < 0.85)
    low    = sum(1 for m in mappings if m["confidence"] < 0.65)
    print(f"\n  Confidence summary: "
          f"{Fore.GREEN}{high} high{Style.RESET_ALL}  "
          f"{Fore.YELLOW}{medium} medium{Style.RESET_ALL}  "
          f"{Fore.RED}{low} low{Style.RESET_ALL}")

    # ── STEP 3: Demonstrate correction ───────────────────────────────────────
    SECTION("STEP 3 — Feedback Loop: Teaching a Correction")
    print(f"  {Fore.YELLOW}Simulating a user correction...{Style.RESET_ALL}")
    print(f"  User says: 'cust_nm actually contains Customer Name, not a city'")
    correct_mapping("cust_nm", "Customer Name")

    # ── STEP 4: Clean data ────────────────────────────────────────────────────
    SECTION("STEP 4 — Data Cleaning")
    df_clean, report = clean(df, mappings)
    print_cleaning_report(report)

    # ── STEP 5: KPIs ─────────────────────────────────────────────────────────
    SECTION("STEP 5 — KPI Calculation")
    kpis = calculate_kpis(df_clean, mappings)
    print_kpis(kpis)

    # region and category breakdown
    if "revenue_by_region" in kpis:
        print(f"\n  {'Revenue by Region'}")
        print(f"  {'─'*35}")
        for reg, val in kpis["revenue_by_region"].items():
            bar_len = int(val / max(kpis["revenue_by_region"].values()) * 25)
            bar = "█" * bar_len
            print(f"  {reg:<10} {Fore.CYAN}{bar:<25}{Style.RESET_ALL} ₹{val:,.0f}")

    if "revenue_by_category" in kpis:
        print(f"\n  {'Revenue by Product Category'}")
        print(f"  {'─'*35}")
        for cat, val in kpis["revenue_by_category"].items():
            bar_len = int(val / max(kpis["revenue_by_category"].values()) * 25)
            bar = "█" * bar_len
            print(f"  {cat:<15} {Fore.CYAN}{bar:<25}{Style.RESET_ALL} ₹{val:,.0f}")

    # ── STEP 6: Anomaly detection ─────────────────────────────────────────────
    SECTION("STEP 6 — Anomaly Detection")
    anomalies = detect_anomalies(df_clean, mappings)
    if anomalies:
        for a in anomalies:
            sev_color = Fore.RED if a["severity"] == "HIGH" else Fore.YELLOW
            print(f"  {sev_color}[{a['severity']}]{Style.RESET_ALL} {a['type']}")
            print(f"         {a['description']}\n")
    else:
        print(f"  {Fore.GREEN}No anomalies detected.{Style.RESET_ALL}")

    # ── STEP 7: Forecast ──────────────────────────────────────────────────────
    SECTION("STEP 7 — Sales Forecast (Next 3 Months)")
    forecast = run_forecast(df_clean, mappings)
    if forecast and "error" not in forecast:
        for month, val in forecast.items():
            print(f"  {month}   →   {Fore.GREEN}₹{val:,.2f} (projected){Style.RESET_ALL}")
    else:
        print(f"  {forecast.get('error', 'Forecast unavailable')}")

    # ── STEP 8: Natural language query session ────────────────────────────────
    SECTION("STEP 8 — Natural Language Query (type 'exit' to quit)")
    print(f"  {Fore.YELLOW}Ask any business question about the sales data.{Style.RESET_ALL}")
    print(f"  Examples:")
    print(f"    • What is the total revenue?")
    print(f"    • Which region has the highest sales?")
    print(f"    • Show top customers")
    print(f"    • Revenue by product")
    print(f"    • How many unique customers?")
    print(f"    • What are the overdue invoices?")
    print(f"    • Show monthly trend\n")

    while True:
        try:
            q = input(f"  {Fore.CYAN}You:{Style.RESET_ALL} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ["exit", "quit", "q", "bye"]:
            print(f"\n  {Fore.GREEN}POC demo complete. ✓{Style.RESET_ALL}\n")
            break
        result = answer(q, df_clean, mappings, kpis)
        print(f"\n  {Fore.GREEN}Agent:{Style.RESET_ALL} {result}\n")


if __name__ == "__main__":
    run()
