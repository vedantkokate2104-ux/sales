# Sales Analysis Agent — Proof of Concept

A fully working demo of the Sales Analysis Agent.
**No Azure account. No API key. No internet connection needed.**

---

## What this POC demonstrates

| Step | What it does |
|------|-------------|
| 1 | Loads raw ERP data (619 rows, 17 cryptic columns) |
| 2 | Schema AI maps column names to business meanings automatically |
| 3 | Feedback loop — shows how a human correction is remembered |
| 4 | Data cleaning — fixes nulls, removes duplicates, flags outliers |
| 5 | KPI engine — revenue, top customers, region breakdown, product performance |
| 6 | Anomaly detection — unusual order values, weak regions, excessive discounts |
| 7 | Sales forecast — projects next 3 months |
| 8 | Natural language queries — ask plain English questions, get real answers |

---

## Setup (2 minutes)

**Step 1 — Install dependencies**
```
pip install -r requirements.txt
```

**Step 2 — Run the POC**
```
cd src
python main.py
```

That is all. No configuration needed.

---

## Try these questions in the chat session

```
What is the total revenue?
Which region has the highest sales?
Show top customers
Revenue by product
How many unique customers?
What are the overdue invoices?
Show monthly trend
Total discount given
Average discount percentage
How many invoices?
Revenue by salesperson
```

---

## Project structure

```
sales_agent_poc/
├── data/
│   └── erp_export.csv       ← 619-row sample ERP dataset (17 columns)
├── src/
│   ├── main.py              ← run this
│   ├── schema_mapper.py     ← column name → business meaning engine
│   ├── data_loader.py       ← CSV loader and column inspector
│   ├── data_cleaner.py      ← cleaning pipeline
│   ├── analytics.py         ← KPIs, anomaly detection, forecasting
│   └── query_engine.py      ← natural language query handler
└── requirements.txt
```

---

## How this maps to the full production system

| POC (this project)               | Production (Azure)                     |
|----------------------------------|----------------------------------------|
| Rule-based synonym mapper        | Azure OpenAI GPT-4o                    |
| JSON file memory                 | Azure AI Search                        |
| CSV file upload                  | Azure Data Factory → live ERP          |
| Pandas in-memory                 | Azure SQL Database                     |
| Simple linear trend forecast     | Prophet / XGBoost                      |
| Terminal chat interface          | React dashboard + chat UI              |

The core logic — schema mapping, cleaning, KPI calculation, anomaly detection,
natural language queries — is identical in both versions.

Switching to Azure OpenAI requires changing approximately 10 lines in schema_mapper.py.

---

## Sample output

```
╔══════════════════════════════════════════════════════╗
║       SALES ANALYSIS AGENT — PROOF OF CONCEPT       ║
╚══════════════════════════════════════════════════════╝

STEP 2 — Schema AI: Mapping Column Names

  ERP Column    Business Meaning        Confidence
  inv_dt        Invoice Date            97%
  cust_nm       Customer Name           97%
  net_val       Net Sales Value         97%
  disc_pct      Discount Percentage     97%
  disc_val      Discount Amount         97%
  reg_cd        Region                  97%
  pmt_stat      Payment Status          97%

STEP 5 — KPI Calculation

  Total Revenue                  ₹47,383,014.61
  Avg Order Value                ₹78,971.69
  Unique Customers               15
  Total Discount Given           ₹4,897,331.03

STEP 6 — Anomaly Detection

  [HIGH]   Unusually High Order Value — 4 invoices above normal range
  [MEDIUM] Excessive Discounting — 22 invoices above 12% discount
```
