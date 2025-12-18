Automated Income Statement Extraction & Financial Performance Analysis
Overview
This project is a Python-based analytics engine designed to automatically extract, normalize and analyse income statements from audited annual report PDFs. It addresses a common real world finance problem: financial disclosures are often unstructured, inconsistently formatted, and not machine readable.
Rather than relying on a single brittle rule, the system uses multi-layer heuristic detection, combining semantic cues, structural analysis, table extraction and word level numeric recovery to reliably identify and reconstruct income statements across different report layouts.
The output is a clean, year-by-year financial dataset with profitability metrics, growth analysis and publication-ready visualisations.
Key Capabilities
🔹 Intelligent Income Statement Detection
•	Semantic identification of income statement pages using heading keywords
•	Structural scoring based on text density, numeric density and layout patterns
•	Automatic handling of vertically and horizontally oriented tables
🔹 Robust Data Extraction
•	Primary extraction via table detection from PDFs
•	Fallback word-level numeric reconstruction when tables are missing or malformed
•	Defensive handling of noisy, partial or fragmented disclosures
🔹 Financial Data Normalisation
•	Standardisation of numeric formats (thousands separators, parentheses negatives)
•	Intelligent mapping of key income statement lines:
o	Revenue / Turnover
o	Gross Profit
o	Operating Profit
o	Profit Before Tax
o	Net Income
🔹 Financial Analytics
•	Profitability margins (gross, operating, net)
•	Conversion ratios (e.g. GP-to-OP efficiency)
•	Year-on-year growth analysis
•	Margin spread analysis
🔹 Visual Analytics
•	Trend visualisation using matplotlib
•	Regression overlays with R² statistics
•	One figure per chart discipline for clarity and academic reporting

Project Structure
project-root/
│
├── data/                  # Annual report PDFs (one per year)
├── src/                   # Core extraction and analytics logic
│   ├── extractor.py       # PDF parsing and table/word extraction
│   ├── classifier.py      # Income statement detection & scoring
│   ├── cleaner.py         # Data cleaning and normalisation
│   ├── analytics.py       # Margin, growth, and ratio calculations
│   └── visualisation.py   # Charts and trend analysis
│
├── outputs/               # Cleaned datasets and generated figures
├── main.py                # End-to-end pipeline runner
├── requirements.txt       # Python dependencies
└── README.md

Technologies Used
•	Python
•	pandas – data manipulation and aggregation
•	numpy – numerical operations
•	pdfplumber – PDF parsing and table extraction
•	matplotlib – financial and statistical visualisation

Use Case
This project is suitable for:
•	Financial statement analysis and benchmarking
•	Strategy and performance analytics
•	Research in accounting, finance and business analytics
•	Situations where financial data is locked in unstructured PDF reports
It is particularly relevant for consulting, investment analysis, corporate finance and academic research where automation and data reliability are critical.

How It Works (High-Level)
1.	Discover and load annual report PDFs
2.	Identify candidate income statement pages using semantic and structural heuristics
3.	Extract financial tables or reconstruct numeric rows from word-level data
4.	Normalise and map key financial line items
5.	Aggregate multi-year data
6.	Compute profitability, growth and efficiency metrics
7.	Generate visual analytics

Motivation
The motivation behind this project was to move beyond textbook datasets and work with messy, real financial disclosures as they exist in practice. The goal was to demonstrate:
•	Applied data engineering skills
•	Financial statement literacy
•	Robust analytical thinking under real-world constraints

Disclaimer
This project is for academic research (The influence of dynamic capabilities on competitive advantage: A conceptual model for South African Fast- Moving) consumer goods and portfolio demonstration purposes. While care has been taken to ensure robustness and accuracy, results should be independently verified before use in professional or investment contexts.

Author
Developed as part of an academic research project and professional portfolio in quantitative finance, business analytics, and strategic analysis.






Automates extraction, normalization and analysis of income statements from PDF annual reports. Generates clean multi-year datasets, key profitability metrics, growth analysis and visualizations using Python, pandas, pdfplumber and matplotlib for research and portfolio purposes.
