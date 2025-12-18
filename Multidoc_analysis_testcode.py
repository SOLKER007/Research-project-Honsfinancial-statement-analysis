import re
from datetime import datetime
import numpy as np
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
import traceback  # for detailed error traces

# ======= CONFIG =======
# (Optional) Single file path. You can still keep this, but multi-file run below uses pdf_paths.
#  pdf_path = r"C:\Users\mwila\Downloads\Woolies Financials\picknpay-audited-annual-financial-statements-2018.pdf"

# ======= MULTI-FILE CONFIG =======
# Option A: Explicit mapping of year/label -> PDF path
# pdf_paths = {
#     "2016": r"C:\Users\mwila\Downloads\Woolies Financials\picknpay-audited-annual-financial-statements-2016.pdf",
#     "2017": r"C:\Users\mwila\Downloads\Woolies Financials\picknpay-audited-annual-financial-statements-2017.pdf",
#     "2018": r"C:\Users\mwila\Downloads\Woolies Financials\picknpay-audited-annual-financial-statements-2018.pdf",
# }

# Option B: Auto-discover all PDFs in a folder and infer year from filename
pdf_paths = {}
for p in glob.glob(r"C:\Users\mwila\Downloads\Woolies Financials\Woolworths\*.pdf"):
    m = re.search(r"(19|20)\d{2}", os.path.basename(p))
    key = m.group(0) if m else os.path.splitext(os.path.basename(p))[0]
    pdf_paths[key] = p

# -----------------------
# 0) Find pages with the income statement heading
# -----------------------
INCOME_TITLES = [
    r"\bGROUP\s+STATEMENT\s+OF\s+COMPREHENSIVE\s+INCOME\b",
    r"\bSTATEMENT\s+OF\s+PROFIT\s+OR\s+LOSS\b",
    r"\b(INCOME|COMPREHENSIVE\s+INCOME)\s+STATEMENT\b",
]

def find_income_pages(pdf):
    hits = []
    for i, page in enumerate(pdf.pages):
        try:
            txt = (page.extract_text() or "").upper()
        except Exception:
            txt = ""
        if any(re.search(pat, txt, flags=re.I) for pat in INCOME_TITLES):
            hits.append(i)  # zero-based
    return hits

# -----------------------
# 1) Extract tables near those pages (2 variants per table)
# -----------------------
def make_variants(table):
    """Return two variants: (A) guessed header row, (B) raw with generic headers."""
    raw = pd.DataFrame(table)

    # A) header-guess (first dense row up to 4 tries)
    df = raw.copy()
    header_idx = 0
    for i in range(min(4, len(df))):
        row = df.iloc[i]
        if row.notna().sum() >= max(2, int(0.6 * len(row))):
            header_idx = i
            break
    try:
        df.columns = df.iloc[header_idx].astype(str)
        df = df.iloc[header_idx + 1:].reset_index(drop=True)
    except Exception:
        df = raw.copy()  # fallback

    # B) raw, fixed names
    raw2 = raw.copy()
    raw2.columns = [f"col_{i}" for i in range(raw2.shape[1])]
    return [df, raw2]

def extract_candidate_tables(pdf, target_pages, lookahead=4):  # lookahead widened
    all_tables = []
    considered = set()
    for p in target_pages:
        for j in range(p, min(len(pdf.pages), p + 1 + lookahead)):
            if j in considered:
                continue
            considered.add(j)
            page = pdf.pages[j]
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            if tables:
                print(f"✅ Found {len(tables)} table(s) on page {j+1}")
            for t in tables:
                if not t:
                    continue
                for cand in make_variants(t):
                    if cand.shape[0] >= 2 and cand.shape[1] >= 2:
                        all_tables.append(cand)
    return all_tables

# -----------------------
# 2) Classifier + scoring
# -----------------------
income_cues = [
    "group statement of comprehensive income",
    "revenue", "turnover", "cost of sales", "gross profit",
    "operating profit", "profit before tax", "profit for the year",
    "earnings per share", "headline earnings per share",
]

def table_text(df, max_cells=500):
    cols = " ".join(map(str, df.columns))
    sample = df.astype(str).stack().head(max_cells)
    return (cols + " " + " ".join(sample)).lower()

def score_text(text, cues):
    s = 0
    for c in cues:
        if c in text:
            s += 3 if " " in c else 1
    return s

def score_structural(df):
    num_cols = 0
    year_hits = 0
    nrows = len(df)
    for c in df.columns:
        col = df[c] if isinstance(df[c], pd.Series) else pd.Series(df[c])
        s = pd.to_numeric(col, errors="coerce")
        if s.notna().sum() >= max(2, int(0.4 * nrows)):
            num_cols += 1
        if re.search(r"\b20\d{2}\b", str(c)):
            year_hits += 1
    return 0.5 * num_cols + 2 * year_hits

def score_textiness(df, cols_to_check=3):
    """How much human text exists in the first few columns (no applymap)."""
    if df.empty:
        return 0.0
    take = min(cols_to_check, df.shape[1])
    block = df.iloc[:, :take].astype(str)
    has_letters_cols = [
        col.str.contains(r"[A-Za-z]", regex=True, na=False) for _, col in block.items()
    ]
    if not has_letters_cols:
        return 0.0
    has_letters = pd.concat(has_letters_cols, axis=1)
    return has_letters.values.sum() / (block.shape[0] * block.shape[1])

def classify_table(df):
    """Return (label, score)."""
    txt = table_text(df)
    score = score_text(txt, income_cues) + score_structural(df) + 6.0 * score_textiness(df)
    return ("income_statement" if score >= 6 else "unknown", score)

# -----------------------
# 3) Cleaning utilities  (label-aware + transpose heuristic)
# -----------------------
def _dedupe_names(cols):
    seen, out = {}, []
    for c in cols:
        name = str(c) if c is not None else ""
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
    return out

def _to_numeric_series(s: pd.Series) -> pd.Series:
    txt = s.astype(str)
    neg = txt.str.contains(r"\(", regex=True, na=False)
    txt = txt.str.replace(r"[(),]", "", regex=True).str.replace(" ", "", regex=False)
    txt = txt.str.replace(r"[^\d\.\-\+]", "", regex=True)
    out = pd.to_numeric(txt, errors="coerce")
    out[neg] = -out[neg]
    return out

def _first_text_col(df: pd.DataFrame) -> int:
    best_j, best_score = 0, -1.0
    n = len(df)
    for j in range(df.shape[1]):
        col = df.iloc[:, j].astype(str)
        non_null = (col.str.strip() != "").sum() / max(1, n)
        has_letters = col.str.contains(r"[A-Za-z]", regex=True, na=False).sum() / max(1, n)
        numeric = pd.to_numeric(col, errors="coerce").notna().sum() / max(1, n)
        score = has_letters * 1.2 + non_null * 0.8 - numeric * 1.0
        if score > best_score:
            best_score = score
            best_j = j
    return best_j

def _row_label_from_cells(row: pd.Series, max_cols=5) -> str:
    take = min(max_cols, len(row))
    cells = [str(row.iloc[i]) if i < len(row) else "" for i in range(take)]
    for cell in cells:
        if re.search(r"[A-Za-z]", cell or ""):
            lab = cell
            break
    else:
        lab = " ".join(cells)
    lab = re.sub(r"\s+", " ", str(lab)).strip()
    return lab[:200]

def clean_statement(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if df.shape[1] < 2 or df.shape[0] < 1:
        return df
    df.columns = _dedupe_names(df.columns)

    def _coerce_numeric_cols(dfin: pd.DataFrame) -> pd.DataFrame:
        out = dfin.copy()
        for c in list(out.columns):
            ser = _to_numeric_series(out[c])
            if ser.notna().sum() >= max(1, int(0.4 * len(ser))):
                out[c] = ser
        return out

    label_col_idx = _first_text_col(df)
    if label_col_idx != 0:
        cols = list(df.columns)
        cols = [cols[label_col_idx]] + cols[:label_col_idx] + cols[label_col_idx+1:]
        df = df[cols]

    labels = df.apply(_row_label_from_cells, axis=1)
    labels = labels.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    empty_ratio = (labels.eq("") | labels.eq("None")).mean()
    if empty_ratio > 0.5 and df.shape[1] >= 2:
        labels2 = (df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str)).str.replace(r"\s+", " ", regex=True).str.strip()
        labels = labels.where(labels.ne(""), labels2)

    num_df = _coerce_numeric_cols(df.iloc[:, 1:].copy())
    num_df.index = labels
    num_df.index.name = "Item"
    num_df = num_df.dropna(how="all", axis=1)

    lab_index = pd.Index(num_df.index).astype(str)
    lab_is_num = pd.to_numeric(lab_index, errors="coerce").notna().mean()
    empty_ratio = pd.Series(lab_index).str.strip().eq("").mean()
    lab_is_empty = empty_ratio > 0.5

    if (lab_is_num > 0.6 or lab_is_empty) and num_df.shape[1] >= 2:
        flipped = df.T.reset_index(drop=False)
        flipped.columns = _dedupe_names(flipped.columns)
        label_col_idx = _first_text_col(flipped)
        cols = list(flipped.columns)
        if label_col_idx != 0:
            cols = [cols[label_col_idx]] + cols[:label_col_idx] + cols[label_col_idx+1:]
            flipped = flipped[cols]
        labels2 = flipped.apply(_row_label_from_cells, axis=1).astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        num_df2 = _coerce_numeric_cols(flipped.iloc[:, 1:].copy())
        num_df2.index = labels2
        num_df2.index.name = "Item"
        num_df2 = num_df2.dropna(how="all", axis=1)

        lab2_index = pd.Index(num_df2.index).astype(str)
        lab2_is_num = pd.to_numeric(lab2_index, errors="coerce").notna().mean()
        lab2_empty_ratio = pd.Series(lab2_index).str.strip().eq("").mean()
        lab2_is_empty = lab2_empty_ratio > 0.5

        if (lab2_is_num < lab_is_num) and (not lab2_is_empty or lab2_is_num < 0.4):
            num_df = num_df2

    return num_df

# -----------------------
# 5) Row picking helpers
# -----------------------
def _norm(s: str) -> str:
    s = str(s).lower().replace("\n", " ")
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = s.replace(" and ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _find_row_by_cues_anywhere(df: pd.DataFrame, cues_norm):
    index_norm = {lbl: _norm(lbl) for lbl in df.index}
    for orig, n in index_norm.items():
        for cue in cues_norm:
            if cue in n:
                return orig
    search_cols = min(5, df.shape[1])
    for r in range(df.shape[0]):
        for c in range(search_cols):
            cell = _norm(df.iloc[r, c])
            for cue in cues_norm:
                if cue in cell:
                    return df.index[r]
    return None

# NOTE: added "sales" to cover alt wordings
revenue_cues = ["revenue", "turnover", "group turnover", "external turnover", "sales"]
gross_profit_cues = ["gross profit", "gross profit before", "gross profit after"]
operating_profit_cues = [
    "operating profit", "profit from operations", "profit from operation",
    "trading profit", "operating profit before", "profit before financing costs"
]
pbt_cues = [
    "profit before tax", "profit before taxation", "profit before income tax",
    "profit before tax and", "profit before taxation and"
]
net_income_cues = [
    "profit for the year", "profit for the period",
    "profit attributable to owners", "profit attributable to equity holders",
    "profit attributable to ordinary shareholders"
]

revenue_cues_n = [_norm(x) for x in revenue_cues]
gross_profit_cues_n = [_norm(x) for x in gross_profit_cues]
operating_profit_cues_n = [_norm(x) for x in operating_profit_cues]
pbt_cues_n = [_norm(x) for x in pbt_cues]
net_income_cues_n = [_norm(x) for x in net_income_cues]

# -----------------------
# 6A) Profitability from table (if possible)
#     Also return an amounts dataframe
# -----------------------
def compute_from_table(df: pd.DataFrame, rm_map: dict):
    rev_lbl = rm_map.get("Revenue")
    if rev_lbl is None:
        return None, None
    rows = {}
    for key, lbl in rm_map.items():
        if lbl is None:
            continue
        rows[key] = pd.to_numeric(df.loc[lbl], errors="coerce")
    amounts = pd.DataFrame(rows)  # columns = metrics, index = periods (years/cols)
    # margins for all available columns
    out = pd.DataFrame(index=amounts.index)
    if "Gross Profit" in amounts:
        out["Gross Margin"] = amounts["Gross Profit"] / amounts["Revenue"]
    if "Operating Profit" in amounts:
        out["Operating Margin"] = amounts["Operating Profit"] / amounts["Revenue"]
    if "Net Income" in amounts:
        out["Net Margin"] = amounts["Net Income"] / amounts["Revenue"]
    if "Profit Before Tax" in amounts:
        out["PBT Margin"] = amounts["Profit Before Tax"] / amounts["Revenue"]
    return out if out.shape[1] else None, amounts

# -----------------------
# Shared helpers for word-level fallback
# -----------------------
def normalize_number(txt: str) -> float | None:
    """
    Parse an accounting-style number token into float.
    - Ignores percentages.
    - Supports parentheses for negatives and Unicode minus.
    - Strips currency symbols and thousands separators.
    """
    if txt is None:
        return None
    if not isinstance(txt, str):
        txt = str(txt)

    if "%" in txt:
        return None

    t = txt.strip()
    t = t.replace("\u2212", "-").replace("−", "-")
    neg = ("(" in t) and (")" in t)
    t = t.replace("(", "").replace(")", "")
    t = (t.replace("\u2009", "")
           .replace("\u00a0", "")
           .replace(" ", "")
           .replace(",", ""))
    t = re.sub(r"[^0-9.\-]", "", t)
    if t.count(".") > 1:
        parts = t.split(".")
        t = "".join(parts[:-1]) + "." + parts[-1]
    if t in {"", "-", ".", "-."}:
        return None
    try:
        val = float(t)
        return -val if neg else val
    except Exception:
        return None

def read_lines(page, y_tol=2.5):
    words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    lines = {}
    for w in words:
        y = round(w["top"] / y_tol)
        lines.setdefault(y, []).append(w)
    for y in lines:
        lines[y].sort(key=lambda w: w["x0"])
    out = []
    for y, ws in sorted(lines.items(), key=lambda kv: kv[0]):
        txt = " ".join(w["text"] for w in ws)
        out.append((y, ws, txt))
    return out

def numeric_segments(ws, gap_tol=8.0):
    """
    Group adjacent numeric-like word tokens into number segments.
    Returns list of (value, right_edge_x) for each segment found on the line.
    """
    segs, cur = [], []

    def is_num_token(t: str) -> bool:
        if not t:
            return False
        if "%" in t:
            return False
        if re.search(r"[A-Za-z]", t):
            return False
        return re.fullmatch(r"[\(\)\-\d,.\s\u2009\u00a0]+", t) is not None

    for w in ws:
        t = w["text"]
        if is_num_token(t):
            if cur:
                prev = cur[-1]
                gap = w["x0"] - prev["x1"]
                if gap <= gap_tol:
                    cur.append(w)
                else:
                    segs.append(cur); cur = [w]
            else:
                cur = [w]
        else:
            if cur:
                segs.append(cur); cur = []
    if cur:
        segs.append(cur)

    out = []
    for seg in segs:
        txt = " ".join(w["text"] for w in seg)
        val = normalize_number(txt)
        if val is not None:
            out.append((val, seg[-1]["x1"]))
    return out

# -----------------------
# 6B) Word-level fallback (Latest + Prior) -> returns (margins_df, amounts_df)
# -----------------------
def find_metrics_from_words_two_periods(pdf, target_pages):
    def _norm2(s: str) -> str:
        s = str(s).lower().replace("\n", " ")
        s = re.sub(r"\s*-\s*", "-", s)
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = s.replace(" and ", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s
    cue_sets = {
        "Revenue": [_norm2(x) for x in revenue_cues],
        "Gross Profit": [_norm2(x) for x in gross_profit_cues],
        "Operating Profit": [_norm2(x) for x in (operating_profit_cues + ["trading profit"])],
        "Profit Before Tax": [_norm2(x) for x in pbt_cues],
        "Net Income": [_norm2(x) for x in net_income_cues],
    }

    cand = {k: [] for k in cue_sets}
    for p in target_pages:
        page = pdf.pages[p]
        for _, ws, line_text in read_lines(page):
            nline = _norm2(line_text)
            segs = numeric_segments(ws, gap_tol=8.0)
            if not segs:
                continue
            # Prefer larger-looking amounts, but don’t discard smaller ones entirely
            filtered = [seg for seg in segs if abs(seg[0]) >= 500]
            segs_sorted = sorted(filtered if filtered else segs, key=lambda t: t[1])
            vals = [v for v, _ in segs_sorted[-2:]]  # [prior?, latest]
            if not vals:
                continue
            for key, cues in cue_sets.items():
                if any(c in nline for c in cues):
                    cand[key].append(vals)

    rev_pairs = [v for v in cand["Revenue"] if v]
    revenue_latest = max((pair[-1] for pair in rev_pairs), default=None)
    revenue_prior  = max((pair[-2] for pair in rev_pairs if len(pair) > 1), default=None)

    def pick_close(pairs, upper, pos=-1):
        if upper is None:
            return None
        pos_vals = []
        for p in pairs:
            try:
                val = p[pos]
            except IndexError:
                continue
            if abs(val) <= upper:
                pos_vals.append(val)
        return max(pos_vals) if pos_vals else None

    upper_latest = abs(revenue_latest) * 1.1 if revenue_latest else None
    upper_prior  = abs(revenue_prior)  * 1.1 if revenue_prior  else None

    metrics_latest = {"Revenue": revenue_latest, "Gross Profit": None, "Operating Profit": None, "Profit Before Tax": None, "Net Income": None}
    metrics_prior  = {"Revenue": revenue_prior,  "Gross Profit": None, "Operating Profit": None, "Profit Before Tax": None, "Net Income": None}

    for key in ["Gross Profit", "Operating Profit", "Profit Before Tax", "Net Income"]:
        pairs = cand[key]
        metrics_latest[key] = pick_close(pairs, upper_latest, pos=-1)
        metrics_prior[key]  = pick_close(pairs, upper_prior,  pos=-2)

    print("🧪 Captured amounts (Rm-ish):",
          {"Latest": metrics_latest, "Prior": metrics_prior})

    amounts_df = pd.concat(
        [pd.DataFrame(metrics_latest, index=["Latest"]),
         pd.DataFrame(metrics_prior,  index=["Prior"])],
        axis=0
    )

    # Save amounts per-file (safe)
    out_amt = r"C:\Users\mwila\Downloads\income_statement_amounts.csv"
    try:
        amounts_df.to_csv(out_amt, encoding="utf-8-sig")
    except PermissionError:
        out_amt = rf"C:\Users\mwila\Downloads\income_statement_amounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        amounts_df.to_csv(out_amt, encoding="utf-8-sig")
    print("📂 Saved amounts:", out_amt)

    frames = []
    for label, row in amounts_df.iterrows():
        rev = row.get("Revenue")
        if pd.notna(rev) and rev != 0:
            data = {}
            if pd.notna(row.get("Gross Profit")):
                data["Gross Margin"] = row["Gross Profit"] / rev
            if pd.notna(row.get("Operating Profit")):
                data["Operating Margin"] = row["Operating Profit"] / rev
            if pd.notna(row.get("Net Income")):
                data["Net Margin"] = row["Net Income"] / rev
            if pd.notna(row.get("Profit Before Tax")):
                data["PBT Margin"] = row["Profit Before Tax"] / rev
            if data:
                frames.append(pd.DataFrame(data, index=[label]))
    margins_df = pd.concat(frames) if frames else None
    return margins_df, amounts_df

# -----------------------
# 7) Extra analysis metrics
# -----------------------
def compute_extra_metrics(margins_df: pd.DataFrame, amounts_df: pd.DataFrame) -> dict:
    out = {}

    # Margin spread (Gross - Operating), OP/GP conversion
    if {"Gross Margin", "Operating Margin"}.issubset(margins_df.columns):
        spread = margins_df["Gross Margin"] - margins_df["Operating Margin"]
        out["Margin Spread"] = spread
        if "Gross Profit" in amounts_df.columns and "Operating Profit" in amounts_df.columns:
            conv = amounts_df["Operating Profit"] / amounts_df["Gross Profit"]
            out["OP-to-GP Conversion"] = conv

    if "PBT Margin" in margins_df.columns:
        out["PBT Margin"] = margins_df["PBT Margin"]

    # YoY growth (Latest vs Prior) if we have both periods
    def last_two(series_like):
        if series_like is None:
            return None
        s = series_like.dropna()
        if s.size < 2:
            return None
        return s.iloc[-2], s.iloc[-1]

    growth = {}
    for key in ["Revenue", "Gross Profit", "Operating Profit", "Net Income"]:
        if key in amounts_df.columns:
            prior_latest = last_two(amounts_df[key])
            if prior_latest:
                prior, latest = prior_latest
                if pd.notna(prior) and prior != 0 and pd.notna(latest):
                    growth[key + " YoY"] = (latest - prior) / abs(prior)
    if growth:
        out["YoY Growth"] = pd.Series(growth)

    return out

# Save amounts & margins CSVs (safe)
def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, encoding="utf-8-sig")
        return path
    except PermissionError:
        newp = rf"{path[:-4]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(newp, encoding="utf-8-sig")
        return newp

# --------------------------------------------------------
# 8) Visualisations helpers (matplotlib only; one chart/fig; no custom colors)
# --------------------------------------------------------
plt.ioff()  # show() will block until you close the window

def show_fig(fig):
    fig.tight_layout()
    plt.show()
    plt.close(fig)

# Helpers for regression & period axis
year_re = re.compile(r"\b(19|20)\d{2}\b")
def as_period_axis(idx_like):
    labels = [str(x) for x in idx_like]
    years = []
    for s in labels:
        m = year_re.search(s)
        years.append(int(m.group(0)) if m else None)
    if any(y is not None for y in years):
        x = []
        cur = None
        for y in years:
            if y is not None:
                x.append(float(y)); cur = y
            else:
                cur = (cur + 1) if cur is not None else 0
                x.append(float(cur))
        return x, labels
    else:
        return list(map(float, range(len(labels)))), labels

def fit_linreg(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return None
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return a, b, r2

def plot_line_with_regression(x, y, labels, title, ylab):
    if len(x) < 1:
        return
    fig = plt.figure()
    plt.plot(x, y, marker="o")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel(ylab)
    plt.title(title)
    reg = fit_linreg(x, y)
    if reg:
        a, b, r2 = reg
        xgrid = np.linspace(min(x), max(x), 100)
        ygrid = a * xgrid + b
        plt.plot(xgrid, ygrid, linestyle="--", label=f"Trend (R²={r2:.2f})")
        plt.legend()
    show_fig(fig)

def plot_scatter_with_regression(x, y, title, xlab, ylab):
    if len(x) < 2:
        return
    fig = plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    reg = fit_linreg(x, y)
    if reg:
        a, b, r2 = reg
        xgrid = np.linspace(min(x), max(x), 100)
        ygrid = a * xgrid + b
        plt.plot(xgrid, ygrid, linestyle="--", label=f"Fit (R²={r2:.2f})")
        plt.legend()
    show_fig(fig)

# --------------------------------------------------------
# 9) Single-PDF analysis wrapped for reuse in multi-file run
# --------------------------------------------------------
def analyze_pdf(one_pdf_path):
    with pdfplumber.open(one_pdf_path) as pdf:
        pages = find_income_pages(pdf)
        if not pages:
            raise ValueError(f"Income statement heading not found in: {one_pdf_path}")
        print("🎯 Income statement likely on/near pages:", [p+1 for p in pages])
        candidates = extract_candidate_tables(pdf, pages, lookahead=4)

    if not candidates:
        raise RuntimeError(f"No usable tables near income statement pages in: {one_pdf_path}")

    scored = []
    for idx, cand in enumerate(candidates):
        try:
            lbl, sc = classify_table(cand)
            if lbl == "income_statement":
                sc += 5
            sc += score_structural(cand) + 6.0 * score_textiness(cand)
            scored.append((sc, idx, cand))
        except Exception as e:
            print(f"  ⚠️ classify/score failed on table {idx}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)  # one-line origin
            continue

    if not scored:
        raise RuntimeError("All candidate tables failed classification/scoring.")
    scored.sort(key=lambda x: (-x[0], x[1]))

    best_df = None
    for _, _, cand in scored:
        try:
            cleaned = clean_statement(cand)
        except Exception as e:
            print(f"  ⚠️ clean_statement failed: {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)  # one-line origin
            continue
        lab_textiness = (
            pd.Series(cleaned.index.astype(str))
              .str.contains(r"[A-Za-z]", regex=True).mean()
            if cleaned is not None and not cleaned.empty else 0.0
        )
        if cleaned.shape[0] >= 2 and cleaned.shape[1] >= 1 and lab_textiness >= 0.15:
            best_df = cleaned
            break

    if best_df is not None:
        raw_is_csv = r"C:\Users\mwila\Downloads\chosen_income_statement_raw.csv"
        try:
            best_df.to_csv(raw_is_csv, encoding="utf-8-sig")
        except PermissionError:
            raw_is_csv = rf"C:\Users\mwila\Downloads\chosen_income_statement_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            best_df.to_csv(raw_is_csv, encoding="utf-8-sig")
        print("📂 Saved raw income-statement candidate to:", raw_is_csv)
        print("🧾 First 30 index labels:", list(map(str, best_df.index[:30])))

    row_map = {}
    if best_df is not None:
        row_map = {
            "Revenue": _find_row_by_cues_anywhere(best_df, revenue_cues_n),
            "Gross Profit": _find_row_by_cues_anywhere(best_df, gross_profit_cues_n),
            "Operating Profit": _find_row_by_cues_anywhere(best_df, operating_profit_cues_n),
            "Profit Before Tax": _find_row_by_cues_anywhere(best_df, pbt_cues_n),
            "Net Income": _find_row_by_cues_anywhere(best_df, net_income_cues_n),
        }
        print("🔎 Row matches (normalized):", row_map)

    margins, amounts = None, None
    if best_df is not None:
        margins, amounts = compute_from_table(best_df, row_map)

    if margins is None or margins.empty or amounts is None or amounts.empty:
        try:
            with pdfplumber.open(one_pdf_path) as pdf:
                pages = find_income_pages(pdf)
                margins, amounts = find_metrics_from_words_two_periods(pdf, pages)
        except Exception as e:
            print(f"  ⚠️ word-level fallback failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            margins, amounts = None, None

    if margins is None or margins.empty:
        for k, (_, _, cand) in enumerate(scored[:5], 1):
            try:
                clean_k = clean_statement(cand)
                clean_k.to_csv(rf"C:\Users\mwila\Downloads\income_candidate_clean_{k}.csv", encoding="utf-8-sig")
            except Exception:
                pass
        raise ValueError(f"Could not compute margins from tables or page text: {one_pdf_path}")

    print("\n✅ Final profitability margins (per column/year):")
    print(margins)

    extras = compute_extra_metrics(margins, amounts)

    if best_df is not None:
        out_is = r"C:\Users\mwila\Downloads\chosen_income_statement.csv"
        path_is = safe_save_csv(best_df, out_is)
        print("📂 Saved:", path_is)

    out_amt = safe_save_csv(amounts, r"C:\Users\mwila\Downloads\income_statement_amounts.csv")
    out_mgn = safe_save_csv(margins, r"C:\Users\mwila\Downloads\profitability_margins.csv")
    print("📂 Saved:", out_amt)
    print("📂 Saved:", out_mgn)

    if extras:
        for name, obj in extras.items():
            if isinstance(obj, pd.Series):
                df = obj.to_frame(name=name)
            elif isinstance(obj, pd.DataFrame):
                df = obj.copy()
            else:
                continue
            p = safe_save_csv(df, rf"C:\Users\mwila\Downloads\extra_{name.lower().replace(' ', '_')}.csv")
            print(f"📂 Saved extra metric: {name} -> {p}")

    return best_df, margins, amounts, extras

# --------------------------------------------------------
# 10) MULTI-FILE RUN: analyze many PDFs and combine by year
# --------------------------------------------------------
combined_amounts = []
combined_margins = []
extras_list = []

def _year_key(k: str):
    m = re.search(r"(19|20)\d{2}", k)
    return int(m.group(0)) if m else 0

for year_label, path in sorted(pdf_paths.items(), key=lambda kv: _year_key(kv[0])):
    print(f"\n🚀 Processing {year_label}: {path}")
    try:
        best_df, margins, amounts, extras = analyze_pdf(path)
    except Exception as e:
        print(f"⚠️ Skipping {year_label} due to error: {e!r}")
        traceback.print_exc()
        continue

    am = amounts.copy()
    am["__Year"] = str(year_label)
    mg = margins.copy()
    mg["__Year"] = str(year_label)

    am.index = pd.Index([f"{year_label}-{idx}" for idx in am.index], name="Period")
    mg.index = pd.Index([f"{year_label}-{idx}" for idx in mg.index], name="Period")

    combined_amounts.append(am)
    combined_margins.append(mg)
    extras_list.append((str(year_label), extras))

# ---- CLEAN CONCAT (no FutureWarning): drop all-NA columns before concat ----
def _drop_all_na_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.loc[:, df.notna().any(axis=0)]

combined_amounts = [
    _drop_all_na_cols(df) for df in combined_amounts
    if df is not None and not df.empty
]
combined_amounts = [df for df in combined_amounts if not df.empty]

combined_margins = [
    _drop_all_na_cols(df) for df in combined_margins
    if df is not None and not df.empty
]
combined_margins = [df for df in combined_margins if not df.empty]

if combined_amounts:
    amounts_all = pd.concat(combined_amounts, axis=0, join="outer", copy=False)
else:
    amounts_all = pd.DataFrame()

if combined_margins:
    margins_all = pd.concat(combined_margins, axis=0, join="outer", copy=False)
else:
    margins_all = pd.DataFrame()

# Optional: save the stitched CSVs
if not amounts_all.empty:
    amounts_all.to_csv(r"C:\Users\mwila\Downloads\income_statement_amounts_ALL.csv", encoding="utf-8-sig")
if not margins_all.empty:
    margins_all.to_csv(r"C:\Users\mwila\Downloads\profitability_margins_ALL.csv", encoding="utf-8-sig")

print("\n✅ Aggregated across PDFs:")
print(amounts_all.tail() if not amounts_all.empty else "No amounts aggregated.")
print(margins_all.tail() if not margins_all.empty else "No margins aggregated.")

# --------------------------------------------------------
# 11) VISUALISATIONS
# --------------------------------------------------------
# A) Net Margin (Latest) by Year (bar)
if not margins_all.empty and "Net Margin" in margins_all.columns:
    fig = plt.figure()
    latest_rows = [i for i in margins_all.index if i.endswith("-Latest")]
    if latest_rows:
        data = margins_all.loc[latest_rows].copy()
        data["__Year"] = data.index.to_series().str.split("-").str[0]
        vals = data.groupby("__Year")["Net Margin"].mean()
        plt.bar(range(len(vals)), list(vals.values))
        plt.xticks(range(len(vals)), list(vals.index))
        plt.ylabel("Net Margin (ratio)")
        plt.title("Net Margin (Latest) by Year")
        show_fig(fig)

# B) Revenue (Latest) by Year with regression line
if not amounts_all.empty and "Revenue" in amounts_all.columns:
    latest_rows = [i for i in amounts_all.index if i.endswith("-Latest")]
    if latest_rows:
        data = amounts_all.loc[latest_rows].copy()
        data["__Year"] = data["__Year"].astype(str)
        x = [float(y) for y in data["__Year"]]
        y = [float(v) if pd.notna(v) else np.nan for v in data["Revenue"]]
        plot_line_with_regression(x, y, list(data["__Year"]), "Revenue (Latest) by Year", "Revenue (Rm)")

# C) Operating Margin (Latest) by Year with regression
if not margins_all.empty and "Operating Margin" in margins_all.columns:
    latest_rows = [i for i in margins_all.index if i.endswith("-Latest")]
    if latest_rows:
        data = margins_all.loc[latest_rows].copy()
        data["__Year"] = data["__Year"].astype(str)
        x = [float(y) for y in data["__Year"]]
        y = [float(v) if pd.notna(v) else np.nan for v in data["Operating Margin"]]
        plot_line_with_regression(x, y, list(data["__Year"]), "Operating Margin (Latest) by Year", "Operating Margin (ratio)")

# D) Scatter: Net Income vs Revenue across years (Latest only)
if (not amounts_all.empty and {"Revenue", "Net Income"}.issubset(amounts_all.columns)):
    latest_rows = [i for i in amounts_all.index if i.endswith("-Latest")]
    if latest_rows:
        data = amounts_all.loc[latest_rows].copy()
        x_vals = [float(v) for v in data["Revenue"] if pd.notna(v)]
        y_vals = [float(v) for v in data["Net Income"] if pd.notna(v)]
        n = min(len(x_vals), len(y_vals))
        plot_scatter_with_regression(x_vals[:n], y_vals[:n],
                                     "Net Income vs Revenue (Latest, All Years)",
                                     "Revenue (Rm)", "Net Income (Rm)")

# E) (Optional) Per-year grouped bars for margins across periods (Latest/Prior) within each year
if not margins_all.empty:
    fig = plt.figure()
    metric = "PBT Margin" if "PBT Margin" in margins_all.columns else None
    if metric:
        idx = list(margins_all.index)
        years = sorted(set(i.split("-")[0] for i in idx))
        periods = ["Prior", "Latest"]
        vals, labels, x_pos = [], [], []
        cur = 0
        for y in years:
            for p in periods:
                key = f"{y}-{p}"
                if key in margins_all.index:
                    v = margins_all.loc[key, metric]
                    vals.append(float(v) if pd.notna(v) else 0.0)
                else:
                    vals.append(0.0)
                labels.append(f"{y}-{p}")
                x_pos.append(cur)
                cur += 1
        if labels:
            plt.bar(x_pos, vals)
            plt.xticks(x_pos, labels, rotation=15, ha="right")
            plt.ylabel(f"{metric} (ratio)")
            plt.title(f"{metric} by Year & Period")
            show_fig(fig)

print("\n✅ Done. Multi-file analysis complete with year-distinguished visuals.")
