# StockAI Terminal — Project Notes

A running log of what's in this project and the work that's been done so far.

---

## What this is

A Streamlit web app (**StockAI Terminal**) that predicts the 30-trading-day
direction (UP / DOWN) for ~31 large-cap US stocks across four sectors, and uses
the Groq LLM API to generate a plain-English analysis on top of the model's
output.

- **Model:** scikit-learn `AdaBoostClassifier` (DecisionTree base, depth 3,
  100 estimators) trained on 10 years of daily OHLCV + fundamentals.
- **Features (10):** RSI, MACD, MACD_Hist, BB_Position, Volume_Ratio,
  Volatility, Daily_Return, PE_Ratio, Revenue_Growth, Profit_Margin.
- **Target:** `Close.shift(-30) / Close - 1 > 0` (binary).
- **Split:** time-based 80/20.
- **LLM layer:** Groq (`groq` Python SDK) — gives a written narrative based on
  the model's probability + the latest indicators.

---

## Files in the repo

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — search box, chart, prediction card, LLM commentary. |
| `retrain.py` | Standalone trainer — pulls 10y data, fits AdaBoost, writes pickles + log. |
| `ada_model.pkl` | Pickled trained classifier. |
| `model_config.pkl` | `{feature_cols, importance}` dict consumed by `app.py`. |
| `retrain_log.json` | Append-only history of every retrain (last 50 entries). |
| `requirements.txt` | streamlit, yfinance, pandas, numpy, scikit-learn, plotly, groq. |
| `.github/workflows/retrain.yml` | Weekly GitHub Action — Sunday 00:00 UTC. |
| `.gitignore` | venv, secrets, Jupyter checkpoints, `.claude/`, etc. |
| `stock_analyser.ipynb` | Original exploratory notebook (untracked). |

---

## Universe trained on (31 tickers, 4 sectors)

- **Tech (15):** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, RDDT, AMD, CRM,
  INTC, NFLX, SHOP, UBER, SNAP
- **Finance (5):** JPM, GS, V, MA, PYPL
- **Real Estate (6):** O, AMT, PLD, SPG, WELL, DLR
- **Defence (5):** LMT, RTX, NOC, GD, BA

Stocks with < 200 trading days of history are skipped automatically.

---

## Latest retrain (from `retrain_log.json`)

- **Date:** 2026-05-11 19:56 UTC
- **Rows:** 74,999 total (59,255 train / 14,814 test)
- **Accuracy:** **57.44%** (baseline UP rate is 59.8%, so model is roughly on par
  with always-predict-UP — feature engineering is the next lever)
- **Top features by importance:**
  - Volatility — 41.7%
  - PE_Ratio — 17.7%
  - MACD_Hist — 16.9%
  - Profit_Margin — 13.8%
  - MACD — 6.8%
  - Revenue_Growth — 3.0%
  - RSI / Volume_Ratio / Daily_Return — ~0%

---

## Local setup

```powershell
# from C:\Users\ADULT\Desktop\stock-analyser
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1            # Python 3.10.11
pip install -r requirements.txt        # if needed
$env:GROQ_API_KEY = "sk-..."           # required for LLM commentary
streamlit run app.py
```

Retrain manually:

```powershell
python retrain.py
```

---

## Work log (commits, oldest → newest)

1. **`37cb73d` — StockAI Terminal v1: 10yr AdaBoost + Groq LLM + auto-retrain**
   Initial drop: `app.py` (1,794 lines), `retrain.py`, pickled model + config,
   first `retrain_log.json` entry, `.gitignore`, `requirements.txt`.

2. **`787903e` — Fix SVG rendering, add Groq retry, add weekly retrain workflow**
   - Inline SVG icons fixed (no broken/escaped markup).
   - Groq client wrapped in retry/backoff so transient 5xx don't break the UI.
   - GitHub Action `retrain.yml` added — runs every Sunday 00:00 UTC, commits
     refreshed pickles + log back to `main`.

3. **`4f63b3d` — Fix .gitignore corruption, source Groq key safely, sanitize HTML**
   - Rewrote `.gitignore` (had been UTF-16/garbled).
   - `GROQ_API_KEY` read from env / Streamlit secrets, never hard-coded.
   - All user-controlled strings escaped via `_esc()` before being injected into
     `unsafe_allow_html` blocks → blocks injection through ticker names etc.

4. **`699ee6d` — Make UI responsive: tablet, mobile, and phone breakpoints**
   Added CSS media queries so the Bloomberg-style layout scales down to phone
   widths.

5. **`17950ad` — Mobile-first UX: main-area search, popular grid, compact results
   search**
   Latest. Moved search into the main panel, added a "popular tickers" grid for
   one-tap selection, and a more compact in-results search.

---

## CI / automation

- **Weekly retrain** (`.github/workflows/retrain.yml`)
  - Trigger: `cron: "0 0 * * 0"` (Sunday 00:00 UTC) + manual `workflow_dispatch`.
  - Runs `python retrain.py` on `ubuntu-latest` with Python 3.10.
  - Commits `ada_model.pkl`, `model_config.pkl`, `retrain_log.json` if changed,
    with message `chore(model): weekly retrain YYYY-MM-DD — accuracy NN.NN%`.
  - Uses the default `GITHUB_TOKEN` for the push (no PAT required).

---

## Security notes

- No secrets committed. `GROQ_API_KEY` is read from environment or
  `st.secrets`, never inlined.
- All dynamic strings rendered into HTML go through `html.escape()` first.
- `.gitignore` blocks `venv/`, `.env*`, `*.secret`, `.streamlit/secrets.toml`,
  `.claude/`.

---

## Known limitations / ideas

- 57% accuracy is barely above the 59.8% always-UP baseline → real lift is
  small. Worth trying: regime features, sector-relative momentum, cross-asset
  signals (rates, dollar, VIX), per-sector models, calibrated probabilities.
- `RSI`, `Volume_Ratio`, `Daily_Return` carry ~0 importance — could be dropped
  or replaced with engineered variants.
- Only 31 tickers, all US large-cap. No international, no small-cap.
- Predictions are 30-day directional only — no magnitude / no risk sizing.
- Sector accuracy in the log only shows Real Estate + Defence because the
  80/20 time split puts most of the test window in those sectors' rows after
  shuffling by date.

---

## Working state (as of this note)

- Branch: `main`, in sync with `origin/main`.
- Untracked: `stock_analyser.ipynb` (the original exploration notebook —
  intentionally not committed; superseded by `app.py` + `retrain.py`).
- venv activated: `venv\Scripts\python.exe` (Python 3.10.11).
