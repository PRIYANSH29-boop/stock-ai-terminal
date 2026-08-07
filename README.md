# StockAI Terminal

An end-to-end **ML + LLM prototype** for stock analysis: it pulls market data,
trains a directional signal model, validates it honestly, and uses an LLM to
explain the model's output in plain English — then redeploys and retrains itself
weekly. The breadth is the point. **It is a well-built prototype, not a production
system**, and its value is the explainability and the grounded-LLM layer, not the
prediction accuracy (which is, honestly, about a coin flip — see below).

To be precise about that word: a pickled model committed to git, retrained by
commit, on yfinance's free API, served on Streamlit's free tier is a prototype.
A good one — end-to-end and self-maintaining — but a prototype.

**Live:** https://stock-ai-terminal.streamlit.app/

> ### Where this sits: earlier exploration, since superseded
>
> StockAI Terminal asks *"will this stock go up?"* — direction prediction on one
> stock from its own features. I took that same question into
> **[RankAlpha](https://github.com/PRIYANSH29-boop/financeai)** and audited it
> properly. After purging look-ahead leakage and charging realistic costs,
> single-stock direction landed at roughly a **coin flip**: it is dominated by
> market-wide moves nobody can predict. The leakage fix documented below was the
> first sighting of that result; RankAlpha is where it was measured to the end.
>
> So the question changed — from *"will this stock go up?"* to *"will it
> outperform the other 499?"*. That is a cross-sectional **ranking** problem,
> where the market's move cancels between the legs and what is left is actually
> about the individual stock. It is the approach that survived, and it is where
> the work continued.
>
> This repo stays up as the earlier rung, not as a current recommendation. The
> part that carried forward is the **grounded-LLM layer** — an explainer fed
> structured model output so it cites numbers rather than inventing them.

---

## What it does

Predicts the **30-trading-day direction** (UP / DOWN) for ~31 large-cap US stocks
across four sectors, shows the technicals on a chart, and generates a written
analysis on top of the model's output.

```
yfinance (10y OHLCV)
      │
      ▼
technical indicators ──► AdaBoost classifier ──► P(UP) + feature importances
                                                        │
                                                        ▼
                                          Llama 3.3 70B (via Groq), fed the
                                          structured model output, writes a
                                          grounded plain-English explanation
                                                        │
                                                        ▼
                                          Streamlit UI  ◄── weekly GitHub Actions
                                                              retrain commits the
                                                              refreshed model back
```

---

## The model

- **Algorithm:** scikit-learn `AdaBoostClassifier` (DecisionTree base, depth 3,
  100 estimators, lr 0.1).
- **Target:** `Close.shift(-30) / Close - 1 > 0` — did the price close higher 30
  trading days later?
- **Features (7):** RSI, MACD, MACD histogram, Bollinger-band position, volume
  ratio, volatility, daily return.
- **Universe:** 31 tickers across Tech, Finance, Real Estate, and Defence;
  ~10 years of daily data, ~75,000 rows.

### How it's validated

A **per-ticker chronological 80/20 split with a 30-row embargo** at the boundary.
The target looks 30 days into the future, so the last 30 training rows of each
ticker would otherwise have forward windows that overlap the test period — classic
label leakage. The embargo drops exactly those rows, per ticker, so the split is
clean.

> This is a single time-based split with leakage protection — **not** walk-forward
> (rolling-window) cross-validation. Rolling CV is a sensible next step; it isn't
> what runs today.

### How it actually performs

Stating this plainly because it's the honest number:

| Metric | Value |
|---|---|
| Directional accuracy | **55.4%** |
| Always-UP baseline | 55.2% |
| Lift over baseline | **+0.23 pp** |
| ROC-AUC | 0.549 |
| DOWN-class recall | 0.026 |

In other words, after the leakage fix below, the model is **barely distinguishable
from "always predict UP"** — it predicts UP almost every time. On a genuinely hard
problem that's unsurprising, and it's exactly why this project leans on the
explanation layer rather than the prediction.

### The leakage fix (the part worth reading)

An earlier version of the model included three fundamentals — PE ratio, revenue
growth, profit margin — and scored ~57%. That number was wrong. `yfinance`'s
`.info` returns a **current snapshot**, and those present-day values were being
stamped onto all 10 years of each ticker's history. The model could see the
future.

Removing them dropped accuracy from ~57% to ~55% — i.e. **most of the apparent
edge was lookahead bias.** The free tier doesn't expose reliable point-in-time
fundamentals (`.quarterly_financials` is shallow and gets restated), so the honest
move was to drop the leaky features rather than launder them. The accuracy table
above is the post-fix truth.

### What the model learned

With the leak gone, the feature importances are the interesting output:

| Feature | Importance |
|---|---|
| Volatility | **64%** |
| MACD histogram | 17% |
| MACD | 14% |
| RSI | 3.8% |
| Volume ratio, daily return, BB position | ~0% |

Volatility dominates; the much-loved RSI contributes almost nothing. Surfacing
that kind of finding is what the explainability layer is for.

---

## The LLM layer

A **Llama 3.3 70B** model, served via **Groq**, turns the model's output into a
readable analysis. Crucially, it's fed the **structured model output** — the
probability, the latest indicator values, the feature importances — so it **cites
those numbers rather than inventing its own.**

It **never gives buy or sell advice.** It explains what the model saw and why, and
stops there. The UI repeats the disclaimer (≈55% backtest accuracy, "one signal
among many, not a recommendation") on every result.

---

## Automation

The model **retrains weekly via GitHub Actions** (`.github/workflows/retrain.yml`,
Sunday 00:00 UTC, plus manual dispatch). The job runs `retrain.py` on a clean
runner, and if the refreshed `ada_model.pkl` / `model_config.pkl` /
`retrain_log.json` differ, it commits them back to `main` with a message like
`chore(model): weekly retrain 2026-05-20 — accuracy 55.43%`. No manual step, no
PAT — it uses the default `GITHUB_TOKEN`.

`retrain_log.json` is an append-only history (last 50 runs) of every retrain's
metrics, so the model's performance over time is auditable rather than a single
claimed number.

---

## Honest limitations

- **AdaBoost was chosen, not validated against alternatives.** It hasn't been
  benchmarked against XGBoost or a gradient-boosted baseline, so it's the current
  choice, not a justified one.
- **Per-prediction explanations are global, not local.** Today the explanation
  uses model-wide feature importances. **SHAP on individual predictions is
  planned, not done** — there are no per-prediction attributions yet.
- **The signal barely beats the baseline.** +0.23 pp over always-UP is, for
  practical purposes, no edge. This is a directional-classification prototype, not
  a trading strategy.
- **Narrow universe, single horizon.** 31 US large-cap tickers, 30-day directional
  only — no international/small-cap coverage, no magnitude, no risk sizing.
- **Validation is one split, not rolling CV** (see above).

---

## Run it locally

```powershell
# Python 3.10
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY = "your-key"      # required for the LLM commentary
streamlit run app.py
```

Retrain manually:

```powershell
python retrain.py
```

`GROQ_API_KEY` is read from the environment or `st.secrets` — never committed.
`.gitignore` blocks `venv/`, `.env*`, `*.secret`, and `.streamlit/secrets.toml`.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — search, chart, prediction card, LLM commentary. |
| `retrain.py` | Standalone trainer — pulls 10y data, fits AdaBoost, writes pickles + log. |
| `ada_model.pkl` | Pickled trained classifier (committed; refreshed by the weekly job). |
| `model_config.pkl` | `{feature_cols, importance}` consumed by `app.py`. |
| `retrain_log.json` | Append-only history of every retrain (last 50 runs). |
| `.github/workflows/retrain.yml` | Weekly retrain — Sunday 00:00 UTC. |
| `requirements.txt` | streamlit, yfinance, pandas, numpy, scikit-learn, plotly, groq. |

---

*Not financial advice. This is an engineering project about explainable ML, not a
tool for making investment decisions.*
