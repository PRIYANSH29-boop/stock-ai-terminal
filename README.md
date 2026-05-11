# ⚡ StockAI Terminal

A production-grade stock analysis system that combines **AdaBoost machine learning**, **Llama 3.3 70B** reasoning via Groq, and **10 years of live market data** to produce explainable equity research — deployed as a Streamlit web app with automatic weekly retraining.

> **Not a toy demo.** This follows the architecture of production AI systems used in fintech — ML model predicts, LLM explains, everything cites its sources.

[![Live App](https://img.shields.io/badge/Live_App-StockAI_Terminal-0ea5e9?style=for-the-badge&logo=streamlit&logoColor=white)](https://stock-ai-terminal.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-000000?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Auto Retrain](https://img.shields.io/badge/Auto_Retrain-Weekly_via_GitHub_Actions-10b981?style=flat&logo=githubactions&logoColor=white)

---

## What it does

Type any stock — Apple, Reddit, Tesla, or any ticker — and get a complete equity briefing:

```
User types "AAPL" →
  ┌──────────────────────────────────────────────────────────┐
  │  1. Pull 10 years of price data + fundamentals           │
  │  2. Calculate 10 technical indicators                    │
  │  3. AdaBoost ML model predicts 30-day direction          │
  │  4. Model ranks which factors matter most                │
  │  5. Fetch latest news headlines                          │
  │  6. Llama 3.3 70B writes analyst-grade research report   │
  │  7. Everything displayed in a dark trading dashboard     │
  └──────────────────────────────────────────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      StockAI Terminal                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │   DATA   │──▶│ SIGNALS  │──▶│ FEATURE  │──▶│  LLM BRAIN  │ │
│  │ PIPELINE │   │ (AdaBoost│   │ RANKING  │   │ (Groq/Llama)│ │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬──────┘ │
│       │              │              │                 │         │
│  yfinance        10yr trained   Importance %     Analyst-grade  │
│  10yr OHLCV      31 stocks     per indicator     research with  │
│  Fundamentals    Walk-forward   "Volatility:     cited numbers  │
│  Live news       validation     42%, MACD: 21%"  Never hallu-   │
│                                                  cinates        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              STREAMLIT DASHBOARD (DARK THEME)              ││
│  │  Prediction │ Indicators │ Charts │ News │ AI Analysis     ││
│  │  Risk metrics │ Fundamentals │ Performance │ Glossary      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              AUTO-RETRAIN (GitHub Actions — Weekly)         ││
│  │    Every Sunday: pull fresh data → retrain → commit model   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Key features

**ML Prediction Engine**
- AdaBoost classifier trained on 10 years of data across 31 stocks (4 sectors)
- Walk-forward temporal validation — no look-ahead bias
- Outputs probability score + feature importance ranking
- Accuracy: ~57% (honest — stock prediction is hard)

**Explainable AI**
- Model reveals which factors drive each prediction
- Key finding: Volatility (42%) and MACD (21%) dominate. RSI (0%) is noise for 30-day predictions
- Every number in the analysis is cited from actual data

**LLM Research Reports**
- Llama 3.3 70B via Groq generates comprehensive equity briefings
- Sections: Verdict, Technical Picture, Fundamentals, Bull/Bear Case, Risks, Beginner Glossary
- LLM receives structured ML output — never hallucinates, always cites sources
- Never gives buy/sell advice — analysis only

**Live Data**
- Real-time prices, indicators, fundamentals via Yahoo Finance
- RSI, MACD, Bollinger Bands, Stochastic, ATR, SMA crossovers
- Volume analysis, drawdown curves, return distributions
- 10 news headlines with summaries and source links

**Auto-Retraining Pipeline**
- GitHub Actions workflow retrains the model every Sunday
- Pulls fresh 10-year data, retrains AdaBoost, commits updated model
- Model freshness badge in the UI (green/amber/red based on age)
- Manual retrain: `python retrain.py`

**Beginner Mode**
- Toggle explanations for every indicator and metric
- Full glossary of 20+ financial terms in plain English
- Designed so someone who's never invested can understand the output

---

## What the model learned

After training on 74,000+ data points across 31 stocks and 10 years:

```
Feature importance ranking:
  Volatility            41.7%  █████████████████████████████████████████
  PE_Ratio              17.7%  █████████████████
  MACD_Hist             16.9%  ████████████████
  Profit_Margin         13.8%  █████████████
  MACD                   6.8%  ██████
  Revenue_Growth         3.0%  ███
  BB_Position            0.1%  
  RSI                    0.0%  
  Volume_Ratio           0.0%  
  Daily_Return           0.0%  
```

**Insight:** The most popular indicator traders use (RSI) contributes essentially nothing to 30-day predictions. Volatility alone explains 42% of predictive power. This is the kind of finding that makes this project worth discussing in interviews.

---

## Tech stack

| Category | Technologies |
|----------|-------------|
| **ML** | Python, scikit-learn, AdaBoost, pandas, NumPy |
| **LLM** | Groq API, Llama 3.3 70B, prompt engineering |
| **Data** | yfinance, Yahoo Finance API |
| **Visualisation** | Plotly, Streamlit |
| **CI/CD** | GitHub Actions (weekly auto-retrain) |
| **Deployment** | Streamlit Community Cloud |

---

## Project structure

```
stock-ai-terminal/
├── app.py                          # Main Streamlit application (900+ lines)
├── retrain.py                      # Self-contained retraining script
├── ada_model.pkl                   # Trained AdaBoost model
├── model_config.pkl                # Feature columns + importance weights
├── retrain_log.json                # Retraining history (last 50 runs)
├── requirements.txt                # Python dependencies
├── .github/
│   └── workflows/
│       └── retrain.yml             # Weekly auto-retrain via GitHub Actions
├── .gitignore
└── README.md
```

---

## Stocks covered (training data)

| Sector | Stocks |
|--------|--------|
| **Tech** (15) | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, RDDT, AMD, CRM, INTC, NFLX, SHOP, UBER, SNAP |
| **Finance** (5) | JPM, GS, V, MA, PYPL |
| **Real Estate** (6) | O, AMT, PLD, SPG, WELL, DLR |
| **Defence** (5) | LMT, RTX, NOC, GD, BA |

The app can analyse **any stock** — the model was trained on these 31 but predictions work for any ticker with sufficient history.

---

## Run locally

```bash
git clone https://github.com/PRIYANSH29-boop/stock-ai-terminal.git
cd stock-ai-terminal
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# Set your Groq API key (free at console.groq.com)
set GROQ_API_KEY=gsk_your_key_here        # Windows
# export GROQ_API_KEY=gsk_your_key_here   # Mac/Linux

streamlit run app.py
```

## Retrain the model

```bash
python retrain.py
```

This pulls fresh 10-year data for all 31 stocks, retrains AdaBoost, and saves updated model files. Takes ~35 seconds.

---

## Key design decisions

**Why AdaBoost over XGBoost?**
AdaBoost focuses on hard-to-classify examples — stocks that are tricky to predict get more attention during training. Each weak learner (shallow decision tree) captures a different pattern. For this problem, AdaBoost's emphasis on difficult cases proved effective. XGBoost comparison is planned as a future improvement.

**Why walk-forward validation?**
Random train/test splits cause look-ahead bias in financial data — the model would train on 2025 data and get tested on 2023 data it already learned from. Walk-forward uses the first 80% of time for training and the last 20% for testing. Honest evaluation.

**Why Groq + Llama 3.3 70B?**
Groq offers the fastest inference for open-source models — Llama 3.3 70B generates analyst-grade text in under 2 seconds. The LLM receives structured data (not raw prompts), so it cites real numbers instead of hallucinating. It never gives buy/sell advice — only analysis.

**Why auto-retrain weekly?**
Markets change. A model trained on 2020 data doesn't understand 2025 dynamics. Weekly retraining via GitHub Actions keeps the model fresh without manual intervention. The UI shows model age with colour-coded freshness badges.

**Why 57% accuracy is actually fine**
Stock prediction is one of the hardest ML problems. >55% on a 30-day horizon with honest temporal validation is meaningful edge. The real value isn't the prediction — it's the explainability layer that tells you WHY, ranked by importance.

---

## Limitations (honest assessment)

- Model accuracy is ~57% — better than random (50%) but not dramatically
- Feature importance is global (same weights for all stocks) — sector-specific models would improve this
- News sentiment is not quantified — headlines are shown but not scored
- No backtesting dashboard yet — past predictions aren't visualised
- Single model architecture — ensemble of multiple models would be more robust
- Rate limiting on Groq free tier during heavy usage

---

## Future improvements

- [ ] Sector-specific AdaBoost models (different weights for tech vs defence)
- [ ] XGBoost comparison + model ensemble
- [ ] Quantified news sentiment as a feature
- [ ] Backtesting dashboard with historical prediction accuracy
- [ ] Portfolio analysis (multiple stocks at once)
- [ ] SHAP explainability on individual predictions
- [ ] Telegram bot for real-time alerts

---

## Disclaimer

⚠️ **This tool is for educational and research purposes only.** It does not provide financial advice. Model accuracy is approximately 57%. Past performance does not indicate future results. Never make investment decisions based solely on algorithmic output.

---

## Built by

**Priyansh Patel** — BSc Computer Science, University of East London
