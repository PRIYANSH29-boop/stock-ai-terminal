import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import base64
import html
import json
import os
import time
import warnings
from pathlib import Path
from groq import Groq
from datetime import datetime, timezone


def _esc(value) -> str:
    """HTML-escape a value before injecting it into an unsafe_allow_html block.
    Tolerates None / non-strings."""
    if value is None:
        return ""
    return html.escape(str(value), quote=False)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# MODEL + LLM CLIENT
# ============================================================
with open('ada_model.pkl', 'rb') as f:
    ada = pickle.load(f)
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)

feature_cols = config['feature_cols']
importance = config['importance']


def load_retrain_info():
    """Return dict with 'retrained_at' (datetime), 'accuracy' (float, 0–1), and a human label.
    Falls back to (mtime of ada_model.pkl, None, 'unknown') if retrain_log.json is missing."""
    log_path = Path(__file__).resolve().parent / "retrain_log.json"
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
            last = entries[-1] if isinstance(entries, list) else entries
            ts = datetime.fromisoformat(last["retrained_at"].replace("Z", "+00:00"))
            return {
                "retrained_at": ts,
                "accuracy": last.get("accuracy"),
                "source": "log",
            }
        except Exception:
            pass

    # Fallback — use the model file's modification time.
    model_path = Path(__file__).resolve().parent / "ada_model.pkl"
    if model_path.exists():
        ts = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
        return {"retrained_at": ts, "accuracy": None, "source": "mtime"}
    return {"retrained_at": None, "accuracy": None, "source": "missing"}


def humanize_age(ts: datetime) -> str:
    """Return e.g. '3 days ago' / '2 hours ago' / 'just now'."""
    if ts is None:
        return "unknown"
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = now - ts
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hr ago"
    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    weeks = days // 7
    if weeks < 5:
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    return f"{days // 365} year{'s' if days // 365 != 1 else ''} ago"


RETRAIN_INFO = load_retrain_info()


def _resolve_groq_key() -> str:
    """Look for the Groq API key in (1) env var, (2) Streamlit secrets, (3) empty.

    Returns "" if not set — Groq() still constructs and the retry helper turns
    auth errors into a friendly fallback string instead of crashing.
    """
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return ""


GROQ_API_KEY = _resolve_groq_key()
client = Groq(api_key=GROQ_API_KEY or "missing")


# ============================================================
# GROQ CALL — with retry and graceful fallback
# ============================================================
RATE_LIMIT_FALLBACK = (
    "⚠️ AI analysis temporarily unavailable due to rate limiting. "
    "All data, indicators, and predictions above are still live. "
    "Try again in 60 seconds."
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect rate-limit errors across groq-sdk versions without importing
    internal classes (which differ between versions)."""
    cls = type(exc).__name__.lower()
    if "ratelimit" in cls:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def call_groq_with_retry(prompt: str) -> str:
    """Call Groq with one retry on rate-limit errors.

    Behaviour:
      - No API key set                       → return a clear "set GROQ_API_KEY" notice.
      - First attempt fails with rate-limit  → wait 10s, retry once.
      - Retry also fails with rate-limit     → return RATE_LIMIT_FALLBACK.
      - Other errors                         → return a brief failure message
                                                (the rest of the app keeps working).
    """
    if not GROQ_API_KEY:
        return (
            "⚠️ AI analysis disabled — `GROQ_API_KEY` is not set. "
            "All data, indicators, and predictions above are still live. "
            "Set the key as an environment variable or in `.streamlit/secrets.toml` "
            "to enable the LLM analysis."
        )

    attempts = [(0, "first attempt"), (10, "retry after 10s")]
    last_error: Exception | None = None

    for wait_secs, label in attempts:
        if wait_secs:
            time.sleep(wait_secs)
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2200,
            )
            content = response.choices[0].message.content
            return content or RATE_LIMIT_FALLBACK
        except Exception as e:
            last_error = e
            if not _is_rate_limit_error(e):
                # Non-rate-limit failure: don't waste a retry on it.
                return (
                    f"⚠️ AI analysis temporarily unavailable: {type(e).__name__}. "
                    "All data, indicators, and predictions above are still live."
                )
            # Otherwise we fall through to the next attempt (if any).

    # Both attempts rate-limited.
    return RATE_LIMIT_FALLBACK


# ============================================================
# COMPANY NAME → TICKER LOOKUP
# ============================================================
NAME_TO_TICKER = {
    # Tech
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL", "goog": "GOOG",
    "amazon": "AMZN", "amzn": "AMZN",
    "nvidia": "NVDA", "nvda": "NVDA",
    "meta": "META", "facebook": "META", "fb": "META",
    "tesla": "TSLA", "tsla": "TSLA",
    "reddit": "RDDT", "rddt": "RDDT",
    "amd": "AMD",
    "salesforce": "CRM", "crm": "CRM",
    "intel": "INTC", "intc": "INTC",
    "netflix": "NFLX", "nflx": "NFLX",
    "shopify": "SHOP", "shop": "SHOP",
    "uber": "UBER",
    "snap": "SNAP", "snapchat": "SNAP",
    "oracle": "ORCL", "orcl": "ORCL",
    "ibm": "IBM",
    "cisco": "CSCO", "csco": "CSCO",
    "adobe": "ADBE", "adbe": "ADBE",
    "paypal": "PYPL", "pypl": "PYPL",
    "palantir": "PLTR", "pltr": "PLTR",
    "spotify": "SPOT", "spot": "SPOT",
    "airbnb": "ABNB", "abnb": "ABNB",
    "doordash": "DASH", "dash": "DASH",
    "coinbase": "COIN", "coin": "COIN",
    "block": "SQ", "square": "SQ",
    "robinhood": "HOOD", "hood": "HOOD",
    "qualcomm": "QCOM", "qcom": "QCOM",
    "broadcom": "AVGO", "avgo": "AVGO",
    "micron": "MU", "mu": "MU",
    "tsmc": "TSM", "taiwan semi": "TSM", "tsm": "TSM",
    "asml": "ASML",
    "dell": "DELL",
    "hp": "HPQ", "hpq": "HPQ",
    "lyft": "LYFT",
    "pinterest": "PINS", "pins": "PINS",
    "roblox": "RBLX", "rblx": "RBLX",
    "zoom": "ZM", "zm": "ZM",
    "twilio": "TWLO", "twlo": "TWLO",
    "datadog": "DDOG", "ddog": "DDOG",
    "snowflake": "SNOW", "snow": "SNOW",
    "crowdstrike": "CRWD", "crwd": "CRWD",

    # Finance
    "jpmorgan": "JPM", "jp morgan": "JPM", "jpm": "JPM",
    "goldman sachs": "GS", "goldman": "GS", "gs": "GS",
    "visa": "V",
    "mastercard": "MA",
    "bank of america": "BAC", "bofa": "BAC", "bac": "BAC",
    "wells fargo": "WFC", "wfc": "WFC",
    "citi": "C", "citigroup": "C", "citibank": "C",
    "morgan stanley": "MS", "ms": "MS",
    "american express": "AXP", "amex": "AXP", "axp": "AXP",
    "berkshire": "BRK-B", "berkshire hathaway": "BRK-B", "brk": "BRK-B",
    "blackrock": "BLK", "blk": "BLK",
    "schwab": "SCHW", "charles schwab": "SCHW",

    # Retail / Consumer
    "walmart": "WMT", "wmt": "WMT",
    "costco": "COST",
    "target": "TGT", "tgt": "TGT",
    "home depot": "HD", "hd": "HD",
    "lowes": "LOW", "lowe's": "LOW",
    "nike": "NKE", "nke": "NKE",
    "starbucks": "SBUX", "sbux": "SBUX",
    "mcdonalds": "MCD", "mcdonald's": "MCD", "mcd": "MCD",
    "coca cola": "KO", "coke": "KO", "ko": "KO",
    "pepsi": "PEP", "pepsico": "PEP", "pep": "PEP",
    "disney": "DIS", "dis": "DIS",
    "ebay": "EBAY",

    # Healthcare
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ", "jnj": "JNJ",
    "pfizer": "PFE", "pfe": "PFE",
    "moderna": "MRNA", "mrna": "MRNA",
    "unitedhealth": "UNH", "united health": "UNH", "unh": "UNH",
    "eli lilly": "LLY", "lilly": "LLY", "lly": "LLY",
    "merck": "MRK", "mrk": "MRK",
    "abbvie": "ABBV", "abbv": "ABBV",

    # Defence / Industrial
    "lockheed": "LMT", "lockheed martin": "LMT", "lmt": "LMT",
    "raytheon": "RTX", "rtx": "RTX",
    "northrop": "NOC", "northrop grumman": "NOC", "noc": "NOC",
    "general dynamics": "GD", "gd": "GD",
    "boeing": "BA", "ba": "BA",
    "caterpillar": "CAT", "cat": "CAT",
    "ge": "GE", "general electric": "GE",
    "honeywell": "HON", "hon": "HON",
    "ford": "F",
    "general motors": "GM", "gm": "GM",

    # Energy
    "exxon": "XOM", "exxon mobil": "XOM", "xom": "XOM",
    "chevron": "CVX", "cvx": "CVX",
    "shell": "SHEL",
    "bp": "BP",

    # Real Estate (REITs)
    "realty income": "O",
    "american tower": "AMT", "amt": "AMT",
    "prologis": "PLD", "pld": "PLD",
    "simon property": "SPG", "spg": "SPG",
    "welltower": "WELL",
    "digital realty": "DLR", "dlr": "DLR",

    # Indices / ETFs
    "s&p 500": "SPY", "sp500": "SPY", "spy": "SPY",
    "nasdaq": "QQQ", "qqq": "QQQ",
    "dow": "DIA", "dia": "DIA",
    "russell": "IWM", "iwm": "IWM",
    "vti": "VTI", "total stock market": "VTI",
}


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_ticker(query):
    """Turn 'Apple' / 'apple inc.' / 'AAPL' into a valid ticker symbol."""
    if not query:
        return None
    q = query.strip().lower()

    # 1. Direct match in dictionary
    if q in NAME_TO_TICKER:
        return NAME_TO_TICKER[q]

    # 2. Strip common suffixes ("apple inc", "apple corp")
    for suffix in [" inc.", " inc", " corp.", " corp", " corporation",
                   " co.", " co", " ltd.", " ltd", " plc", " company",
                   " holdings", " group", " ag"]:
        if q.endswith(suffix):
            stripped = q[:-len(suffix)].strip()
            if stripped in NAME_TO_TICKER:
                return NAME_TO_TICKER[stripped]

    # 3. Try uppercase as a literal ticker (e.g. "GOOGL", "BRK-B")
    upper = query.strip().upper()
    if upper.replace("-", "").replace(".", "").isalpha() and 1 <= len(upper) <= 6:
        try:
            test = yf.Ticker(upper)
            hist = test.history(period="5d")
            if len(hist) > 0:
                return upper
        except Exception:
            pass

    # 4. Last resort — yfinance search
    try:
        results = yf.Search(query, max_results=3).quotes
        for r in results:
            sym = r.get('symbol')
            if sym and r.get('quoteType') in ('EQUITY', 'ETF'):
                return sym.upper()
    except Exception:
        pass

    return None


# ============================================================
# INDICATOR FUNCTIONS
# ============================================================
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger(data, period=20):
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    bb_position = (data['Close'] - lower) / (upper - lower)
    return upper, lower, bb_position

def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_stochastic(data, period=14):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=3).mean()
    return k, d

def calculate_drawdown(data):
    rolling_max = data['Close'].cummax()
    drawdown = (data['Close'] - rolling_max) / rolling_max * 100
    return drawdown

def calculate_returns(data):
    out = {}
    if len(data) < 2:
        return out
    last = data['Close'].iloc[-1]
    periods = {
        '1D':   1,
        '1W':   5,
        '1M':   21,
        '3M':   63,
        '6M':   126,
        'YTD':  None,
        '1Y':   252,
        '3Y':   252 * 3,
        '5Y':   252 * 5,
        '10Y':  252 * 10,
    }
    for label, days in periods.items():
        try:
            if label == 'YTD':
                this_year = data[data.index.year == data.index[-1].year]
                if len(this_year) > 1:
                    ref = this_year['Close'].iloc[0]
                    out[label] = (last - ref) / ref * 100
            elif days and len(data) > days:
                ref = data['Close'].iloc[-days - 1]
                out[label] = (last - ref) / ref * 100
        except Exception:
            continue
    return out

def calculate_risk_metrics(data):
    daily_ret = data['Close'].pct_change().dropna()
    if len(daily_ret) < 30:
        return {}
    annual_vol = daily_ret.std() * np.sqrt(252) * 100
    annual_ret = daily_ret.mean() * 252 * 100
    risk_free = 4.5
    sharpe = (annual_ret - risk_free) / annual_vol if annual_vol > 0 else 0
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(252) * 100 if len(downside) > 0 else annual_vol
    sortino = (annual_ret - risk_free) / downside_vol if downside_vol > 0 else 0
    rolling_max = data['Close'].cummax()
    drawdown = (data['Close'] - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    pos_days = (daily_ret > 0).sum() / len(daily_ret) * 100
    return {
        'annual_return': annual_ret,
        'annual_volatility': annual_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'positive_days': pos_days,
    }


# ============================================================
# DATA + NEWS
# ============================================================
def get_stock_news(ticker):
    """Pull recent news, defending against None values at every key."""
    s = yf.Ticker(ticker)
    try:
        news = s.news or []
        headlines = []
        for item in news[:10]:
            content = item.get('content') or {}
            provider = content.get('provider') or {}
            canonical = content.get('canonicalUrl') or {}
            headlines.append({
                'title':     content.get('title') or '',
                'publisher': provider.get('displayName') or '',
                'summary':   content.get('summary') or '',
                'date':      content.get('pubDate') or '',
                'link':      canonical.get('url') or '',
            })
        return headlines
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(ticker, period="10y"):
    s = yf.Ticker(ticker)
    d = s.history(period=period)
    info = s.info if hasattr(s, 'info') else {}
    news = get_stock_news(ticker)
    return d, info, news


def analyse_stock(ticker, period="10y"):
    d, info, news = fetch_stock_data(ticker, period)
    if d is None or len(d) < 200:
        return None

    d['RSI'] = calculate_rsi(d)
    d['MACD'], d['MACD_Signal'], d['MACD_Hist'] = calculate_macd(d)
    d['BB_Upper'], d['BB_Lower'], d['BB_Position'] = calculate_bollinger(d)
    d['SMA_50'] = d['Close'].rolling(window=50).mean()
    d['SMA_200'] = d['Close'].rolling(window=200).mean()
    d['EMA_20'] = d['Close'].ewm(span=20).mean()
    d['ATR'] = calculate_atr(d)
    d['Stoch_K'], d['Stoch_D'] = calculate_stochastic(d)
    d['Volume_Ratio'] = d['Volume'] / d['Volume'].rolling(window=20).mean()
    d['Volatility'] = d['Close'].pct_change().rolling(window=30).std()
    d['Daily_Return'] = d['Close'].pct_change()
    d['Drawdown'] = calculate_drawdown(d)

    latest = d.iloc[-1]
    prev = d.iloc[-2]
    day_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

    def _safe(v):
        try:
            x = float(v)
            return 0.0 if (np.isnan(x) or np.isinf(x)) else x
        except (TypeError, ValueError):
            return 0.0

    feature_values = [
        _safe(latest['RSI']),
        _safe(latest['MACD']),
        _safe(latest['MACD_Hist']),
        _safe(latest['BB_Position']),
        _safe(latest['Volume_Ratio']),
        _safe(latest['Volatility']),
        _safe(latest['Daily_Return']),
        _safe(info.get('trailingPE', 0)),
        _safe(info.get('revenueGrowth', 0)),
        _safe(info.get('profitMargins', 0)),
    ]
    features = pd.DataFrame([feature_values], columns=feature_cols)

    # Predict via numpy array — avoids the sklearn "X does not have valid feature names" UserWarning.
    prediction = ada.predict(features.values)[0]
    probability = ada.predict_proba(features.values)[0]
    confidence = max(probability) * 100
    direction = "UP" if prediction == 1 else "DOWN"

    returns = calculate_returns(d)
    risk = calculate_risk_metrics(d)

    news_text = "\n".join([
        f"- [{n['publisher']}] {n['title']}: {n['summary'][:160]}"
        for n in news[:6] if n['title']
    ])

    importance_text = "\n".join([
        f"  {i+1}. {feat}: {imp*100:.1f}% importance — current value: {features[feat].values[0]:.4f}"
        for i, (feat, imp) in enumerate(
            sorted(zip(feature_cols, ada.feature_importances_),
                   key=lambda x: x[1], reverse=True))
    ])

    returns_text = " | ".join([f"{k}: {v:+.1f}%" for k, v in returns.items()])

    fundamentals_text = f"""
  P/E (trailing):   {info.get('trailingPE', 'N/A')}
  P/E (forward):    {info.get('forwardPE', 'N/A')}
  PEG ratio:        {info.get('pegRatio', 'N/A')}
  Price/Book:       {info.get('priceToBook', 'N/A')}
  Price/Sales:      {info.get('priceToSalesTrailing12Months', 'N/A')}
  Revenue Growth:   {info.get('revenueGrowth', 'N/A')}
  Earnings Growth:  {info.get('earningsGrowth', 'N/A')}
  Profit Margin:    {info.get('profitMargins', 'N/A')}
  Operating Margin: {info.get('operatingMargins', 'N/A')}
  ROE:              {info.get('returnOnEquity', 'N/A')}
  ROA:              {info.get('returnOnAssets', 'N/A')}
  Debt/Equity:      {info.get('debtToEquity', 'N/A')}
  Current Ratio:    {info.get('currentRatio', 'N/A')}
  Dividend Yield:   {info.get('dividendYield', 'N/A')}
  Payout Ratio:     {info.get('payoutRatio', 'N/A')}
  Beta:             {info.get('beta', 'N/A')}
  Analyst Target:   ${info.get('targetMeanPrice', 'N/A')}
  Analyst Rating:   {info.get('recommendationKey', 'N/A')}
"""

    prompt = f"""You are a senior equity research analyst writing a comprehensive briefing for an INDIVIDUAL INVESTOR. The reader may be a beginner, so explain technical jargon in plain English the first time it appears (one short parenthetical is enough). Cite EVERY number you reference.

═══════════════════════════════════════════════
STOCK PROFILE
═══════════════════════════════════════════════
TICKER:        {ticker}
COMPANY:       {info.get('longName', ticker)}
SECTOR:        {info.get('sector', 'Unknown')}
INDUSTRY:      {info.get('industry', 'Unknown')}
MARKET CAP:    ${info.get('marketCap', 0):,.0f}
CURRENT PRICE: ${latest['Close']:.2f}   (today {day_change:+.2f}%)
52W RANGE:     ${info.get('fiftyTwoWeekLow', 0):.2f} – ${info.get('fiftyTwoWeekHigh', 0):.2f}
EMPLOYEES:     {info.get('fullTimeEmployees', 'N/A')}
DESCRIPTION:   {(info.get('longBusinessSummary', '') or '')[:600]}

═══════════════════════════════════════════════
ML MODEL OUTPUT  (AdaBoost, 30-day horizon)
═══════════════════════════════════════════════
PREDICTION:    {direction}
CONFIDENCE:    {confidence:.1f}%
NOTE: backtest accuracy ~53%. Treat as ONE input among many.

INDICATOR IMPORTANCE & VALUES:
{importance_text}

═══════════════════════════════════════════════
HISTORICAL RETURNS  (10-year window)
═══════════════════════════════════════════════
{returns_text}

═══════════════════════════════════════════════
RISK METRICS
═══════════════════════════════════════════════
Annualised Return:     {risk.get('annual_return', 0):.1f}%
Annualised Volatility: {risk.get('annual_volatility', 0):.1f}%
Sharpe Ratio:          {risk.get('sharpe', 0):.2f}
Sortino Ratio:         {risk.get('sortino', 0):.2f}
Max Drawdown:          {risk.get('max_drawdown', 0):.1f}%
Positive Days:         {risk.get('positive_days', 0):.1f}%

═══════════════════════════════════════════════
TREND SIGNALS
═══════════════════════════════════════════════
SMA 50:  ${latest.get('SMA_50', 0):.2f}    SMA 200: ${latest.get('SMA_200', 0):.2f}
RSI:     {latest['RSI']:.1f}     Stochastic %K: {latest.get('Stoch_K', 0):.1f}
ATR:     {latest.get('ATR', 0):.2f}  (avg daily range in $)
Trend:   {'GOLDEN CROSS (bullish)' if latest.get('SMA_50', 0) > latest.get('SMA_200', 0) else 'DEATH CROSS (bearish)'}

═══════════════════════════════════════════════
FUNDAMENTALS
═══════════════════════════════════════════════
{fundamentals_text}

═══════════════════════════════════════════════
RECENT NEWS  (last 6 headlines)
═══════════════════════════════════════════════
{news_text}

═══════════════════════════════════════════════
WRITE YOUR REPORT WITH THESE EXACT SECTIONS
═══════════════════════════════════════════════

### 📌 Verdict
One-line summary of the model's call and your own read of the data.

### 🏢 What the Company Does
2–3 sentences. Plain English. Imagine the reader has never heard of it.

### 📊 Technical Picture
Walk through the TOP 3 indicators by model importance. For each: state the value, explain what it means in plain English, and say whether it supports or contradicts the model's prediction.

### 💰 Fundamentals Health Check
Comment on valuation (P/E, P/B), growth (revenue, earnings), profitability (margins, ROE), and balance sheet (debt). Use the actual numbers. Flag anything unusual.

### 📈 Performance Track Record
Reference the 1Y, 3Y, 5Y, 10Y returns. Compare to a typical S&P 500 long-run return (~10%/yr). Comment on volatility and max drawdown — what would it have felt like to hold this stock?

### 🌍 Sector Context
Briefly position this stock within its sector and industry trends right now.

### 📰 News Signal
For each of the most relevant 2–3 headlines, explain why it matters to a shareholder.

### 🐂 Bull Case  (3 bullet points)
Strongest reasons price could rise over 6–12 months.

### 🐻 Bear Case  (3 bullet points)
Strongest reasons price could fall over 6–12 months.

### ⚠️ Key Risks  (5 bullet points)
Specific, concrete risks. Not generic.

### 👤 Who Is This For?
What kind of investor profile does this suit? (e.g. growth seekers, dividend hunters, defensive, speculative). What is it NOT good for?

### 🎓 Beginner Glossary
Pick 3 terms from your analysis a beginner might not know. Define each in one sentence.

RULES:
- NEVER give buy/sell/hold advice — analysis only.
- Cite numbers. No vague claims.
- Use Markdown headers (###) exactly as shown above.
- Keep total output under 1100 words, but be thorough."""

    analysis_text = call_groq_with_retry(prompt)

    return {
        'ticker': ticker,
        'company': info.get('longName', ticker),
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown'),
        'price': latest['Close'],
        'day_change': day_change,
        'direction': direction,
        'confidence': confidence,
        'analysis': analysis_text,
        'news': news,
        'data': d,
        'latest': latest,
        'info': info,
        'importance': dict(zip(feature_cols, ada.feature_importances_)),
        'features': features,
        'returns': returns,
        'risk': risk,
    }


# ============================================================
# PAGE CONFIG + CSS
# ============================================================
# SVG monogram logo for "PP" (Priyansh Patel) — used as favicon, sidebar and header brand
def pp_logo(size: int = 48, drop_shadow: bool = False) -> str:
    """Return the PP monogram SVG at an explicit `size`x`size` pixel size,
    rendered on ONE LINE so Streamlit's markdown parser treats it as inline HTML
    rather than breaking out of the surrounding <div> block.

    Why this matters: the previous version used a triple-quoted SVG with
    width="100%" height="100%". The leading/trailing newlines made
    markdown-it-py close the parent HTML container early, which caused two
    visible bugs: (1) the SVG expanded to viewport size, (2) the trailing
    </div> tags from the header markdown leaked through as plain text on
    the welcome screen.
    """
    grad_id = f"ppGrad{size}"
    glow_id = f"ppGlow{size}"
    style = "display:inline-block;vertical-align:middle;"
    if drop_shadow:
        style += "filter:drop-shadow(0 4px 12px rgba(14,165,233,0.4));"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
        f'width="{size}" height="{size}" style="{style}">'
        f'<defs>'
        f'<linearGradient id="{grad_id}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="#0ea5e9"/>'
        f'<stop offset="100%" stop-color="#0284c7"/>'
        f'</linearGradient>'
        f'<linearGradient id="{glow_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>'
        f'<stop offset="100%" stop-color="#0ea5e9" stop-opacity="0"/>'
        f'</linearGradient>'
        f'</defs>'
        f'<rect x="2" y="2" width="60" height="60" rx="14" ry="14" fill="url(#{grad_id})"/>'
        f'<rect x="2" y="2" width="60" height="32" rx="14" ry="14" fill="url(#{glow_id})"/>'
        f'<text x="32" y="42" text-anchor="middle" '
        f'font-family="Inter, Helvetica, Arial, sans-serif" '
        f'font-size="28" font-weight="800" fill="#ffffff" letter-spacing="-1.5">PP</text>'
        f'<circle cx="52" cy="14" r="3" fill="#10b981"/>'
        f'</svg>'
    )


# Backward-compat alias used by any remaining references.
PP_LOGO_SVG = pp_logo(48)

# NOTE: Streamlit's page_icon doesn't accept inline SVG or base64 data URIs reliably,
# so we use an emoji for the browser tab and render the PP SVG inside the app instead.
st.set_page_config(page_title="StockAI Terminal · PP", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

.stApp { background-color: #0a0a0f; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1500px; }

[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] * { font-family: 'JetBrains Mono', monospace; }

.terminal-header {
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
}
.terminal-title {
    font-size: 1.8rem; font-weight: 700; color: #f1f5f9;
    font-family: 'Inter', sans-serif; margin: 0;
    display: flex; align-items: center; gap: 10px;
}
.terminal-sub {
    font-size: 0.8rem; color: #64748b;
    font-family: 'JetBrains Mono', monospace; margin-top: 4px;
}
.terminal-live {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.7rem; color: #10b981;
    font-family: 'JetBrains Mono', monospace;
    background: #10b98112; padding: 3px 10px;
    border-radius: 6px; border: 1px solid #10b98130;
}
.terminal-live::before {
    content: ''; width: 6px; height: 6px;
    border-radius: 50%; background: #10b981;
    animation: blink 1.5s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.pred-card {
    border-radius: 12px; padding: 24px; text-align: center;
    border: 1px solid; position: relative; overflow: hidden;
}
.pred-up   { background: linear-gradient(135deg, #052e16 0%, #0a3622 100%); border-color: #10b98140; }
.pred-down { background: linear-gradient(135deg, #2d0a0a 0%, #3b1219 100%); border-color: #ef444440; }
.pred-direction { font-family: 'Inter', sans-serif; font-size: 2rem; font-weight: 700; margin: 0; }
.pred-confidence { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; margin: 4px 0; }
.pred-note { font-size: 0.7rem; color: #64748b; font-family: 'JetBrains Mono', monospace; }

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px; margin-bottom: 16px;
}
.metric-card {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 10px; padding: 14px 16px;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 1px;
}
.metric-value {
    font-family: 'Inter', sans-serif;
    font-size: 1.4rem; font-weight: 700; color: #f1f5f9;
    margin-top: 2px;
}
.metric-change-up { color: #10b981; }
.metric-change-down { color: #ef4444; }

.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 2px;
    padding: 16px 0 8px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 12px;
}

.imp-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0; border-bottom: 1px solid #111827;
}
.imp-name { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #94a3b8; min-width: 130px; }
.imp-bar-bg { flex: 1; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
.imp-bar { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
.imp-pct { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #f1f5f9; min-width: 45px; text-align: right; }
.imp-val { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748b; min-width: 65px; text-align: right; }

.news-card {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 10px; padding: 14px 18px;
    margin-bottom: 8px; transition: border-color 0.2s;
}
.news-card:hover { border-color: #334155; }
.news-publisher {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #0ea5e9;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.news-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem; font-weight: 500;
    color: #e2e8f0; margin: 4px 0; line-height: 1.4;
}
.news-summary { font-size: 0.78rem; color: #64748b; line-height: 1.5; }

.analysis-card {
    background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border: 1px solid #1e293b; border-radius: 12px;
    padding: 28px; font-family: 'Inter', sans-serif;
    font-size: 0.92rem; color: #cbd5e1; line-height: 1.75;
}
.analysis-card h3 {
    color: #f1f5f9; font-size: 1.05rem;
    margin: 20px 0 8px 0; padding-bottom: 4px;
    border-bottom: 1px solid #1e293b;
}
.analysis-card h3:first-child { margin-top: 0; }

.gauge-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
}
.gauge-card {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 10px; padding: 14px 16px; text-align: center;
}
.gauge-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.gauge-value { font-family: 'Inter', sans-serif; font-size: 1.6rem; font-weight: 700; margin: 4px 0; }
.gauge-status { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 2px 8px; border-radius: 4px; display: inline-block; }
.gauge-help { font-size: 0.65rem; color: #475569; margin-top: 6px; font-family: 'Inter', sans-serif; line-height: 1.3; }

.beginner-tip {
    background: #0c1f33;
    border-left: 3px solid #0ea5e9;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 10px 0;
    font-family: 'Inter', sans-serif;
    line-height: 1.55;
}
.beginner-tip strong { color: #0ea5e9; }

.glossary-card {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 8px;
}
.glossary-term {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; color: #0ea5e9;
    font-weight: 600; margin-bottom: 4px;
}
.glossary-def { font-size: 0.85rem; color: #cbd5e1; line-height: 1.55; }

.disclaimer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #475569;
    text-align: center; padding: 16px;
    border-top: 1px solid #1e293b; margin-top: 30px;
}

/* Welcome hero — prominent main-area search before any analysis is run */
.welcome-hero {
    text-align: center;
    padding: 40px 20px 12px;
    font-family: 'Inter', sans-serif;
}
.welcome-hero-icon { font-size: 3rem; margin-bottom: 12px; line-height: 1; }
.welcome-hero-title {
    font-size: 1.8rem; color: #f1f5f9;
    margin: 0 0 10px; font-weight: 700;
    letter-spacing: -0.5px;
}
.welcome-hero-sub {
    font-size: 0.95rem; color: #64748b;
    max-width: 620px; margin: 0 auto 6px;
    line-height: 1.6;
}
.welcome-popular-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 2px;
    text-align: center; margin: 28px 0 12px;
    padding-bottom: 8px; border-bottom: 1px solid #1e293b;
}
.welcome-hint {
    text-align: center; font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #475569;
    margin-top: 22px; line-height: 1.6;
}

/* Compact search at top of results — small "search another stock" row */
.compact-search-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 8px 0 6px;
}

/* Anchors used by the popular-grid scoping rules in the mobile media query */
.popular-grid-start, .popular-grid-end { display: none; }

.stTextInput input {
    background: #111827 !important; border: 1px solid #1e293b !important;
    color: #f1f5f9 !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.05rem !important; border-radius: 10px !important; padding: 10px 14px !important;
}
.stTextInput input:focus { border-color: #0ea5e9 !important; box-shadow: 0 0 0 1px #0ea5e9 !important; }
.stTextInput label, .stSelectbox label, .stRadio label {
    font-family: 'JetBrains Mono', monospace !important;
    color: #64748b !important; font-size: 0.7rem !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
}
.stButton button {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: white !important; font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important; border: none !important; border-radius: 10px !important;
    padding: 12px !important; font-size: 0.9rem !important;
    letter-spacing: 0.5px !important; transition: all 0.2s !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%) !important;
    transform: translateY(-1px) !important;
}
.stSpinner > div { color: #0ea5e9 !important; }
div[data-testid="stExpander"] {
    background: #111827 !important; border: 1px solid #1e293b !important; border-radius: 10px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 1px solid #1e293b;
    overflow-x: auto;
    overflow-y: hidden;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: thin;
    scrollbar-color: #1e293b transparent;
    flex-wrap: nowrap !important;
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { height: 4px; }
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
.stTabs [data-baseweb="tab"] {
    background: #111827 !important;
    color: #64748b !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 16px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.5px !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}
.stTabs [aria-selected="true"] {
    background: #0d1117 !important;
    color: #0ea5e9 !important;
    border-color: #0ea5e9 !important;
}

/* =========================================================
   RESPONSIVE — tablet, mobile, phone breakpoints
   ========================================================= */
@media (max-width: 1024px) {
    .block-container { padding: 1rem !important; max-width: 100% !important; }
    .terminal-header { padding: 16px 18px !important; }
    .terminal-title { font-size: 1.4rem !important; }
    .terminal-sub { font-size: 0.7rem !important; }
}

/* Mobile portrait + landscape */
@media (max-width: 768px) {
    .block-container { padding: 0.5rem !important; }

    /* Header stacks: logo+title on top, status block below */
    .terminal-header { padding: 14px 14px !important; }
    .terminal-header > div { flex-direction: column !important; align-items: flex-start !important; gap: 10px !important; }
    .terminal-title { font-size: 1.15rem !important; }
    .terminal-sub { font-size: 0.65rem !important; }

    /* Prediction card */
    .pred-card { padding: 16px !important; }
    .pred-direction { font-size: 1.35rem !important; }
    .pred-confidence { font-size: 0.95rem !important; }

    /* Cards */
    .metric-card { padding: 10px 12px !important; }
    .metric-value { font-size: 1.05rem !important; }
    .metric-label { font-size: 0.6rem !important; }

    /* Grids — force 2 cols on mobile so they don't go to one giant column */
    .metric-grid { grid-template-columns: repeat(2, 1fr) !important; gap: 8px !important; }
    .gauge-grid  { grid-template-columns: repeat(2, 1fr) !important; gap: 8px !important; }
    .gauge-value { font-size: 1.25rem !important; }
    .gauge-card  { padding: 10px 12px !important; }

    /* Tabs — scrollable strip on mobile */
    .stTabs [data-baseweb="tab"] {
        padding: 6px 10px !important;
        font-size: 0.7rem !important;
        letter-spacing: 0 !important;
    }

    /* Analysis card */
    .analysis-card { padding: 16px !important; font-size: 0.85rem !important; line-height: 1.65 !important; }
    .analysis-card h3 { font-size: 0.95rem !important; }

    /* News cards */
    .news-card { padding: 12px 14px !important; }
    .news-title { font-size: 0.85rem !important; }
    .news-summary { font-size: 0.72rem !important; }

    /* Streamlit horizontal columns stack vertically on mobile */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 10px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Importance bars compact */
    .imp-name { min-width: 80px !important; font-size: 0.7rem !important; }
    .imp-pct  { min-width: 38px !important; font-size: 0.65rem !important; }
    .imp-val  { min-width: 50px !important; font-size: 0.65rem !important; }

    /* Glossary cards */
    .glossary-card { padding: 12px 14px !important; }
    .glossary-term { font-size: 0.78rem !important; }
    .glossary-def  { font-size: 0.78rem !important; }

    /* Disclaimer */
    .disclaimer { font-size: 0.55rem !important; padding: 12px !important; }

    /* Radio row (chart range) wraps */
    div[role="radiogroup"] { flex-wrap: wrap !important; gap: 4px !important; }

    /* Hide sidebar PP logo block on mobile — redundant with main header */
    [data-testid="stSidebar"] .sidebar-logo { display: none !important; }

    /* Buttons get a 44px tap target on mobile (matches Apple HIG / Android guidelines) */
    .stButton button { min-height: 44px !important; padding: 10px 12px !important; font-size: 0.82rem !important; }

    /* Section header smaller */
    .section-header { font-size: 0.62rem !important; padding: 12px 0 6px !important; letter-spacing: 1.2px !important; }

    /* Welcome hero shrinks */
    .welcome-hero { padding: 22px 8px 8px !important; }
    .welcome-hero-icon { font-size: 2.2rem !important; }
    .welcome-hero-title { font-size: 1.2rem !important; }
    .welcome-hero-sub { font-size: 0.82rem !important; padding: 0 8px !important; }
    .welcome-popular-header { margin: 18px 0 10px !important; font-size: 0.62rem !important; }
    .welcome-hint { font-size: 0.6rem !important; margin-top: 16px !important; padding: 0 8px !important; }

    /* Plotly charts capped at 450px on mobile (was 650 desktop).
       Container hides any overflow so the chart's internal SVG re-fits on resize. */
    div[data-testid="stPlotlyChart"] {
        max-height: 460px !important;
        overflow: hidden !important;
    }
    div[data-testid="stPlotlyChart"] > div,
    div[data-testid="stPlotlyChart"] .js-plotly-plot,
    div[data-testid="stPlotlyChart"] .plot-container,
    div[data-testid="stPlotlyChart"] .user-select-none,
    div[data-testid="stPlotlyChart"] .svg-container {
        max-height: 450px !important;
    }

    /* Popular-grid scope: between .popular-grid-start and .popular-grid-end
       anchors, columns stay as a 2-up flex-wrap row instead of stacking. */
    [data-testid="element-container"]:has(.popular-grid-start) ~ [data-testid="element-container"] [data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
    }
    [data-testid="element-container"]:has(.popular-grid-start) ~ [data-testid="element-container"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="element-container"]:has(.popular-grid-start) ~ [data-testid="element-container"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 0 0 calc(50% - 4px) !important;
        width: calc(50% - 4px) !important;
        max-width: calc(50% - 4px) !important;
    }
    /* End marker resets following siblings back to default (column stack). */
    [data-testid="element-container"]:has(.popular-grid-end) ~ [data-testid="element-container"] [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        flex-wrap: nowrap !important;
    }
    [data-testid="element-container"]:has(.popular-grid-end) ~ [data-testid="element-container"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="element-container"]:has(.popular-grid-end) ~ [data-testid="element-container"] [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
    }
}

/* Small phone */
@media (max-width: 480px) {
    .block-container { padding: 0.25rem 0.5rem !important; }
    .terminal-title { font-size: 1rem !important; }
    .pred-direction { font-size: 1.15rem !important; }

    /* Single-column grids on tiny screens — better readability */
    .metric-grid { grid-template-columns: repeat(2, 1fr) !important; }
    .gauge-grid  { grid-template-columns: 1fr !important; }

    .stTabs [data-baseweb="tab"] {
        padding: 5px 8px !important;
        font-size: 0.65rem !important;
    }

    .news-card { padding: 10px 12px !important; }
    .analysis-card { padding: 14px !important; font-size: 0.82rem !important; }

    /* Sidebar buttons stack to single column */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }

    /* Even smaller hero on tiny phones */
    .welcome-hero-title { font-size: 1.05rem !important; }
    .welcome-hero-sub { font-size: 0.75rem !important; }
    .welcome-hero-icon { font-size: 1.9rem !important; }

    /* Tighter chart cap on the smallest screens */
    div[data-testid="stPlotlyChart"] { max-height: 420px !important; }
    div[data-testid="stPlotlyChart"] > div,
    div[data-testid="stPlotlyChart"] .js-plotly-plot,
    div[data-testid="stPlotlyChart"] .plot-container,
    div[data-testid="stPlotlyChart"] .svg-container { max-height: 410px !important; }

    /* Disclaimer shrinks further */
    .disclaimer { font-size: 0.5rem !important; padding: 8px !important; line-height: 1.4 !important; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
if "search_query" not in st.session_state:
    st.session_state.search_query = "Apple"
if "trigger_analysis" not in st.session_state:
    st.session_state.trigger_analysis = False


def _set_query(name):
    st.session_state.search_query = name
    st.session_state.trigger_analysis = True


def _welcome_search():
    """Callback for the welcome-screen search row in the main content area.
    If the user typed something, push it into the sidebar's search_query slot
    so the rest of the flow (analyse_btn / trigger_analysis) stays unchanged.
    Empty submit falls back to whatever is already in search_query."""
    val = (st.session_state.get("welcome_input") or "").strip()
    if val:
        st.session_state.search_query = val
    st.session_state.trigger_analysis = True


def _compact_search():
    """Same as _welcome_search but driven by the compact top-of-results bar.
    Lets mobile users switch tickers without opening the sidebar.

    Clearing compact_input here (inside the on_change callback, before the
    next rerun re-instantiates the widget) keeps the field empty for the
    follow-up search instead of leaving stale text behind."""
    val = (st.session_state.get("compact_input") or "").strip()
    if val:
        st.session_state.search_query = val
        st.session_state.compact_input = ""
    st.session_state.trigger_analysis = True


with st.sidebar:
    st.markdown(
        f"""<div class="sidebar-logo" style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">{pp_logo(42)}<div><div style="font-family:'Inter',sans-serif;font-weight:700;font-size:1.1rem;color:#f1f5f9;line-height:1.1;">StockAI Terminal</div><div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#64748b;letter-spacing:1.5px;">BY PP · ML + LLM RESEARCH</div></div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("**🔍 SEARCH**")
    search_query = st.text_input(
        "Search",
        key="search_query",
        placeholder="Type a company name or ticker (e.g. Apple, Reddit, NVDA)",
        label_visibility="collapsed",
    )

    st.markdown("**📅 HISTORY WINDOW**")
    period = st.select_slider(
        "Period",
        options=["1y", "3y", "5y", "10y"],
        value="10y",
        label_visibility="collapsed",
    )

    beginner_mode = st.toggle("🎓 Beginner mode (explanations on)", value=True)

    analyse_btn = st.button("⚡ ANALYSE", width="stretch")

    st.markdown("---")
    st.markdown("**⭐ POPULAR COMPANIES**  ")
    st.caption("Click any name to analyse instantly")

    popular_buttons = {
        "Tech":       ["Apple", "Microsoft", "Google", "Nvidia", "Amazon", "Meta", "Tesla", "Reddit", "Netflix", "AMD"],
        "Finance":    ["JPMorgan", "Goldman", "Visa", "Mastercard", "Berkshire"],
        "Retail":     ["Walmart", "Costco", "Nike", "Starbucks", "McDonalds", "Disney"],
        "Healthcare": ["Johnson & Johnson", "Pfizer", "Eli Lilly", "UnitedHealth"],
        "Defence":    ["Lockheed", "Boeing", "Raytheon", "Northrop"],
        "Energy":     ["Exxon", "Chevron"],
        "Indices":    ["S&P 500", "Nasdaq", "Dow"],
    }
    for sector_name, names in popular_buttons.items():
        st.markdown(
            f"<div style='color:#64748b;font-size:0.65rem;letter-spacing:1.5px;margin-top:10px;margin-bottom:4px;'>{sector_name.upper()}</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, name in enumerate(names):
            with cols[i % 2]:
                st.button(
                    name,
                    key=f"btn_{sector_name}_{name}",
                    on_click=_set_query,
                    args=(name,),
                    width="stretch",
                )

    st.markdown("---")
    with st.expander("ℹ️ About this app"):
        st.markdown(
            """
            **StockAI Terminal** combines:
            - AdaBoost ML model trained on 10y of data across 31 stocks (4 sectors)
            - Llama 3.3 70B via Groq for natural-language analysis
            - Live Yahoo Finance market data, news, and fundamentals

            **For education & research only.** Not financial advice.
            """
        )

    with st.expander("🔧 Model freshness"):
        info_ts = RETRAIN_INFO.get("retrained_at")
        info_acc = RETRAIN_INFO.get("accuracy")
        info_src = RETRAIN_INFO.get("source", "unknown")
        if info_ts is None:
            st.markdown("Model freshness data unavailable.")
        else:
            st.markdown(
                f"""
                **Last retrained:** `{info_ts.strftime('%Y-%m-%d %H:%M UTC')}`
                **Age:** `{humanize_age(info_ts)}`
                **Test accuracy:** `{f"{info_acc*100:.2f}%" if info_acc is not None else 'n/a'}`
                **Source:** `{info_src}`

                The model retrains automatically every Sunday via GitHub Actions
                (`.github/workflows/retrain.yml`). To retrain locally:
                ```
                python retrain.py
                ```
                """
            )


# ============================================================
# HEADER
# ============================================================
_ts = RETRAIN_INFO.get("retrained_at")
_acc = RETRAIN_INFO.get("accuracy")
_age_str = humanize_age(_ts) if _ts else "unknown"
_date_str = _ts.strftime("%Y-%m-%d") if _ts else "—"
_acc_str = f"{_acc*100:.1f}% acc" if _acc is not None else "accuracy n/a"
_source = RETRAIN_INFO.get("source", "unknown")
_stale_days = ((datetime.now(timezone.utc) - _ts).days if _ts else 999)
_stale_color = "#10b981" if _stale_days < 14 else ("#f59e0b" if _stale_days < 60 else "#ef4444")

st.markdown(
    f'<div class="terminal-header">'
    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
    f'<div style="display:flex; align-items:center; gap:16px;">'
    f'{pp_logo(56, drop_shadow=True)}'
    f'<div>'
    f'<p class="terminal-title">StockAI Terminal</p>'
    f'<p class="terminal-sub">AdaBoost ML · Llama 3.3 70B via Groq · 10y market history</p>'
    f'<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.6rem; color:#475569; letter-spacing:2px; margin-top:6px;">'
    f'A PROJECT BY <span style="color:#0ea5e9; font-weight:700;">PP</span> · PRIYANSH PATEL'
    f'</p>'
    f'</div>'
    f'</div>'
    f'<div style="display:flex; flex-direction:column; align-items:flex-end; gap:6px;">'
    f'<div class="terminal-live">LIVE</div>'
    f'<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.65rem; color:#64748b; text-align:right; line-height:1.4;">'
    f'<div style="color:{_stale_color};">● MODEL RETRAINED {_age_str.upper()}</div>'
    f'<div style="color:#475569;">{_date_str} · {_acc_str}</div>'
    f'</div>'
    f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ============================================================
# WELCOME STATE & RESULT PERSISTENCE
# ============================================================
# Decide whether to (re)run the analysis or just display the cached result.
# This is what stops the UI from blanking out whenever the user toggles a tab,
# changes the chart range, or flips beginner mode.
new_request = analyse_btn or st.session_state.get("trigger_analysis", False)
st.session_state.trigger_analysis = False

cached_result = st.session_state.get("last_result")
cached_key = st.session_state.get("last_key")

raw_query = (search_query or "Apple").strip()
request_key = f"{raw_query}|{period}"

if new_request:
    ticker = resolve_ticker(raw_query)
    if not ticker:
        st.error(
            f"❌ Couldn't find a stock matching **{raw_query}**. "
            "Try the full company name (e.g. *Apple*, *Reddit*) or its ticker symbol (e.g. *AAPL*, *RDDT*)."
        )
        st.stop()

    display_label = f"{raw_query} → {ticker}" if raw_query.upper() != ticker else ticker
    with st.spinner(f"⚡ Scanning {display_label} — pulling {period} data, computing indicators, running ML model, generating analysis..."):
        try:
            result = analyse_stock(ticker, period=period)
        except Exception as e:
            st.error(f"❌ Failed to analyse {ticker}: {e}")
            st.stop()

    if result is None:
        st.error(
            f"❌ Found ticker **{ticker}** but not enough price history to analyse it "
            f"(need at least 200 trading days). It might be a recent IPO."
        )
        st.stop()

    st.session_state.last_result = result
    st.session_state.last_key = request_key
elif cached_result is not None:
    result = cached_result
else:
    # No prior analysis — show welcome screen with main-area search + popular grid
    # so mobile users can start an analysis without opening the sidebar.
    st.markdown(
        """
        <div class="welcome-hero">
            <div class="welcome-hero-icon">📊</div>
            <div class="welcome-hero-title">Analyse any stock in seconds</div>
            <div class="welcome-hero-sub">
                Type a company name or ticker — get an ML-driven 30-day prediction,
                technical indicators, 10-year performance, risk metrics, fundamentals,
                bull/bear cases, and recent news, all explained in plain English.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prominent search input + analyse button.
    # On desktop these sit side-by-side; mobile stacks them via the existing
    # stHorizontalBlock flex-direction:column rule.
    hero_in, hero_btn = st.columns([5, 1])
    with hero_in:
        st.text_input(
            "Welcome search",
            key="welcome_input",
            placeholder="Type a company name or ticker — e.g. Apple, Tesla, NVDA",
            label_visibility="collapsed",
            on_change=_welcome_search,
        )
    with hero_btn:
        st.button(
            "⚡ ANALYSE",
            key="welcome_analyse_btn",
            on_click=_welcome_search,
            width="stretch",
        )

    st.markdown(
        '<div class="welcome-popular-header">⭐ Popular companies — tap to analyse</div>',
        unsafe_allow_html=True,
    )

    # Anchor for the CSS scope: between these two markers the mobile rules
    # turn st.columns into a 2-up wrapping grid instead of a vertical stack.
    st.markdown('<div class="popular-grid-start"></div>', unsafe_allow_html=True)

    welcome_picks = [
        "Apple", "Tesla", "Nvidia", "Reddit", "Google",
        "Amazon", "Microsoft", "Netflix", "JPMorgan", "Boeing",
    ]
    row_a = st.columns(5)
    for i, name in enumerate(welcome_picks[:5]):
        with row_a[i]:
            st.button(
                name,
                key=f"welcome_pop_{name}",
                on_click=_set_query,
                args=(name,),
                width="stretch",
            )
    row_b = st.columns(5)
    for i, name in enumerate(welcome_picks[5:]):
        with row_b[i]:
            st.button(
                name,
                key=f"welcome_pop_{name}",
                on_click=_set_query,
                args=(name,),
                width="stretch",
            )

    st.markdown('<div class="popular-grid-end"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="welcome-hint">Or open the sidebar (top-left ☰) for the full company list, '
        'history window, and beginner-mode toggle.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

ticker = result['ticker']


latest = result['latest']
d = result['data']
info = result['info']
returns = result['returns']
risk = result['risk']
change_color = "metric-change-up" if result['day_change'] >= 0 else "metric-change-down"
change_sign = "+" if result['day_change'] >= 0 else ""


# ============================================================
# COMPACT TOP-OF-PAGE SEARCH (so mobile can switch tickers
# without opening the sidebar)
# ============================================================
st.markdown('<div class="compact-search-label">🔍 Search another stock</div>', unsafe_allow_html=True)
cs_in, cs_btn = st.columns([5, 1])
with cs_in:
    st.text_input(
        "Compact search",
        key="compact_input",
        placeholder=f"Currently analysing {ticker} — type a different name or ticker to switch",
        label_visibility="collapsed",
        on_change=_compact_search,
    )
with cs_btn:
    st.button(
        "⚡ GO",
        key="compact_go_btn",
        on_click=_compact_search,
        width="stretch",
    )

# ============================================================
# TOP: PREDICTION + KEY METRICS
# ============================================================
pred_col, metrics_col = st.columns([1, 1])

with pred_col:
    pred_class = "pred-up" if result['direction'] == "UP" else "pred-down"
    pred_color = "#10b981" if result['direction'] == "UP" else "#ef4444"
    arrow = "↑" if result['direction'] == "UP" else "↓"
    st.markdown(f"""
    <div class="pred-card {pred_class}">
        <p class="pred-direction" style="color: {pred_color};">{arrow} MODEL PREDICTS {result['direction']}</p>
        <p class="pred-confidence" style="color: {pred_color};">{result['confidence']:.1f}% confidence</p>
        <p class="pred-note">30-day prediction · AdaBoost · ~53% backtest accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    if beginner_mode:
        st.markdown(
            """
            <div class="beginner-tip">
              <strong>What this means:</strong> The model thinks the stock is more likely to move in this
              direction over the next 30 days. Confidence near 50% means the model is unsure.
              Backtest accuracy is only ~53%, so treat this as ONE signal among many — not a recommendation.
            </div>
            """,
            unsafe_allow_html=True,
        )

with metrics_col:
    fifty_two_low = info.get('fiftyTwoWeekLow', 0) or 0
    fifty_two_high = info.get('fiftyTwoWeekHigh', 0) or 1
    range_pct = (result['price'] - fifty_two_low) / max(fifty_two_high - fifty_two_low, 1e-9) * 100

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Price</div>
            <div class="metric-value">${result['price']:.2f}</div>
            <div class="{change_color}" style="font-size:0.8rem;font-family:'JetBrains Mono',monospace;">{change_sign}{result['day_change']:.2f}% today</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Market cap</div>
            <div class="metric-value">${(info.get('marketCap', 0) or 0)/1e9:.1f}B</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">52W range</div>
            <div class="metric-value" style="font-size:0.95rem;">${fifty_two_low:.0f} – ${fifty_two_high:.0f}</div>
            <div style="font-size:0.7rem; color:#64748b; font-family:'JetBrains Mono',monospace;">{range_pct:.0f}% of range</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sector</div>
            <div class="metric-value" style="font-size:0.95rem;">{_esc(result['sector'])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-family:Inter; font-size:0.85rem; color:#94a3b8;'>"
        f"<strong style='color:#f1f5f9;'>{_esc(result['company'])}</strong> · "
        f"{_esc(result['industry'])}</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# TABS
# ============================================================
tab_ai, tab_charts, tab_perf, tab_risk, tab_fund, tab_news, tab_glossary = st.tabs([
    "🤖 AI Analysis",
    "📊 Charts & Indicators",
    "📈 Performance",
    "⚠️ Risk",
    "📋 Fundamentals",
    "📰 News",
    "🎓 Glossary",
])


# ------------------------------------------------------------
# TAB: AI ANALYSIS
# ------------------------------------------------------------
with tab_ai:
    st.markdown('<div class="section-header">🤖 AI equity research — Llama 3.3 70B</div>', unsafe_allow_html=True)
    if beginner_mode:
        st.markdown(
            """
            <div class="beginner-tip">
              <strong>How to read this report:</strong> It walks through what the company does, what the technical
              indicators show, the fundamentals (is the business healthy?), and a balanced bull-case / bear-case.
              The AI is not allowed to give you buy/sell advice — only to help you understand the data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render the LLM markdown report into our custom analysis-card HTML.
    fixed_lines = []
    for line in result['analysis'].split("\n"):
        stripped = line.strip()
        if stripped.startswith("### "):
            fixed_lines.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("- "):
            fixed_lines.append(f"<div style='margin-left:18px;'>• {stripped[2:]}</div>")
        elif stripped:
            fixed_lines.append(f"<p style='margin:6px 0;'>{stripped}</p>")
    analysis_html = "\n".join(fixed_lines)

    st.markdown(f"""
    <div class="analysis-card">
        {analysis_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🎯 What drove the model decision</div>', unsafe_allow_html=True)
    if beginner_mode:
        st.markdown(
            """
            <div class="beginner-tip">
              <strong>Feature importance</strong> tells you which inputs the ML model weighted most heavily.
              A higher % means that signal pulled more weight in producing today's prediction.
              The value column shows the actual current reading for this stock.
            </div>
            """,
            unsafe_allow_html=True,
        )

    sorted_imp = sorted(result['importance'].items(), key=lambda x: x[1], reverse=True)
    imp_colors = ['#0ea5e9', '#0ea5e9', '#0ea5e9', '#38bdf8', '#38bdf8',
                  '#64748b', '#64748b', '#64748b', '#475569', '#475569']
    imp_html = ""
    for idx, (feat, imp) in enumerate(sorted_imp):
        val = result['features'][feat].values[0]
        bar_width = imp * 100 * 2.8
        color = imp_colors[idx] if idx < len(imp_colors) else '#475569'
        imp_html += f"""
        <div class="imp-row">
            <span class="imp-name">{feat}</span>
            <div class="imp-bar-bg">
                <div class="imp-bar" style="width:{bar_width}%;background:{color};"></div>
            </div>
            <span class="imp-pct">{imp*100:.1f}%</span>
            <span class="imp-val">{val:.4f}</span>
        </div>"""
    st.markdown(imp_html, unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB: CHARTS & INDICATORS
# ------------------------------------------------------------
with tab_charts:
    st.markdown('<div class="section-header">⚡ Live indicators</div>', unsafe_allow_html=True)

    rsi_val = latest['RSI']
    rsi_status = ("OVERBOUGHT", "#ef4444", "#ef444420", "Stock has rallied hard — possibly due a pullback") if rsi_val > 70 \
        else ("OVERSOLD", "#10b981", "#10b98120", "Stock has sold off — possibly due a bounce") if rsi_val < 30 \
        else ("NEUTRAL", "#f59e0b", "#f59e0b20", "Momentum is in normal range")

    macd_val = latest['MACD']
    macd_status = ("BULLISH", "#10b981", "#10b98120", "Short-term avg above long-term avg") if macd_val > 0 \
        else ("BEARISH", "#ef4444", "#ef444420", "Short-term avg below long-term avg")

    bb_val = latest['BB_Position']
    bb_status = ("UPPER BAND", "#ef4444", "#ef444420", "Price near top of typical range") if bb_val > 0.8 \
        else ("LOWER BAND", "#10b981", "#10b98120", "Price near bottom of typical range") if bb_val < 0.2 \
        else ("MID RANGE", "#0ea5e9", "#0ea5e920", "Price in normal trading range")

    vol_val = latest['Volume_Ratio']
    vol_status = ("HIGH VOLUME", "#f59e0b", "#f59e0b20", "Unusually active trading day") if vol_val > 1.5 \
        else ("LOW VOLUME", "#64748b", "#64748b20", "Quieter than average") if vol_val < 0.7 \
        else ("NORMAL", "#0ea5e9", "#0ea5e920", "Volume in line with recent average")

    sma_cross = "GOLDEN CROSS" if latest.get('SMA_50', 0) > latest.get('SMA_200', 0) else "DEATH CROSS"
    sma_color = "#10b981" if sma_cross == "GOLDEN CROSS" else "#ef4444"
    sma_bg = "#10b98120" if sma_cross == "GOLDEN CROSS" else "#ef444420"
    sma_help = "Long-term bullish trend" if sma_cross == "GOLDEN CROSS" else "Long-term bearish trend"

    stoch_val = latest.get('Stoch_K', 50)
    stoch_status = ("OVERBOUGHT", "#ef4444", "#ef444420", "Closing near recent highs") if stoch_val > 80 \
        else ("OVERSOLD", "#10b981", "#10b98120", "Closing near recent lows") if stoch_val < 20 \
        else ("NEUTRAL", "#0ea5e9", "#0ea5e920", "In middle of recent range")

    st.markdown(f"""
    <div class="gauge-grid">
        <div class="gauge-card">
            <div class="gauge-label">RSI (14)</div>
            <div class="gauge-value" style="color:{rsi_status[1]};">{rsi_val:.1f}</div>
            <span class="gauge-status" style="background:{rsi_status[2]};color:{rsi_status[1]};">{rsi_status[0]}</span>
            {f'<div class="gauge-help">{rsi_status[3]}</div>' if beginner_mode else ''}
        </div>
        <div class="gauge-card">
            <div class="gauge-label">MACD</div>
            <div class="gauge-value" style="color:{macd_status[1]};">{macd_val:.2f}</div>
            <span class="gauge-status" style="background:{macd_status[2]};color:{macd_status[1]};">{macd_status[0]}</span>
            {f'<div class="gauge-help">{macd_status[3]}</div>' if beginner_mode else ''}
        </div>
        <div class="gauge-card">
            <div class="gauge-label">Bollinger</div>
            <div class="gauge-value" style="color:{bb_status[1]};">{bb_val:.2f}</div>
            <span class="gauge-status" style="background:{bb_status[2]};color:{bb_status[1]};">{bb_status[0]}</span>
            {f'<div class="gauge-help">{bb_status[3]}</div>' if beginner_mode else ''}
        </div>
        <div class="gauge-card">
            <div class="gauge-label">Stochastic %K</div>
            <div class="gauge-value" style="color:{stoch_status[1]};">{stoch_val:.1f}</div>
            <span class="gauge-status" style="background:{stoch_status[2]};color:{stoch_status[1]};">{stoch_status[0]}</span>
            {f'<div class="gauge-help">{stoch_status[3]}</div>' if beginner_mode else ''}
        </div>
        <div class="gauge-card">
            <div class="gauge-label">Volume</div>
            <div class="gauge-value" style="color:{vol_status[1]};">{vol_val:.2f}x</div>
            <span class="gauge-status" style="background:{vol_status[2]};color:{vol_status[1]};">{vol_status[0]}</span>
            {f'<div class="gauge-help">{vol_status[3]}</div>' if beginner_mode else ''}
        </div>
        <div class="gauge-card">
            <div class="gauge-label">SMA cross</div>
            <div class="gauge-value" style="color:{sma_color};font-size:1.1rem;">{sma_cross}</div>
            <span class="gauge-status" style="background:{sma_bg};color:{sma_color};">50 vs 200</span>
            {f'<div class="gauge-help">{sma_help}</div>' if beginner_mode else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Price action</div>', unsafe_allow_html=True)

    chart_range = st.radio(
        "Chart range",
        options=["3M", "6M", "1Y", "3Y", "5Y", "10Y"],
        index=2,
        horizontal=True,
        label_visibility="collapsed",
    )
    range_map = {"3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "10Y": 2520}
    recent = d.tail(range_map[chart_range])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        row_heights=[0.55, 0.2, 0.25])

    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent['Open'], high=recent['High'],
        low=recent['Low'], close=recent['Close'], name='Price',
        increasing_line_color='#10b981', decreasing_line_color='#ef4444',
        increasing_fillcolor='#10b981', decreasing_fillcolor='#ef4444',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=recent.index, y=recent['BB_Upper'],
        line=dict(color='rgba(14,165,233,0.15)', width=1), name='BB Upper', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['BB_Lower'],
        line=dict(color='rgba(14,165,233,0.15)', width=1), fill='tonexty',
        fillcolor='rgba(14,165,233,0.04)', name='BB Lower', showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=recent.index, y=recent['SMA_50'],
        line=dict(color='#f59e0b', width=1.5, dash='dot'), name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['SMA_200'],
        line=dict(color='#ef4444', width=1.5, dash='dot'), name='SMA 200'), row=1, col=1)

    vol_colors = ['#10b981' if c >= o else '#ef4444' for c, o in zip(recent['Close'], recent['Open'])]
    fig.add_trace(go.Bar(x=recent.index, y=recent['Volume'],
        marker_color=vol_colors, opacity=0.5, name='Volume', showlegend=False), row=2, col=1)

    macd_colors = ['#10b981' if v >= 0 else '#ef4444' for v in recent['MACD_Hist']]
    fig.add_trace(go.Bar(x=recent.index, y=recent['MACD_Hist'],
        marker_color=macd_colors, name='MACD Hist', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MACD'],
        line=dict(color='#0ea5e9', width=1.5), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MACD_Signal'],
        line=dict(color='#f97316', width=1.5), name='Signal'), row=3, col=1)

    fig.update_layout(
        height=650, template='plotly_dark',
        paper_bgcolor='#0a0a0f', plot_bgcolor='#0d1117',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10, color='#64748b', family='JetBrains Mono')),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=20, t=30, b=20),
        font=dict(family='JetBrains Mono', color='#64748b'),
    )
    fig.update_xaxes(gridcolor='#1e293b', zeroline=False)
    fig.update_yaxes(gridcolor='#1e293b', zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"responsive": True, "displayModeBar": False})

    if beginner_mode:
        st.markdown(
            """
            <div class="beginner-tip">
              <strong>How to read this chart:</strong> Green candles = price closed above where it opened (up day),
              red = down day. The dotted lines are the 50-day and 200-day moving averages (long-term trend).
              The shaded blue band is the Bollinger Band — when price touches the top/bottom edges it's unusually
              high/low. Below: volume bars, then MACD (momentum).
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# TAB: PERFORMANCE
# ------------------------------------------------------------
with tab_perf:
    st.markdown('<div class="section-header">📈 Historical returns</div>', unsafe_allow_html=True)
    if beginner_mode:
        st.markdown(
            """
            <div class="beginner-tip">
              <strong>How to read this:</strong> Each box shows the total % gain or loss over that period.
              Compare against the S&P 500 long-run average of ~10%/year. Big drawdowns in here would have
              <em>felt</em> awful even if the long-run number looks great.
            </div>
            """,
            unsafe_allow_html=True,
        )

    ret_html = '<div class="metric-grid">'
    for label, value in returns.items():
        color = "#10b981" if value >= 0 else "#ef4444"
        sign = "+" if value >= 0 else ""
        ret_html += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{sign}{value:.1f}%</div>
        </div>"""
    ret_html += '</div>'
    st.markdown(ret_html, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📉 Long-term price curve</div>', unsafe_allow_html=True)
    initial = d['Close'].iloc[0]
    growth = d['Close'] / initial * 10000

    years_actual = (d.index[-1] - d.index[0]).days / 365.25

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=d.index, y=growth,
        line=dict(color='#0ea5e9', width=2),
        fill='tozeroy', fillcolor='rgba(14,165,233,0.08)',
        name='$10k invested',
    ))
    fig2.update_layout(
        height=400, template='plotly_dark',
        paper_bgcolor='#0a0a0f', plot_bgcolor='#0d1117',
        title=dict(text=f"Growth of $10,000 invested in {ticker} ({years_actual:.1f} years)", font=dict(color='#94a3b8', size=14)),
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=30),
        font=dict(family='JetBrains Mono', color='#64748b'),
    )
    fig2.update_xaxes(gridcolor='#1e293b', zeroline=False)
    fig2.update_yaxes(gridcolor='#1e293b', zeroline=False, tickprefix='$', tickformat=',.0f')
    st.plotly_chart(fig2, width="stretch", config={"responsive": True, "displayModeBar": False})

    final_value = growth.iloc[-1]
    multiplier = final_value / 10000
    st.markdown(
        f"<div style='font-family:Inter; font-size:0.95rem; color:#cbd5e1; text-align:center; padding:10px;'>"
        f"$10,000 invested {len(d)} trading days ago would be worth "
        f"<strong style='color:#10b981;'>${final_value:,.0f}</strong> "
        f"<span style='color:#64748b;'>({multiplier:.2f}x)</span></div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# TAB: RISK
# ------------------------------------------------------------
with tab_risk:
    st.markdown('<div class="section-header">⚠️ Risk metrics</div>', unsafe_allow_html=True)
    if beginner_mode:
        st.markdown(
            """
            <div class="beginner-tip">
              <strong>Risk metrics in plain English:</strong><br>
              • <strong>Volatility</strong> = how bumpy the ride is. Above ~25% is high.<br>
              • <strong>Sharpe ratio</strong> = return per unit of risk. Above 1 is good, above 2 is excellent.<br>
              • <strong>Max drawdown</strong> = the worst peak-to-trough fall you'd have endured. -50% means
              you'd have watched half your money vanish before recovering.<br>
              • <strong>Positive days</strong> = % of trading days the stock closed up. ~52–55% is normal.
            </div>
            """,
            unsafe_allow_html=True,
        )

    sharpe_color = "#10b981" if risk.get('sharpe', 0) > 1 else "#f59e0b" if risk.get('sharpe', 0) > 0 else "#ef4444"
    vol_color = "#ef4444" if risk.get('annual_volatility', 0) > 35 else "#f59e0b" if risk.get('annual_volatility', 0) > 22 else "#10b981"
    dd_color = "#ef4444" if risk.get('max_drawdown', 0) < -40 else "#f59e0b" if risk.get('max_drawdown', 0) < -25 else "#10b981"

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Annual return</div>
            <div class="metric-value" style="color:{'#10b981' if risk.get('annual_return',0) > 0 else '#ef4444'};">{risk.get('annual_return', 0):.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Annual volatility</div>
            <div class="metric-value" style="color:{vol_color};">{risk.get('annual_volatility', 0):.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe ratio</div>
            <div class="metric-value" style="color:{sharpe_color};">{risk.get('sharpe', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sortino ratio</div>
            <div class="metric-value" style="color:{sharpe_color};">{risk.get('sortino', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max drawdown</div>
            <div class="metric-value" style="color:{dd_color};">{risk.get('max_drawdown', 0):.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Positive days</div>
            <div class="metric-value">{risk.get('positive_days', 0):.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Beta (vs SPX)</div>
            <div class="metric-value">{f"{info['beta']:.2f}" if isinstance(info.get('beta'), (int, float)) else 'N/A'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ATR (avg range)</div>
            <div class="metric-value">${latest.get('ATR', 0):.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📉 Drawdown curve</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=d.index, y=d['Drawdown'],
        line=dict(color='#ef4444', width=1.5),
        fill='tozeroy', fillcolor='rgba(239,68,68,0.15)',
        name='Drawdown',
    ))
    fig3.update_layout(
        height=300, template='plotly_dark',
        paper_bgcolor='#0a0a0f', plot_bgcolor='#0d1117',
        title=dict(text="Drawdown from peak (%)", font=dict(color='#94a3b8', size=13)),
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=30),
        font=dict(family='JetBrains Mono', color='#64748b'),
    )
    fig3.update_xaxes(gridcolor='#1e293b', zeroline=False)
    fig3.update_yaxes(gridcolor='#1e293b', zeroline=False, ticksuffix='%')
    st.plotly_chart(fig3, width="stretch", config={"responsive": True, "displayModeBar": False})

    st.markdown('<div class="section-header">📊 Daily return distribution</div>', unsafe_allow_html=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(
        x=d['Daily_Return'].dropna() * 100,
        nbinsx=80,
        marker_color='#0ea5e9',
        opacity=0.7,
    ))
    fig4.update_layout(
        height=300, template='plotly_dark',
        paper_bgcolor='#0a0a0f', plot_bgcolor='#0d1117',
        title=dict(text="Distribution of daily returns (%)", font=dict(color='#94a3b8', size=13)),
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=30),
        font=dict(family='JetBrains Mono', color='#64748b'),
    )
    fig4.update_xaxes(gridcolor='#1e293b', zeroline=False, ticksuffix='%')
    fig4.update_yaxes(gridcolor='#1e293b', zeroline=False)
    st.plotly_chart(fig4, width="stretch", config={"responsive": True, "displayModeBar": False})


# ------------------------------------------------------------
# TAB: FUNDAMENTALS
# ------------------------------------------------------------
with tab_fund:
    st.markdown('<div class="section-header">📋 Valuation</div>', unsafe_allow_html=True)
    if beginner_mode:
        st.markdown(
            """
            <div class="beginner-tip">
              <strong>Valuation in plain English:</strong>
              <strong>P/E ratio</strong> = price per $1 of annual earnings. Lower can mean cheaper.
              <strong>P/B</strong> = price vs. book value (assets minus liabilities).
              <strong>PEG</strong> = P/E adjusted for growth — below 1 is often considered cheap-for-growth.
            </div>
            """,
            unsafe_allow_html=True,
        )

    def fmt_pct(v):
        if v is None or v == 'N/A':
            return 'N/A'
        try:
            return f"{float(v)*100:.1f}%"
        except Exception:
            return 'N/A'

    def fmt_pct_smart(v):
        """For fields like dividendYield that yfinance returns as either decimal
        (0.025) or percent (2.5) depending on version. Treat anything ≥ 1 as
        already-percent (>=100% dividend yield is unrealistic)."""
        if v is None or v == 'N/A':
            return 'N/A'
        try:
            f = float(v)
            if f == 0:
                return '0.0%'
            return f"{f:.2f}%" if abs(f) >= 1 else f"{f*100:.2f}%"
        except Exception:
            return 'N/A'

    def fmt_num(v, prefix="", suffix=""):
        if v is None or v == 'N/A':
            return 'N/A'
        try:
            return f"{prefix}{float(v):.2f}{suffix}"
        except Exception:
            return 'N/A'

    valuation = {
        'P/E (trailing)':  fmt_num(info.get('trailingPE')),
        'P/E (forward)':   fmt_num(info.get('forwardPE')),
        'PEG ratio':       fmt_num(info.get('pegRatio')),
        'Price/Book':      fmt_num(info.get('priceToBook')),
        'Price/Sales':     fmt_num(info.get('priceToSalesTrailing12Months')),
        'EV/EBITDA':       fmt_num(info.get('enterpriseToEbitda')),
        '52w High':        fmt_num(info.get('fiftyTwoWeekHigh'), prefix="$"),
        '52w Low':         fmt_num(info.get('fiftyTwoWeekLow'),  prefix="$"),
    }

    fund_html = '<div class="metric-grid">'
    for label, value in valuation.items():
        fund_html += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:1.1rem;">{value}</div>
        </div>"""
    fund_html += '</div>'
    st.markdown(fund_html, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📈 Growth & Profitability</div>', unsafe_allow_html=True)
    growth_prof = {
        'Revenue Growth':   fmt_pct(info.get('revenueGrowth')),
        'Earnings Growth':  fmt_pct(info.get('earningsGrowth')),
        'EPS (TTM)':        fmt_num(info.get('trailingEps'), prefix="$"),
        'Profit Margin':    fmt_pct(info.get('profitMargins')),
        'Operating Margin': fmt_pct(info.get('operatingMargins')),
        'Gross Margin':     fmt_pct(info.get('grossMargins')),
        'ROE':              fmt_pct(info.get('returnOnEquity')),
        'ROA':              fmt_pct(info.get('returnOnAssets')),
    }
    gp_html = '<div class="metric-grid">'
    for label, value in growth_prof.items():
        gp_html += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:1.1rem;">{value}</div>
        </div>"""
    gp_html += '</div>'
    st.markdown(gp_html, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🏦 Balance Sheet & Dividend</div>', unsafe_allow_html=True)
    balance = {
        'Debt/Equity':       fmt_num(info.get('debtToEquity')),
        'Current Ratio':     fmt_num(info.get('currentRatio')),
        'Quick Ratio':       fmt_num(info.get('quickRatio')),
        'Total Cash':        f"${(info.get('totalCash', 0) or 0)/1e9:.1f}B" if info.get('totalCash') else 'N/A',
        'Total Debt':        f"${(info.get('totalDebt', 0) or 0)/1e9:.1f}B" if info.get('totalDebt') else 'N/A',
        'Dividend Yield':    fmt_pct_smart(info.get('dividendYield')),
        'Payout Ratio':      fmt_pct(info.get('payoutRatio')),
        'Beta':              fmt_num(info.get('beta')),
    }
    bal_html = '<div class="metric-grid">'
    for label, value in balance.items():
        bal_html += f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:1.1rem;">{value}</div>
        </div>"""
    bal_html += '</div>'
    st.markdown(bal_html, unsafe_allow_html=True)

    st.markdown('<div class="section-header">👥 Analyst Consensus</div>', unsafe_allow_html=True)
    target_low = info.get('targetLowPrice', 'N/A')
    target_high = info.get('targetHighPrice', 'N/A')
    target_mean = info.get('targetMeanPrice', 'N/A')
    rec_key = info.get('recommendationKey') or 'N/A'
    _na = info.get('numberOfAnalystOpinions')
    num_analysts = str(_na) if _na is not None else 'N/A'

    analyst_html = f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Avg target</div>
            <div class="metric-value">{fmt_num(target_mean, prefix="$")}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Low target</div>
            <div class="metric-value" style="color:#ef4444;">{fmt_num(target_low, prefix="$")}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">High target</div>
            <div class="metric-value" style="color:#10b981;">{fmt_num(target_high, prefix="$")}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Consensus</div>
            <div class="metric-value" style="font-size:1.1rem; text-transform:uppercase;">{_esc(rec_key)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label"># Analysts</div>
            <div class="metric-value">{num_analysts}</div>
        </div>
    </div>"""
    st.markdown(analyst_html, unsafe_allow_html=True)

    if info.get('longBusinessSummary'):
        with st.expander("🏢 Business description"):
            st.markdown(
                f"<div style='font-family:Inter; font-size:0.9rem; color:#cbd5e1; line-height:1.7;'>"
                f"{_esc(info['longBusinessSummary'])}</div>",
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------
# TAB: NEWS
# ------------------------------------------------------------
with tab_news:
    st.markdown('<div class="section-header">📰 Market intelligence</div>', unsafe_allow_html=True)
    if not result['news']:
        st.info("No recent news available for this ticker.")
    for n in result['news']:
        summary = n.get('summary') or ''
        title = n.get('title') or '(untitled)'
        publisher = n.get('publisher') or 'Unknown'
        link = n.get('link') or ''
        link_safe = html.escape(link, quote=True)
        link_html = (
            f'<a href="{link_safe}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#0ea5e9;font-size:0.75rem;text-decoration:none;">Read full article →</a>'
            if link else ''
        )
        date_text = ""
        raw_date = n.get('date') or ''
        if raw_date:
            try:
                date_text = datetime.fromisoformat(raw_date.replace('Z', '+00:00')).strftime("%b %d, %Y")
            except Exception:
                date_text = raw_date[:10]

        trimmed = summary[:280] + ('...' if len(summary) > 280 else '')
        st.markdown(f"""
        <div class="news-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="news-publisher">{_esc(publisher)}</div>
                <div style="font-size:0.7rem; color:#475569; font-family:'JetBrains Mono',monospace;">{_esc(date_text)}</div>
            </div>
            <div class="news-title">{_esc(title)}</div>
            <div class="news-summary">{_esc(trimmed)}</div>
            <div style="margin-top:8px;">{link_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB: GLOSSARY
# ------------------------------------------------------------
with tab_glossary:
    st.markdown('<div class="section-header">🎓 Beginner glossary — every term explained</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="beginner-tip">
          New to investing? Start here. Every technical term used in this app is defined below in plain English.
        </div>
        """,
        unsafe_allow_html=True,
    )

    glossary = [
        ("📈 Stock / Equity", "A share representing partial ownership of a company. Owning 1 share of Apple makes you a (very) small Apple owner."),
        ("💰 Market Cap", "Total dollar value of all the company's shares. Calculated as price × shares outstanding. Big cap (>$200B), Mid ($10–200B), Small (<$10B)."),
        ("📊 P/E Ratio (Price/Earnings)", "How much investors pay for each $1 the company earns annually. P/E of 30 = $30 paid per $1 of yearly profit. Lower can mean cheaper — but cheap can mean troubled."),
        ("⏩ Forward P/E", "Same as P/E but based on *expected* future earnings instead of past. Useful for fast-growing companies."),
        ("📐 PEG Ratio", "P/E divided by growth rate. PEG < 1 is sometimes considered cheap-for-growth. Above 2 is often expensive-for-growth."),
        ("📚 P/B Ratio (Price/Book)", "Price compared to the company's net worth on its balance sheet. P/B < 1 means you'd pay less than the company's accounting value."),
        ("📉 RSI (Relative Strength Index)", "Momentum indicator from 0–100. Above 70 = overbought (might fall), below 30 = oversold (might bounce). Around 50 is neutral."),
        ("〰️ MACD", "Compares short-term momentum (12-day avg) vs long-term (26-day). Positive = bullish, negative = bearish, crossings often signal trend changes."),
        ("🎯 Bollinger Bands", "Price boundaries based on volatility. When price touches the upper band, it's unusually high; lower band = unusually low."),
        ("📏 SMA (Simple Moving Average)", "Average price over X days. SMA 50 = 50-day avg. SMA 200 = 200-day avg. When 50 crosses ABOVE 200 = 'golden cross' (bullish). Below = 'death cross' (bearish)."),
        ("💥 Volatility", "How much the price swings. High volatility = bumpy ride. Measured as annualised standard deviation of daily returns."),
        ("📊 Beta", "How much the stock moves vs the overall market. Beta 1 = moves with market. Beta 2 = twice as volatile. Beta 0.5 = half as volatile."),
        ("⭐ Sharpe Ratio", "Return earned per unit of risk taken. Above 1 = good. Above 2 = excellent. Below 0 = you'd be better off in T-bills."),
        ("📉 Max Drawdown", "Biggest peak-to-trough drop. If a stock has -50% max drawdown, holders watched half their money vanish before any recovery."),
        ("💵 Dividend Yield", "Annual dividends paid as % of share price. 3% yield = $3 yearly per $100 invested. High yields can signal trouble — be careful."),
        ("⚖️ Debt/Equity", "Company's debt as a multiple of shareholder equity. Above 100 (i.e. 1.0×) means more debt than equity — riskier in downturns."),
        ("💹 ROE (Return on Equity)", "Profit generated per $1 of shareholder equity. Above 15% is generally good. Tells you how efficiently the company uses owner money."),
        ("💰 Profit Margin", "% of revenue that becomes profit. Software companies often >20%. Grocery stores often 1–3%. Sector context matters."),
        ("🌊 Volume Ratio", "Today's trading volume vs the 20-day average. Above 1.5× often signals important news or big moves."),
        ("🎢 ATR (Average True Range)", "Typical daily price range in dollars. Useful for setting stop-loss distances or sizing positions."),
        ("🔮 30-day Prediction", "The ML model's guess at whether the stock will end higher or lower 30 trading days from now. Backtest accuracy ~53%."),
    ]
    for term, definition in glossary:
        st.markdown(f"""
        <div class="glossary-card">
            <div class="glossary-term">{term}</div>
            <div class="glossary-def">{definition}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# DISCLAIMER
# ============================================================
st.markdown(
    f'<div class="disclaimer">'
    f'⚠️ FOR EDUCATIONAL &amp; RESEARCH PURPOSES ONLY · NOT FINANCIAL ADVICE · '
    f'MODEL ACCURACY ~53% · PAST PERFORMANCE ≠ FUTURE RESULTS · '
    f'NEVER INVEST BASED SOLELY ON ALGORITHMIC OUTPUT<br>'
    f'<span style="display:inline-flex; align-items:center; gap:8px; margin-top:8px;">'
    f'{pp_logo(18)}'
    f'<span>Built by <strong style="color:#94a3b8;">Priyansh Patel (PP)</strong> · '
    f'AdaBoost ML + Llama 3.3 70B via Groq · Data via Yahoo Finance</span>'
    f'</span>'
    f'</div>',
    unsafe_allow_html=True,
)
