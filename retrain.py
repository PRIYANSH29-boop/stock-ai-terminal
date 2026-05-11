"""
retrain.py — Refresh the AdaBoost stock-direction model on the latest 10y of data.

Usage:
    python retrain.py

Outputs:
    ada_model.pkl       — trained AdaBoost classifier
    model_config.pkl    — feature_cols + importance dict (consumed by app.py)
    retrain_log.json    — append-only history of every retrain run

Designed to run unattended (locally or in CI). On any unrecoverable error the
script exits non-zero so a GitHub Action can fail loudly.
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
STOCKS = {
    "Tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "RDDT", "AMD", "CRM", "INTC", "NFLX", "SHOP", "UBER", "SNAP",
    ],
    "Finance":     ["JPM", "GS", "V", "MA", "PYPL"],
    "Real Estate": ["O", "AMT", "PLD", "SPG", "WELL", "DLR"],
    "Defence":     ["LMT", "RTX", "NOC", "GD", "BA"],
}

FEATURE_COLS = [
    "RSI", "MACD", "MACD_Hist", "BB_Position",
    "Volume_Ratio", "Volatility", "Daily_Return",
    "PE_Ratio", "Revenue_Growth", "Profit_Margin",
]

MIN_ROWS_PER_STOCK = 200       # minimum trading days required to include a stock
PREDICTION_HORIZON_DAYS = 30   # target = "did price go up 30 trading days later?"
TRAIN_TEST_SPLIT = 0.80
PERIOD = "10y"

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ada_model.pkl"
CONFIG_PATH = ROOT / "model_config.pkl"
LOG_PATH = ROOT / "retrain_log.json"


# ============================================================
# INDICATORS
# ============================================================
def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data: pd.DataFrame):
    ema12 = data["Close"].ewm(span=12).mean()
    ema26 = data["Close"].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger(data: pd.DataFrame, period: int = 20):
    sma = data["Close"].rolling(window=period).mean()
    std = data["Close"].rolling(window=period).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    bb_position = (data["Close"] - lower) / (upper - lower)
    return upper, lower, bb_position


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["RSI"] = calculate_rsi(df)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calculate_macd(df)
    df["BB_Upper"], df["BB_Lower"], df["BB_Position"] = calculate_bollinger(df)
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(window=20).mean()
    df["Volatility"] = df["Close"].pct_change().rolling(window=30).std()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Future_Return"] = df["Close"].shift(-PREDICTION_HORIZON_DAYS) / df["Close"] - 1
    df["Target"] = (df["Future_Return"] > 0).astype(int)
    return df


# ============================================================
# DATA PULL
# ============================================================
def fetch_stock(ticker: str, sector: str) -> pd.DataFrame | None:
    """Pull 10y of price data + fundamentals for one ticker. Returns None on failure."""
    try:
        s = yf.Ticker(ticker)
        d = s.history(period=PERIOD)
        if len(d) < MIN_ROWS_PER_STOCK:
            print(f"  {ticker:6s}  SKIPPED ({len(d)} rows, need {MIN_ROWS_PER_STOCK})")
            return None

        d = add_all_indicators(d)

        info = s.info or {}
        d["Ticker"] = ticker
        d["Sector"] = sector
        d["Market_Cap"] = info.get("marketCap", 0) or 0
        d["PE_Ratio"] = info.get("trailingPE", 0) or 0
        d["Revenue_Growth"] = info.get("revenueGrowth", 0) or 0
        d["Profit_Margin"] = info.get("profitMargins", 0) or 0

        mc = d["Market_Cap"].iloc[-1]
        d["Cap_Size"] = "Large" if mc > 200e9 else "Mid" if mc > 10e9 else "Small"

        print(f"  {ticker:6s}  OK   ({len(d):5d} rows, {sector})")
        return d
    except Exception as e:
        print(f"  {ticker:6s}  ERROR: {e}")
        return None


def build_master_dataframe() -> tuple[pd.DataFrame, dict[str, int]]:
    """Pull every stock, return concatenated DataFrame and per-ticker row counts."""
    all_frames = []
    row_counts: dict[str, int] = {}

    print("=" * 60)
    print(f"PULLING {sum(len(v) for v in STOCKS.values())} STOCKS — period={PERIOD}")
    print("=" * 60)

    for sector, tickers in STOCKS.items():
        print(f"\n[{sector}]")
        for ticker in tickers:
            df = fetch_stock(ticker, sector)
            if df is not None:
                all_frames.append(df)
                row_counts[ticker] = len(df)

    if not all_frames:
        raise RuntimeError("No stocks returned data — cannot train.")

    master = pd.concat(all_frames)
    return master, row_counts


# ============================================================
# TRAIN
# ============================================================
def train_model(master_df: pd.DataFrame):
    """Time-based 80/20 split → AdaBoost → return (model, metrics_dict)."""
    df = master_df.dropna(subset=FEATURE_COLS + ["Target"]).copy()

    X = df[FEATURE_COLS]
    y = df["Target"]

    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Total usable rows: {len(X):,}")
    print(f"Training set:      {len(X_train):,} rows")
    print(f"Test set:          {len(X_test):,} rows")
    print(f"Target balance:    UP {y.mean()*100:.1f}%  |  DOWN {(1-y.mean())*100:.1f}%")

    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    )
    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Test accuracy: {accuracy*100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    # Feature importance ranking
    importance_pairs = sorted(
        zip(FEATURE_COLS, ada.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("Feature importance ranking:")
    for feat, imp in importance_pairs:
        bar = "█" * int(imp * 100)
        print(f"  {feat:20s} {imp*100:5.1f}%  {bar}")

    # Accuracy by sector
    test_slice = df.iloc[split_idx:].copy()
    test_slice["Predicted"] = y_pred
    print("\nAccuracy by sector:")
    sector_accuracy = {}
    for sector in test_slice["Sector"].unique():
        sub = test_slice[test_slice["Sector"] == sector]
        sect_acc = accuracy_score(sub["Target"], sub["Predicted"])
        sector_accuracy[sector] = sect_acc
        print(f"  {sector:15s} {sect_acc*100:5.1f}%  ({len(sub):,} rows)")

    metrics = {
        "accuracy": float(accuracy),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_up_pct": float(y.mean() * 100),
        "importance": {feat: float(imp) for feat, imp in importance_pairs},
        "sector_accuracy": {k: float(v) for k, v in sector_accuracy.items()},
    }
    return ada, metrics


# ============================================================
# PERSIST
# ============================================================
def save_model(ada, importance: dict[str, float]) -> None:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ada, f)
    with open(CONFIG_PATH, "wb") as f:
        pickle.dump({"feature_cols": FEATURE_COLS, "importance": importance}, f)
    print(f"\n✓ Saved model     → {MODEL_PATH.name}")
    print(f"✓ Saved config    → {CONFIG_PATH.name}")


def append_log(metrics: dict, row_counts: dict[str, int]) -> dict:
    """Append a new entry to retrain_log.json and return that entry."""
    entry = {
        "retrained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "period": PERIOD,
        "horizon_days": PREDICTION_HORIZON_DAYS,
        "tickers_used": list(row_counts.keys()),
        "tickers_skipped": [
            t for tickers in STOCKS.values() for t in tickers if t not in row_counts
        ],
        "rows_per_ticker": row_counts,
        "total_rows": int(sum(row_counts.values())),
        "accuracy": metrics["accuracy"],
        "train_rows": metrics["train_rows"],
        "test_rows": metrics["test_rows"],
        "target_up_pct": metrics["target_up_pct"],
        "importance": metrics["importance"],
        "sector_accuracy": metrics["sector_accuracy"],
    }

    history = []
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                history = loaded
            else:
                history = [loaded]
        except Exception:
            history = []

    history.append(entry)
    # Keep last 50 retrain entries to bound file size.
    history = history[-50:]

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"✓ Logged retrain  → {LOG_PATH.name}  ({len(history)} entries)")
    return entry


# ============================================================
# MAIN
# ============================================================
def main() -> int:
    started = datetime.now(timezone.utc)
    print(f"\n⚡ retrain.py — {started.isoformat(timespec='seconds')}\n")

    master_df, row_counts = build_master_dataframe()

    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total stocks used: {len(row_counts)}")
    print(f"Total rows:        {sum(row_counts.values()):,}")
    print(f"Sectors:           {list(master_df['Sector'].unique())}")
    print("Rows per sector:")
    print(master_df.groupby("Sector")["Ticker"].nunique().to_string())

    ada, metrics = train_model(master_df)
    save_model(ada, metrics["importance"])
    entry = append_log(metrics, row_counts)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"\n✓ Retrain complete in {elapsed:.1f}s — accuracy {entry['accuracy']*100:.2f}%")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Retrain FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
