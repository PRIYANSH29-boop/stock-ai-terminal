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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
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

# NOTE: PE_Ratio, Revenue_Growth, Profit_Margin were dropped to remove lookahead.
# yfinance .info returns CURRENT snapshot values that were being stamped onto all
# 10 years of history — the model could "see the future." yfinance does expose
# .quarterly_financials, but coverage is shallow (~5 quarters in the free tier),
# the values get restated, and proper publication-lag alignment isn't reliably
# doable without a paid point-in-time data source. Falling back to (a): drop them.
FEATURE_COLS = [
    "RSI", "MACD", "MACD_Hist", "BB_Position",
    "Volume_Ratio", "Volatility", "Daily_Return",
]
DROPPED_FUNDAMENTAL_COLS = ["PE_Ratio", "Revenue_Growth", "Profit_Margin"]

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
def split_train_test_with_embargo(
    df: pd.DataFrame,
    train_frac: float = TRAIN_TEST_SPLIT,
    embargo: int = PREDICTION_HORIZON_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-ticker chronological 80/20 split with an embargo at the boundary.

    The target is Close.shift(-embargo)/Close - 1 > 0, so a training row at
    index i depends on prices at i+embargo. Without an embargo the last
    `embargo` training rows have forward windows that overlap test-period
    prices — classic label leakage. Per-ticker so the embargo doesn't bleed
    across the seams of pd.concat'd frames.
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for ticker, sub in df.groupby("Ticker", sort=False):
        sub = sub.sort_index()  # chronological within ticker
        n = len(sub)
        split = int(n * train_frac)
        train_cut = max(0, split - embargo)
        train_parts.append(sub.iloc[:train_cut])
        test_parts.append(sub.iloc[split:])
    train = pd.concat(train_parts) if train_parts else df.iloc[:0]
    test = pd.concat(test_parts) if test_parts else df.iloc[:0]
    return train, test


def train_model(master_df: pd.DataFrame):
    """Per-ticker time-based split with embargo → AdaBoost → honest metrics."""
    df = master_df.dropna(subset=FEATURE_COLS + ["Target"]).copy()

    train_df, test_df = split_train_test_with_embargo(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df["Target"].astype(int)
    X_test, y_test = test_df[FEATURE_COLS], test_df["Target"].astype(int)

    baseline_up_train = float(y_train.mean()) if len(y_train) else float("nan")
    baseline_up_test = float(y_test.mean()) if len(y_test) else float("nan")

    print("\n" + "=" * 60)
    print(f"TRAINING (per-ticker time-based split, embargo={PREDICTION_HORIZON_DAYS} rows)")
    print("=" * 60)
    print(f"Total usable rows: {len(df):,}")
    print(f"Training set:      {len(X_train):,} rows (after embargo)")
    print(f"Test set:          {len(X_test):,} rows")
    print(f"Train target UP:   {baseline_up_train*100:.2f}%")
    print(f"Test  target UP:   {baseline_up_test*100:.2f}%  ← always-UP baseline")

    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    )
    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)
    y_proba = ada.predict_proba(X_test)[:, 1]  # P(class=UP)

    accuracy = float(accuracy_score(y_test, y_pred))
    baseline_always_up_acc = baseline_up_test  # accuracy of "always predict UP"

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0], average=None, zero_division=0
    )
    down_precision = float(prec[0])
    down_recall = float(rec[0])
    down_f1 = float(f1[0])

    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    print("\n" + "=" * 60)
    print("EVALUATION (honest, after leakage fixes)")
    print("=" * 60)
    print(f"Test accuracy:         {accuracy*100:6.2f}%")
    print(f"Always-UP baseline:    {baseline_always_up_acc*100:6.2f}%")
    print(f"Lift over baseline:    {(accuracy - baseline_always_up_acc)*100:+6.2f} pp")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print(f"DOWN-class precision:  {down_precision:.4f}")
    print(f"DOWN-class recall:     {down_recall:.4f}")
    print(f"DOWN-class F1:         {down_f1:.4f}")
    print()
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"                  pred DOWN   pred UP")
    print(f"  actual DOWN     {tn:9d}   {fp:7d}")
    print(f"  actual UP       {fn:9d}   {tp:7d}")
    print()
    print("Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"], zero_division=0))

    importance_pairs = sorted(
        zip(FEATURE_COLS, ada.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("Feature importance ranking:")
    for feat, imp in importance_pairs:
        bar = "█" * int(imp * 100)
        print(f"  {feat:20s} {imp*100:5.1f}%  {bar}")

    test_slice = test_df.copy()
    test_slice["Predicted"] = y_pred
    print("\nAccuracy by sector (model vs always-UP):")
    sector_accuracy: dict[str, float] = {}
    for sector in test_slice["Sector"].unique():
        sub = test_slice[test_slice["Sector"] == sector]
        sect_acc = float(accuracy_score(sub["Target"], sub["Predicted"]))
        sect_base = float(sub["Target"].mean())
        sector_accuracy[sector] = sect_acc
        print(
            f"  {sector:15s} model {sect_acc*100:5.1f}%   "
            f"always-UP {sect_base*100:5.1f}%   ({len(sub):,} rows)"
        )

    metrics = {
        "accuracy": accuracy,
        "baseline_always_up_acc": float(baseline_always_up_acc),
        "lift_over_baseline_pp": float((accuracy - baseline_always_up_acc) * 100),
        "roc_auc": roc_auc,
        "down_precision": down_precision,
        "down_recall": down_recall,
        "down_f1": down_f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_up_pct": float(baseline_up_train * 100),
        "test_up_pct": float(baseline_up_test * 100),
        # Kept for back-compat with previous log entries / app.py:
        "target_up_pct": float(baseline_up_test * 100),
        "importance": {feat: float(imp) for feat, imp in importance_pairs},
        "sector_accuracy": {k: float(v) for k, v in sector_accuracy.items()},
        "split_method": "per_ticker_time_based_with_embargo",
        "embargo_rows": int(PREDICTION_HORIZON_DAYS),
        "fundamentals_dropped": DROPPED_FUNDAMENTAL_COLS,
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
        "baseline_always_up_acc": metrics["baseline_always_up_acc"],
        "lift_over_baseline_pp": metrics["lift_over_baseline_pp"],
        "roc_auc": metrics["roc_auc"],
        "down_precision": metrics["down_precision"],
        "down_recall": metrics["down_recall"],
        "down_f1": metrics["down_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "train_rows": metrics["train_rows"],
        "test_rows": metrics["test_rows"],
        "train_up_pct": metrics["train_up_pct"],
        "test_up_pct": metrics["test_up_pct"],
        "target_up_pct": metrics["target_up_pct"],  # kept for back-compat
        "importance": metrics["importance"],
        "sector_accuracy": metrics["sector_accuracy"],
        "split_method": metrics["split_method"],
        "embargo_rows": metrics["embargo_rows"],
        "fundamentals_dropped": metrics["fundamentals_dropped"],
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
