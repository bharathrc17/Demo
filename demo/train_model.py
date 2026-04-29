"""
train_model.py — Offline XGBoost Model Training for Nifty 50 & BankNifty
==========================================================================

Trains TWO separate binary XGBoost classifiers:
  - Nifty 50   : data/nifty_historical.csv   -> models/xgb_nifty.pkl
  - BankNifty  : data/banknifty_historical.csv -> models/xgb_banknifty.pkl

Both share the same feature column list (saved once to
models/feature_columns.json).

Pipeline steps (per index):
    1. Load historical OHLCV data
    2. Compute 25 technical indicator features (via compute_features)
    3. Create binary labels: 1 (BUY) if close +0.5% in 3 candles, else 0
    4. Chronological 80/20 train/test split (no shuffle -- time series!)
    5. Train XGBClassifier with class-imbalance weighting
    6. Evaluate -- classification report, confusion matrix, feature importance
    7. Backtest simulation with brokerage costs
    8. Save model (.pkl)

Usage:
    python train_model.py
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from src.features import FEATURE_COLUMNS, compute_features  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NIFTY_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "nifty_historical.csv")
BANKNIFTY_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "banknifty_historical.csv")

MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
NIFTY_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_nifty.pkl")
BANKNIFTY_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_banknifty.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

LOOKAHEAD = 3              # candles to look ahead for labelling
LABEL_THRESHOLD = 0.005    # 0.5% rise -> BUY
TRAIN_RATIO = 0.80         # 80% train, 20% test
SIGNAL_PROB_THRESHOLD = 0.68
BROKERAGE_COST = 0.0006    # 0.03% buy + 0.03% sell round-trip


def _separator(title: str) -> None:
    """Print a section banner."""
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


# ======================================================================
# STEP 1 -- Load data
# ======================================================================
def load_data(path: str, index_label: str) -> pd.DataFrame:
    """Load OHLCV CSV, parse dates, sort chronologically."""
    _separator(f"STEP 1 -- Load Data ({index_label})")

    if not os.path.exists(path):
        print(f"  [FAIL] File not found: {path}")
        print(f"    Run 'python download_data.py' to fetch {index_label} historical data.")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Loaded {len(df)} rows from {os.path.basename(path)}")
    print(f"  Date range: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    print(f"  Columns: {list(df.columns)}")

    return df


# ======================================================================
# STEP 2 -- Feature engineering
# ======================================================================
def engineer_features(df: pd.DataFrame, index_label: str) -> pd.DataFrame:
    """Compute all technical indicators via compute_features()."""
    _separator(f"STEP 2 -- Feature Engineering ({index_label})")

    df = compute_features(df)

    print(f"  Shape after compute_features: {df.shape}")
    print(f"  Feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    return df


# ======================================================================
# STEP 3 -- Create labels
# ======================================================================
def create_labels(df: pd.DataFrame, index_label: str) -> pd.DataFrame:
    """
    Binary label: 1 (BUY) if close +0.5% in 3 candles, else 0 (SELL).
    Last 3 rows are dropped (no future data available).
    """
    _separator(f"STEP 3 -- Create Labels ({index_label})")

    df = df.copy()
    df["future_close"] = df["close"].shift(-LOOKAHEAD)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    # Binary label
    df["label"] = (df["future_return"] >= LABEL_THRESHOLD).astype(int)

    # Drop rows without future data
    df.dropna(subset=["future_close"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Remove helper columns
    df.drop(columns=["future_close", "future_return"], inplace=True)

    # Class distribution
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    print(f"  Total samples : {total}")
    print(f"  Label 0 (SELL): {counts.get(0, 0)}  ({counts.get(0, 0)/total*100:.1f}%)")
    print(f"  Label 1 (BUY) : {counts.get(1, 0)}  ({counts.get(1, 0)/total*100:.1f}%)")

    return df


# ======================================================================
# STEP 4 -- Chronological train/test split
# ======================================================================
def split_data(df: pd.DataFrame, index_label: str):
    """
    First 80% for training, last 20% for testing.  Never shuffled --
    preserves temporal order to prevent look-ahead bias.
    """
    _separator(f"STEP 4 -- Train / Test Split ({index_label})")

    split_idx = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    print(f"  Train size: {len(X_train)} rows  ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Test  size: {len(X_test)} rows  ({(1-TRAIN_RATIO)*100:.0f}%)")
    print(f"  Train label distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"  Test  label distribution: {dict(y_test.value_counts().sort_index())}")

    return X_train, y_train, X_test, y_test, test_df


# ======================================================================
# STEP 5 -- Train XGBoost
# ======================================================================
def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                index_label: str) -> XGBClassifier:
    """
    Train an XGBClassifier with class-imbalance weighting.
    scale_pos_weight = count(SELL) / count(BUY).
    """
    _separator(f"STEP 5 -- Train XGBoost ({index_label})")

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    spw = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"  scale_pos_weight: {spw:.4f}  (neg={neg_count}, pos={pos_count})")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=1,
    )

    print("  Training started ...")
    model.fit(X_train, y_train)
    print("  [OK] Training complete.")

    return model


# ======================================================================
# STEP 6 -- Evaluate
# ======================================================================
def evaluate_model(model: XGBClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   index_label: str) -> None:
    """Classification report, confusion matrix, top-10 feature importances."""
    _separator(f"STEP 6 -- Evaluate on Test Set ({index_label})")

    y_pred = model.predict(X_test)

    # Classification report
    print("\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["SELL (0)", "BUY (1)"])
    for line in report.split("\n"):
        print(f"  {line}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n  Confusion Matrix:")
    print(f"                 Predicted SELL  Predicted BUY")
    print(f"  Actual SELL    {cm[0][0]:>14}  {cm[0][1]:>13}")
    print(f"  Actual BUY     {cm[1][0]:>14}  {cm[1][1]:>13}")

    # Accuracy
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    print(f"\n  Accuracy: {accuracy*100:.2f}%")

    # Feature importance (top 10)
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(FEATURE_COLUMNS, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n  Top 10 Feature Importances:")
    for rank, (feat, imp) in enumerate(feat_imp[:10], start=1):
        bar = "#" * int(imp * 100)
        print(f"    {rank:>2}. {feat:<16} {imp:.4f}  {bar}")


# ======================================================================
# STEP 7 -- Backtest simulation
# ======================================================================
def backtest(model: XGBClassifier,
             X_test: pd.DataFrame,
             test_df: pd.DataFrame,
             index_label: str) -> None:
    """
    Simulate trades on the test set.
    Entry = close at signal row, exit = close 3 candles later.
    """
    _separator(f"STEP 7 -- Backtest Simulation ({index_label})")

    probas = model.predict_proba(X_test)[:, 1]  # P(BUY)

    trades: list[dict] = []
    close_prices = test_df["close"].values

    for i in range(len(X_test)):
        if probas[i] > SIGNAL_PROB_THRESHOLD:
            exit_idx = i + LOOKAHEAD
            if exit_idx >= len(close_prices):
                continue

            entry_price = close_prices[i]
            exit_price = close_prices[exit_idx]
            gross_return = (exit_price - entry_price) / entry_price
            net_pnl = gross_return - BROKERAGE_COST

            trades.append({
                "entry_idx": i,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_pnl": net_pnl,
            })

    if not trades:
        print("  [WARN] No trades triggered (probability threshold too high?).")
        return

    trades_df = pd.DataFrame(trades)
    total_signals = len(trades_df)
    wins = int((trades_df["net_pnl"] > 0).sum())
    win_rate = wins / total_signals * 100
    avg_pnl = trades_df["net_pnl"].mean() * 100
    total_return = trades_df["net_pnl"].sum() * 100

    # Max drawdown
    cumulative = (1 + trades_df["net_pnl"]).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_dd = drawdowns.min() * 100

    print(f"  Signal threshold    : predict_proba(BUY) > {SIGNAL_PROB_THRESHOLD}")
    print(f"  Lookahead candles   : {LOOKAHEAD}")
    print(f"  Brokerage (round)   : {BROKERAGE_COST*100:.2f}%")
    print(f"  -------------------------------------------")
    print(f"  Total signals       : {total_signals}")
    print(f"  Winning trades      : {wins}")
    print(f"  Win rate            : {win_rate:.2f}%")
    print(f"  Avg PnL per trade   : {avg_pnl:+.4f}%")
    print(f"  Total return (cum.) : {total_return:+.4f}%")
    print(f"  Max drawdown        : {max_dd:.4f}%")


# ======================================================================
# STEP 8 -- Save model
# ======================================================================
def save_model(model: XGBClassifier, model_path: str, index_label: str) -> None:
    """Save a trained model to disk."""
    _separator(f"STEP 8 -- Save Model ({index_label})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"  [OK] Model saved to: {model_path}")


def save_feature_columns() -> None:
    """Save the shared feature column list (once for both models)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(FEATURE_COLS_PATH, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print(f"  [OK] Feature columns ({len(FEATURE_COLUMNS)}) saved to: {FEATURE_COLS_PATH}")


# ======================================================================
# Full pipeline for a single index
# ======================================================================
def train_single_index(data_path: str, model_path: str, index_label: str) -> None:
    """Run the full 8-step pipeline for one index."""
    print(f"\n{'#' * 64}")
    print(f"  Training Pipeline: {index_label}")
    print(f"{'#' * 64}")

    # Step 1
    df = load_data(data_path, index_label)

    # Step 2
    df = engineer_features(df, index_label)

    # Step 3
    df = create_labels(df, index_label)

    # Step 4
    X_train, y_train, X_test, y_test, test_df = split_data(df, index_label)

    # Step 5
    model = train_model(X_train, y_train, index_label)

    # Step 6
    evaluate_model(model, X_test, y_test, index_label)

    # Step 7
    backtest(model, X_test, test_df, index_label)

    # Step 8
    save_model(model, model_path, index_label)


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    """Execute the full training pipeline for both Nifty 50 and BankNifty."""
    print("\n" + "=" * 64)
    print("  Nifty 50 & BankNifty XGBoost -- Model Training Pipeline")
    print("=" * 64)

    # Train Nifty 50
    train_single_index(NIFTY_DATA_PATH, NIFTY_MODEL_PATH, "NIFTY")

    # Train BankNifty
    train_single_index(BANKNIFTY_DATA_PATH, BANKNIFTY_MODEL_PATH, "BANKNIFTY")

    # Save shared feature columns (once)
    _separator("Save Shared Feature Columns")
    save_feature_columns()

    print("\n" + "=" * 64)
    print("  Pipeline complete -- both models saved.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
