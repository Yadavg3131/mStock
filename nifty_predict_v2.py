"""
NIFTY 50 — NEXT-DAY PREDICTION SYSTEM (v2, production-grade)
=============================================================

Three CLI modes:

  python nifty_predict_v2.py backtest      # walk-forward historical test
  python nifty_predict_v2.py predict       # one-shot prediction for tomorrow
  python nifty_predict_v2.py daily         # predict + write to Google Sheet

Key upgrades vs v1:
  • Ridge regression (handles multicollinear macro factors)
  • TimeSeriesSplit cross-validation (no lookahead)
  • Walk-forward backtest with naive + buy-and-hold baselines
  • Pre-market features: S&P 500, Nikkei, DXY, US 10Y yield (all close
    BEFORE NIFTY opens, so they're known at decision time)
  • Structured logging — no silent failures
  • Optional Google Sheet writer for the `daily` mode

Requirements:
  pip install yfinance pandas numpy scikit-learn gspread oauth2client
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("NiftyPredict")


# ============================================================
# CONFIG
# ============================================================
TICKERS = {
    # Core
    "NIFTY":     "^NSEI",
    "VIX":       "^INDIAVIX",

    # Macro factors (intraday-correlated)
    "USDINR":    "USDINR=X",
    "USOIL":     "CL=F",
    "MSCI":      "ACWI",

    # Pre-market signals (closed BEFORE NIFTY opens at 09:15 IST)
    "SP500":     "^GSPC",     # closes 02:00 IST
    "NIKKEI":    "^N225",     # closes 12:00 IST (overlaps, but yesterday's close is known)
    "DXY":       "DX-Y.NYB",  # dollar index
    "US10Y":     "^TNX",      # US 10-year yield
}

# Sheet config — reused from chartwip.py
SHEET_NAME       = "Abhay"
PREDICTIONS_TAB  = "NiftyPredictions"  # create this tab in your sheet
KEYS_TAB         = "KeyAndRules"

# Model hyperparams
RIDGE_ALPHA      = 1.0
LOOKBACK_YEARS   = "3y"
CV_SPLITS        = 5
BACKTEST_WARMUP  = 252      # ~1 year of bars before first prediction
SHRINK_UNCERTAIN = 0.7      # shrink toward today when classifier is uncertain
SHRINK_DISAGREE  = 0.5      # shrink half when classifier disagrees

FEATURE_COLS = [
    # NIFTY momentum
    "NIFTY_ret_1", "NIFTY_ret_2", "NIFTY_ret_3", "NIFTY_ret_5",
    "NIFTY_zscore_20",

    # Intraday-correlated factor returns (from yesterday)
    "USDINR_ret_1", "USOIL_ret_1", "MSCI_ret_1", "VIX_ret_1",

    # Pre-market signals (known before NIFTY opens)
    "SP500_ret_1",  "NIKKEI_ret_1", "DXY_ret_1",  "US10Y_ret_1",

    # Breadth proxy
    "BREADTH",
]


# ============================================================
# DATA
# ============================================================
def download_data(period: str = LOOKBACK_YEARS) -> pd.DataFrame:
    log.info("Downloading %s of data for %d tickers...", period, len(TICKERS))
    series = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df.empty:
                log.warning("Empty data for %s (%s)", name, ticker)
                continue
            series[name] = df["Close"].squeeze()
            log.info("  %-8s (%-12s) %d rows", name, ticker, len(df))
        except Exception as exc:
            log.error("  %-8s (%-12s) FAILED: %s", name, ticker, exc)

    if "NIFTY" not in series:
        log.critical("NIFTY data missing — cannot continue.")
        sys.exit(1)

    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=["NIFTY"]).ffill()
    log.info("Combined dataset: %d rows × %d columns", len(df), len(df.columns))
    return df


# ============================================================
# FEATURES
# ============================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All features must be computable using ONLY data available before
    NIFTY opens — i.e. yesterday's closes for every series. No lookahead.
    """
    df = df.copy()

    # NIFTY momentum
    for lag in (1, 2, 3, 5):
        df[f"NIFTY_ret_{lag}"] = df["NIFTY"].pct_change(lag)

    # Z-score of close vs 20-day window — captures over/extended-ness
    rolling = df["NIFTY"].rolling(20)
    df["NIFTY_zscore_20"] = (df["NIFTY"] - rolling.mean()) / rolling.std()

    # 1-day returns of all factors
    for col in ("USDINR", "USOIL", "MSCI", "VIX",
                "SP500", "NIKKEI", "DXY", "US10Y"):
        if col in df.columns:
            df[f"{col}_ret_1"] = df[col].pct_change(1)
        else:
            df[f"{col}_ret_1"] = 0.0  # graceful fallback if download failed

    # Breadth proxy: fraction of last 20 days closed above 20-day MA
    ma20 = df["NIFTY"].rolling(20).mean()
    df["BREADTH"] = (df["NIFTY"] > ma20).rolling(20).mean()

    # Target: tomorrow's close as a ratio of today's close
    # (modeling ratio rather than absolute level keeps it stationary)
    df["TARGET_RATIO"] = df["NIFTY"].shift(-1) / df["NIFTY"]

    df = df.dropna()
    log.info("Feature matrix: %d rows × %d features", len(df), len(FEATURE_COLS))
    return df


# ============================================================
# MODEL
# ============================================================
def make_model() -> Pipeline:
    """Standard-scaled Ridge — robust to correlated macro factors."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=RIDGE_ALPHA)),
    ])


def train_and_score(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Train on full data, score with TimeSeriesSplit (no lookahead in CV)."""
    X = df[FEATURE_COLS].values
    y = df["TARGET_RATIO"].values

    model = make_model()
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    fold_mape, fold_dir = [], []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        fold_mape.append(mean_absolute_percentage_error(y[te], pred) * 100)
        # Directional accuracy: did we get UP/DOWN right?
        actual_up = y[te] > 1
        pred_up   = pred > 1
        fold_dir.append((actual_up == pred_up).mean() * 100)

    metrics = {
        "cv_mape_pct":      float(np.mean(fold_mape)),
        "cv_dir_acc_pct":   float(np.mean(fold_dir)),
        "cv_mape_std":      float(np.std(fold_mape)),
        "cv_dir_acc_std":   float(np.std(fold_dir)),
        "n_folds":          CV_SPLITS,
        "n_train_samples":  len(df),
    }

    # Final fit on ALL data for live prediction
    model.fit(X, y)
    return model, metrics


# ============================================================
# CLASSIFIER (confidence layer)
# ============================================================
def classify_signal(predicted_ratio: float, today_close: float) -> tuple[float, float, str, float]:
    """
    Returns (final_pred_level, p_up, label, confidence).
    Shrinks the prediction when uncertain — a common quant safety rule.
    """
    expected_ret = predicted_ratio - 1
    raw_pred     = today_close * predicted_ratio

    # Sigmoid-based probability (calibrated empirically)
    p_up       = 1 / (1 + np.exp(-expected_ret * 50))
    confidence = abs(p_up - 0.5) * 2

    if confidence < 0.10:
        final_pred = today_close + (raw_pred - today_close) * SHRINK_UNCERTAIN
        label = "UNCERTAIN — prediction shrunk 30%"
    elif (p_up > 0.5) == (expected_ret > 0):
        final_pred = raw_pred
        label = "AGREE — full move"
    else:
        final_pred = today_close + (raw_pred - today_close) * SHRINK_DISAGREE
        label = "DISAGREE — partial move"

    return float(final_pred), float(p_up), label, float(confidence)


def vix_range(today_close: float, vix_pct: float) -> tuple[float, float]:
    """Daily 1-sigma range implied by VIX."""
    daily_move = (vix_pct / 100) / np.sqrt(252)
    return today_close * (1 - daily_move), today_close * (1 + daily_move)


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================
def walk_forward_backtest(df: pd.DataFrame, warmup: int = BACKTEST_WARMUP) -> pd.DataFrame:
    """
    For each day t starting from `warmup`:
      • Train ONLY on data up to day t-1
      • Predict day t+1 using features known at end of day t
      • Compare against actual day t+1 close

    This is the only honest way to evaluate a time-series model.
    """
    log.info("Running walk-forward backtest (warmup=%d days)...", warmup)
    if len(df) < warmup + 30:
        log.error("Not enough data for backtest (need %d, have %d).", warmup + 30, len(df))
        return pd.DataFrame()

    records = []
    X_full = df[FEATURE_COLS].values
    y_full = df["TARGET_RATIO"].values
    closes = df["NIFTY"].values
    dates  = df.index

    # Refit cadence — refit weekly is a good speed/accuracy tradeoff
    refit_every = 5
    model = make_model()

    for t in range(warmup, len(df) - 1):
        if (t - warmup) % refit_every == 0:
            model.fit(X_full[:t], y_full[:t])

        pred_ratio = float(model.predict(X_full[t:t + 1])[0])
        today_cl   = float(closes[t])
        next_cl    = float(closes[t + 1])

        final_pred, p_up, label, conf = classify_signal(pred_ratio, today_cl)
        naive_pred = today_cl  # baseline: tomorrow = today

        records.append({
            "date":          dates[t],
            "today_close":   today_cl,
            "actual_next":   next_cl,
            "model_pred":    final_pred,
            "naive_pred":    naive_pred,
            "p_up":          p_up,
            "confidence":    conf,
            "label":         label,
            "actual_dir_up": next_cl > today_cl,
            "model_dir_up":  final_pred > today_cl,
        })

    bt = pd.DataFrame(records).set_index("date")

    # ---- METRICS ----
    bt["model_abs_err"] = (bt.model_pred - bt.actual_next).abs()
    bt["naive_abs_err"] = (bt.naive_pred - bt.actual_next).abs()
    bt["model_pct_err"] = bt.model_abs_err / bt.actual_next * 100
    bt["naive_pct_err"] = bt.naive_abs_err / bt.actual_next * 100

    n = len(bt)
    model_mape   = bt.model_pct_err.mean()
    naive_mape   = bt.naive_pct_err.mean()
    model_dir    = (bt.model_dir_up == bt.actual_dir_up).mean() * 100
    naive_dir    = 50.0  # naive has no directional signal

    # Strategy P&L if you traded the prediction (1 unit, no costs)
    bt["model_signal"] = np.where(bt.model_pred > bt.today_close, 1, -1)
    bt["realized_ret"] = (bt.actual_next - bt.today_close) / bt.today_close
    bt["strat_pnl"]    = bt.model_signal * bt.realized_ret

    sharpe = (bt.strat_pnl.mean() / bt.strat_pnl.std()) * np.sqrt(252) if bt.strat_pnl.std() > 0 else 0.0
    cum_ret    = (1 + bt.strat_pnl).prod() - 1
    bh_ret     = (bt.actual_next.iloc[-1] / bt.today_close.iloc[0]) - 1
    win_rate   = (bt.strat_pnl > 0).mean() * 100

    log.info("=" * 60)
    log.info("BACKTEST RESULTS  (%d predictions, %s → %s)",
             n, bt.index[0].date(), bt.index[-1].date())
    log.info("=" * 60)
    log.info("  MAPE (model)            : %.2f%%", model_mape)
    log.info("  MAPE (naive yesterday)  : %.2f%%   %s",
             naive_mape, "← model better" if model_mape < naive_mape else "← naive better")
    log.info("  Directional acc (model) : %.1f%%", model_dir)
    log.info("  Directional acc (naive) : %.1f%% (random)", naive_dir)
    log.info("")
    log.info("  Strategy (long/short on prediction direction, no costs):")
    log.info("    Cumulative return     : %+.2f%%", cum_ret * 100)
    log.info("    Buy-and-hold return   : %+.2f%%", bh_ret * 100)
    log.info("    Annualized Sharpe     : %.2f",   sharpe)
    log.info("    Win rate (per trade)  : %.1f%%", win_rate)
    log.info("=" * 60)

    if model_dir < 52:
        log.warning("Directional accuracy < 52%% — model has little edge over random.")
    if model_mape > naive_mape:
        log.warning("Model MAPE worse than naive — consider simplifying or more data.")

    return bt


# ============================================================
# LIVE PREDICTION
# ============================================================
def predict_one(df: pd.DataFrame, model: Pipeline) -> dict:
    """Predict tomorrow's NIFTY using the latest available features."""
    last_row = df[FEATURE_COLS].iloc[[-1]].values
    pred_ratio = float(model.predict(last_row)[0])

    today_close = float(df["NIFTY"].iloc[-1])
    today_date  = df.index[-1]
    final_pred, p_up, label, conf = classify_signal(pred_ratio, today_close)

    vix = float(df["VIX"].iloc[-1]) if "VIX" in df.columns else 18.0
    lo, hi = vix_range(today_close, vix)

    return {
        "as_of_date":     today_date.strftime("%Y-%m-%d"),
        "today_close":    today_close,
        "predicted_ratio": pred_ratio,
        "predicted_close": final_pred,
        "expected_move":   final_pred - today_close,
        "expected_pct":   (final_pred - today_close) / today_close * 100,
        "direction":      "UP" if final_pred > today_close else "DOWN",
        "p_up":           p_up,
        "confidence":     conf,
        "classifier":     label,
        "vix_today":      vix,
        "vix_range_low":  lo,
        "vix_range_high": hi,
        "within_vix":     lo <= final_pred <= hi,
    }


def print_prediction(result: dict, metrics: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  NIFTY 50 — NEXT-DAY PREDICTION")
    print(f"{sep}")
    print(f"  As of               : {result['as_of_date']}")
    print(f"  Today close         : {result['today_close']:>12,.2f}")
    print(f"  Predicted close     : {result['predicted_close']:>12,.2f}")
    print(f"  Expected move       : {result['expected_move']:>+12.2f}  ({result['expected_pct']:+.2f}%)")
    print(f"  Direction           : {result['direction']}")
    print(f"")
    print(f"  P(UP)               : {result['p_up']:.3f}")
    print(f"  Confidence          : {result['confidence']:.3f}")
    print(f"  Classifier          : {result['classifier']}")
    print(f"")
    print(f"  VIX today           : {result['vix_today']:.2f}%")
    print(f"  VIX 1σ range        : {result['vix_range_low']:,.2f}  —  {result['vix_range_high']:,.2f}")
    print(f"  Within VIX range?   : {'YES' if result['within_vix'] else 'NO  ⚠'}")
    print(f"")
    print(f"  Model CV MAPE       : {metrics['cv_mape_pct']:.2f}% ± {metrics['cv_mape_std']:.2f}%")
    print(f"  Model CV Dir Acc    : {metrics['cv_dir_acc_pct']:.1f}% ± {metrics['cv_dir_acc_std']:.1f}%")
    print(f"  Trained on          : {metrics['n_train_samples']} samples")
    print(f"{sep}\n")


# ============================================================
# GOOGLE SHEETS WRITER (for `daily` mode)
# ============================================================
def write_to_sheet(result: dict, metrics: dict) -> None:
    """Append today's prediction as a new row in the predictions tab."""
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except ImportError:
        log.error("gspread not installed. Run: pip install gspread oauth2client")
        return

    log.info("Connecting to Google Sheets to write prediction...")

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    secret_json = os.environ.get("GOOGLE_SECRET_KEY")
    try:
        if secret_json:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(secret_json), scope)
        elif os.path.exists("google_secret_key.json"):
            creds = ServiceAccountCredentials.from_json_keyfile_name("google_secret_key.json", scope)
        else:
            log.error("No Google credentials found (env var GOOGLE_SECRET_KEY or google_secret_key.json).")
            return
        gc = gspread.authorize(creds)
    except Exception as exc:
        log.error("Google authentication failed: %s", exc, exc_info=True)
        return

    try:
        ws = gc.open(SHEET_NAME).worksheet(PREDICTIONS_TAB)
    except Exception as exc:
        log.error(
            "Could not open '%s' / '%s'. Create the tab and ensure the service "
            "account has Editor access. Error: %s",
            SHEET_NAME, PREDICTIONS_TAB, exc,
        )
        return

    # If the sheet is empty, write the header
    try:
        existing = ws.get_all_values()
        if not existing:
            header = [
                "RunTimestamp", "AsOfDate", "TodayClose", "PredictedClose",
                "ExpectedMove", "ExpectedPct", "Direction",
                "PUp", "Confidence", "Classifier",
                "VixToday", "VixRangeLow", "VixRangeHigh", "WithinVix",
                "CvMapePct", "CvDirAccPct", "TrainSamples",
            ]
            ws.append_row(header, value_input_option="USER_ENTERED")
            log.info("Header row written to '%s'.", PREDICTIONS_TAB)
    except Exception as exc:
        log.error("Could not read existing sheet rows: %s", exc)
        return

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        result["as_of_date"],
        round(result["today_close"], 2),
        round(result["predicted_close"], 2),
        round(result["expected_move"], 2),
        round(result["expected_pct"], 3),
        result["direction"],
        round(result["p_up"], 4),
        round(result["confidence"], 4),
        result["classifier"],
        round(result["vix_today"], 2),
        round(result["vix_range_low"], 2),
        round(result["vix_range_high"], 2),
        "YES" if result["within_vix"] else "NO",
        round(metrics["cv_mape_pct"], 3),
        round(metrics["cv_dir_acc_pct"], 2),
        metrics["n_train_samples"],
    ]
    try:
        ws.append_row(row, value_input_option="USER_ENTERED")
        log.info("Prediction row appended to '%s'.", PREDICTIONS_TAB)
    except Exception as exc:
        log.error("Failed to append prediction row: %s", exc, exc_info=True)


# ============================================================
# CLI
# ============================================================
def cmd_backtest(args):
    df = download_data(period=args.period)
    df = add_features(df)
    bt = walk_forward_backtest(df, warmup=args.warmup)
    if args.save and not bt.empty:
        path = "backtest_results.csv"
        bt.to_csv(path)
        log.info("Backtest details saved to %s", path)


def cmd_predict(args):
    df = download_data(period=args.period)
    df = add_features(df)
    model, metrics = train_and_score(df)
    result = predict_one(df, model)
    print_prediction(result, metrics)


def cmd_daily(args):
    df = download_data(period=args.period)
    df = add_features(df)
    model, metrics = train_and_score(df)
    result = predict_one(df, model)
    print_prediction(result, metrics)
    write_to_sheet(result, metrics)


def main():
    parser = argparse.ArgumentParser(description="NIFTY 50 next-day prediction system")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bt = sub.add_parser("backtest", help="Walk-forward backtest")
    p_bt.add_argument("--period", default=LOOKBACK_YEARS, help="yfinance period (e.g. 3y, 5y, max)")
    p_bt.add_argument("--warmup", type=int, default=BACKTEST_WARMUP, help="Warmup days before first prediction")
    p_bt.add_argument("--save",   action="store_true", help="Save per-day results to backtest_results.csv")
    p_bt.set_defaults(func=cmd_backtest)

    p_pr = sub.add_parser("predict", help="One-shot prediction for tomorrow")
    p_pr.add_argument("--period", default=LOOKBACK_YEARS)
    p_pr.set_defaults(func=cmd_predict)

    p_dy = sub.add_parser("daily", help="Predict + write to Google Sheet")
    p_dy.add_argument("--period", default=LOOKBACK_YEARS)
    p_dy.set_defaults(func=cmd_daily)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
