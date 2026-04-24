import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import io
import os
import sys
import json
import time
import logging
import urllib3
from collections import defaultdict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("EODJournal")

# ==========================================
# CONFIGURATION
# ==========================================
IMGBB_API_KEY = os.environ.get("IMGBB_API_KEY", "bc9bd3a639aec812b3b705bfcd7810ce")

SHEET_NAME = "Abhay"
TAB_NAME   = "mStock working"
KEYS_TAB   = "KeyAndRules"

INDEX_MAP = {
    "NIFTY":     "1/26000",
    "BANKNIFTY": "1/26009",
    "SENSEX":    "4/51",
    "BANKEX":    "4/69",
}

# NFO = NSE F&O (NIFTY, BANKNIFTY options); BFO = BSE F&O (SENSEX, BANKEX options).
EXCHANGE_SEGMENT_MAP = {
    "NSE": "1", "NFO": "2", "CDS": "3",
    "BSE": "4", "BFO": "5", "BCD": "6", "MCX": "7",
}

# Retry settings for transient HTTP failures (429 / 5xx)
MAX_RETRIES    = 3
RETRY_BACKOFF  = 2   # seconds — doubles each attempt


# ==========================================
# HELPERS
# ==========================================
def _http_get(url, *, headers, timeout=60, label="API") -> requests.Response | None:
    """
    GET with retry-on-transient-error. Returns Response or None.
    Logs the full error context on every failure so nothing is silent.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, verify=False, timeout=timeout)
        except requests.exceptions.Timeout:
            log.error("[%s] Timeout after %ds (attempt %d/%d)", label, timeout, attempt, MAX_RETRIES)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            return None
        except requests.exceptions.ConnectionError as exc:
            log.error("[%s] Connection error (attempt %d/%d): %s", label, attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            return None
        except requests.exceptions.RequestException as exc:
            log.error("[%s] Request failed (attempt %d/%d): %s", label, attempt, MAX_RETRIES, exc)
            return None

        # Retry on server errors and rate-limit; fail fast on auth/client errors
        if resp.status_code in (429, 500, 502, 503, 504):
            log.warning("[%s] HTTP %d — retrying (attempt %d/%d)", label, resp.status_code, attempt, MAX_RETRIES)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue

        return resp

    return None


def _cleanup(filename: str):
    """Delete a local PNG file, logging any OS error."""
    try:
        if filename and os.path.exists(filename):
            os.remove(filename)
            log.debug("Deleted local file: %s", filename)
    except OSError as exc:
        log.warning("Could not delete local file %s: %s", filename, exc)


def safe_get(row, i):
    return row[i] if 0 <= i < len(row) else ''


# ==========================================
# 1. AUTHENTICATE GOOGLE SHEETS
# ==========================================
def authenticate_google() -> gspread.Client:
    log.info("Connecting to Google Workspace...")
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        secret_json = os.environ.get("GOOGLE_SECRET_KEY")
        if secret_json:
            log.debug("Loading Google credentials from environment variable.")
            creds_dict = json.loads(secret_json)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        else:
            key_file = "google_secret_key.json"
            if not os.path.exists(key_file):
                log.critical(
                    "google_secret_key.json not found and GOOGLE_SECRET_KEY env var is not set. "
                    "Cannot authenticate with Google."
                )
                raise FileNotFoundError(key_file)
            log.debug("Loading Google credentials from %s", key_file)
            creds = ServiceAccountCredentials.from_json_keyfile_name(key_file, scope)

        gc = gspread.authorize(creds)
        log.info("Google Workspace authenticated successfully.")
        return gc

    except (json.JSONDecodeError, ValueError) as exc:
        log.critical("GOOGLE_SECRET_KEY env var contains invalid JSON: %s", exc)
        raise
    except Exception as exc:
        log.critical("Google authentication failed: %s", exc, exc_info=True)
        raise


# ==========================================
# 2. STARTUP VALIDATION
# ==========================================
def validate_config():
    """Fail fast before making any API calls if the environment is misconfigured."""
    errors = []

    if not IMGBB_API_KEY:
        errors.append(
            "IMGBB_API_KEY is empty. Set the env var or hardcode a fallback. "
            "Uploads WILL silently return 400 from imgbb without it."
        )

    if errors:
        for msg in errors:
            log.critical("CONFIG ERROR: %s", msg)
        sys.exit(1)

    log.info("Configuration validated OK.")


# ==========================================
# 3. DOWNLOAD INSTRUMENT MASTER
# ==========================================
def load_instrument_master(api_key: str, jwt_token: str) -> pd.DataFrame | None:
    log.info("Downloading mStock Security Master (this may take ~10–20 seconds)...")

    if not api_key or not jwt_token:
        log.error(
            "mStock API key or JWT token is blank. "
            "Check cells A2/B2 in the '%s' tab.", KEYS_TAB
        )
        return None

    url     = "https://api.mstock.trade/openapi/typea/instruments/scriptmaster"
    headers = {"X-Mirae-Version": "1", "Authorization": f"token {api_key}:{jwt_token}"}

    resp = _http_get(url, headers=headers, timeout=60, label="MasterFile")
    if resp is None:
        return None

    if resp.status_code != 200:
        try:
            err = resp.json()
            log.error(
                "Master file download failed [HTTP %d]: error_type=%s  message=%s",
                resp.status_code, err.get("error_type"), err.get("message"),
            )
        except Exception:
            log.error(
                "Master file download failed [HTTP %d]. Raw response (first 500 chars): %s",
                resp.status_code, resp.text[:500],
            )
        return None

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        log.error("Failed to parse Security Master CSV: %s", exc)
        return None

    required_cols = {"name", "strike", "instrument_type", "expiry", "instrument_token", "exchange"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error(
            "Security Master is missing expected columns: %s. "
            "Got: %s", missing, list(df.columns)
        )
        return None

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    log.info("Security Master loaded: %d instruments.", len(df))
    return df


# ==========================================
# 4. RESOLVE SYMBOL -> TOKEN + SEGMENT
# ==========================================
def get_token_from_string(df: pd.DataFrame, custom_symbol: str):
    """
    Returns (token, segment) or (None, None).
    Logs the exact reason for every lookup failure.
    """
    try:
        parts = custom_symbol.split("-")
        if len(parts) < 4:
            log.error(
                "Symbol '%s' is malformed — expected NAME-EXPIRY-STRIKE-OPTTYPE, got %d parts.",
                custom_symbol, len(parts),
            )
            return None, None

        name        = parts[0]
        opt_type    = parts[3].upper()
        strike      = float(parts[2])
        expiry_date = pd.to_datetime(parts[1]).strftime("%Y-%m-%d")

    except (ValueError, IndexError) as exc:
        log.error("Could not parse symbol '%s': %s", custom_symbol, exc)
        return None, None

    filtered = df[
        (df["name"]            == name) &
        (df["strike"]          == strike) &
        (df["instrument_type"] == opt_type) &
        (df["expiry"]          == expiry_date)
    ]

    if filtered.empty:
        log.warning(
            "No instrument found for symbol '%s' "
            "(searched: name=%s, strike=%s, opt_type=%s, expiry=%s). "
            "Verify the expiry date matches the master file format.",
            custom_symbol, name, strike, opt_type, expiry_date,
        )
        return None, None

    row      = filtered.iloc[0]
    token    = row["instrument_token"]
    exchange = str(row.get("exchange", "")).upper()
    segment  = EXCHANGE_SEGMENT_MAP.get(exchange, "2")

    if exchange not in EXCHANGE_SEGMENT_MAP:
        log.warning(
            "Unknown exchange '%s' for symbol '%s'. Defaulting to segment '2' (NFO). "
            "Add it to EXCHANGE_SEGMENT_MAP if this is wrong.",
            exchange, custom_symbol,
        )

    log.debug("Resolved '%s' → token=%s, segment=%s, exchange=%s", custom_symbol, token, segment, exchange)
    return token, segment


# ==========================================
# 5. FETCH DATA & GENERATE CHART
# ==========================================
def generate_chart(
    endpoint_path: str,
    display_name: str,
    api_key: str,
    jwt_token: str,
    timeframe: int = 1,
    trade_info: dict | None = None,
) -> str | None:

    tf_label = f"{timeframe}min"
    log.info("Generating %s chart for %s...", tf_label, display_name)

    url     = f"https://api.mstock.trade/openapi/typea/instruments/intraday/{endpoint_path}/minute"
    headers = {"X-Mirae-Version": "1", "Authorization": f"token {api_key}:{jwt_token}"}

    resp = _http_get(url, headers=headers, timeout=60, label=f"Chart/{display_name}")
    if resp is None:
        return None

    if resp.status_code != 200:
        try:
            err = resp.json()
            log.error(
                "Chart data fetch failed for '%s' [HTTP %d]: error_type=%s  message=%s",
                display_name, resp.status_code, err.get("error_type"), err.get("message"),
            )
        except Exception:
            log.error(
                "Chart data fetch failed for '%s' [HTTP %d]. Raw response: %s",
                display_name, resp.status_code, resp.text[:500],
            )
        return None

    try:
        data = resp.json()
    except Exception as exc:
        log.error(
            "Could not parse JSON response for '%s'. "
            "Raw response (first 500 chars): %s | Error: %s",
            display_name, resp.text[:500], exc,
        )
        return None

    if data.get("status") != "success":
        log.error(
            "mStock API rejected chart request for '%s': %s",
            display_name, data.get("message", "No message returned"),
        )
        return None

    if "candles" not in data.get("data", {}):
        log.warning("No 'candles' key in API response for '%s'. Full data keys: %s", display_name, list(data.get("data", {}).keys()))
        return None

    raw_candles = data["data"]["candles"]
    if not raw_candles:
        log.warning("Candle list is empty for '%s' — market may have been closed.", display_name)
        return None

    raw_candles.reverse()

    try:
        df = pd.DataFrame(raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = df["timestamp"].astype(str).str.split("+").str[0]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.between_time("09:15", "15:30")

        # Snapshot 1-min data BEFORE resampling for accurate MFE/MAE calculation.
        df_1min = df.reset_index().copy()

        if timeframe == 5:
            df = df.resample("5min").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()

        df.reset_index(inplace=True)

    except Exception as exc:
        log.error("Failed to build DataFrame for '%s': %s", display_name, exc, exc_info=True)
        return None

    if df.empty:
        log.warning("No market-hours data (09:15–15:30) for '%s'.", display_name)
        return None

    # ---- PLOT ----
    filename = f"{display_name}_{tf_label}_chart.png"
    try:
        green_color, red_color, wick_width = "#26a69a", "#ef5350", 1.0

        fig, ax = plt.subplots(figsize=(18, 8), facecolor="white")
        ax.set_facecolor("white")

        df["date_num"] = mdates.date2num(df["timestamp"])
        width = 0.0005 if timeframe == 1 else 0.0025

        up   = df[df.close >= df.open]
        down = df[df.close <  df.open]

        ax.vlines(up.date_num,   up.low,   up.high,   color=green_color, linewidth=wick_width)
        ax.vlines(down.date_num, down.low, down.high, color=red_color,   linewidth=wick_width)
        ax.bar(up.date_num,   up.close   - up.open,   width, bottom=up.open,   color=green_color, edgecolor=green_color)
        ax.bar(down.date_num, down.close - down.open, width, bottom=down.open, color=red_color,   edgecolor=red_color)

        # ---- TRADE OVERLAY ----
        if trade_info:
            _draw_trade_overlay(ax, df, df_1min, display_name, trade_info)

        # ---- AXES STYLING ----
        symbol_name = display_name.split("-")[0].upper()
        if symbol_name in {"BANKNIFTY", "SENSEX", "BANKEX",
                           "BANKNIFTY INDEX", "SENSEX INDEX", "BANKEX INDEX"}:
            interval = 20
        else:
            interval = 10

        ax.yaxis.set_major_locator(ticker.MultipleLocator(interval))
        ax.yaxis.tick_right()

        label_text = "Spot Index Price (₹)" if "INDEX" in display_name else "Premium Price (₹)"
        ax.set_ylabel(label_text, color="black", fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", colors="black", labelsize=10)
        ax.grid(True, linestyle="--", color="#e0e0e0", alpha=0.7)
        ax.set_title(f"{display_name} ({tf_label} Chart)", color="black", fontsize=18, fontweight="bold")

        tick_interval = 5 if timeframe == 1 else 15
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, tick_interval)))
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", rotation=90, colors="black", labelsize=10)

        xmin, xmax = df["date_num"].min(), df["date_num"].max()
        xpad = (xmax - xmin) * 0.003
        ax.set_xlim(xmin - xpad, xmax + xpad)

        plt.tight_layout()
        plt.savefig(filename, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.debug("Chart saved: %s", filename)
        return filename

    except Exception as exc:
        log.error("Failed to render chart for '%s': %s", display_name, exc, exc_info=True)
        plt.close("all")
        _cleanup(filename)
        return None


def _draw_trade_overlay(ax, df, df_1min, display_name, trade_info):
    """Draw entry/exit markers, shaded band, price tags, and P&L badge."""
    try:
        buy_time   = pd.to_datetime(trade_info["buy_time"])
        sell_time  = pd.to_datetime(trade_info["sell_time"])
        buy_price  = float(trade_info["buy_price"])
        sell_price = float(trade_info["sell_price"])
        pnl        = float(trade_info["pnl"])
        direction  = str(trade_info.get("direction", "BUY")).upper()
        pct        = trade_info.get("percentage", "")

        buy_num  = mdates.date2num(buy_time)
        sell_num = mdates.date2num(sell_time)
        has_prices = not (np.isnan(buy_price) or np.isnan(sell_price))

        start_num,  end_num  = (buy_num,  sell_num)  if buy_num  <= sell_num  else (sell_num,  buy_num)
        start_time, end_time = (buy_time, sell_time) if buy_time <= sell_time else (sell_time, buy_time)

        band_color = "#c8e6c9" if pnl >= 0 else "#ffcdd2"
        ax.axvspan(start_num, end_num, alpha=0.30, color=band_color, zorder=0)

        buy_green = "#1b5e20"
        sell_red  = "#b71c1c"

        ax.axvline(buy_num,  color=buy_green, linestyle="--", linewidth=1.3, alpha=0.85, zorder=3)
        ax.axvline(sell_num, color=sell_red,  linestyle="--", linewidth=1.3, alpha=0.85, zorder=3)

        mfe_mae_line = ""

        if has_prices:
            ax.axhline(buy_price,  color=buy_green, linestyle="--", linewidth=1.3, alpha=0.75, zorder=2)
            ax.axhline(sell_price, color=sell_red,  linestyle="--", linewidth=1.3, alpha=0.75, zorder=2)

            y_tf = ax.get_yaxis_transform()
            ax.annotate(f" BUY ₹{buy_price:g} ",
                        xy=(1.0, buy_price), xycoords=y_tf, xytext=(-4, 0),
                        textcoords="offset points", fontsize=9, fontweight="bold",
                        color="white", ha="right", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=buy_green, edgecolor="none", alpha=0.95),
                        zorder=6)
            ax.annotate(f" SELL ₹{sell_price:g} ",
                        xy=(1.0, sell_price), xycoords=y_tf, xytext=(-4, 0),
                        textcoords="offset points", fontsize=9, fontweight="bold",
                        color="white", ha="right", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=sell_red, edgecolor="none", alpha=0.95),
                        zorder=6)

            triangle_y = df["low"].min()
            ax.scatter(buy_num,  triangle_y, marker="^", s=320, color=buy_green, edgecolor="white", linewidth=1.4, zorder=5, clip_on=False)
            ax.scatter(sell_num, triangle_y, marker="v", s=320, color=sell_red,  edgecolor="white", linewidth=1.4, zorder=5, clip_on=False)

            is_long      = (direction == "BUY")
            entry_price  = buy_price if is_long else sell_price
            R            = abs(buy_price - sell_price)
            in_trade     = df_1min[(df_1min["timestamp"] >= start_time) & (df_1min["timestamp"] <= end_time)]

            if in_trade.empty:
                mfe_during = mae_during = 0.0
                log.warning("No 1-min candles found within trade window for '%s' — MFE/MAE set to 0.", display_name)
            else:
                hi, lo = in_trade["high"].max(), in_trade["low"].min()
                if is_long:
                    mfe_during = max(hi - entry_price, 0.0)
                    mae_during = max(entry_price - lo, 0.0)
                else:
                    mfe_during = max(entry_price - lo, 0.0)
                    mae_during = max(hi - entry_price, 0.0)

            def _r(amount):
                return f" ({amount / R:.2f}R)" if R > 0 else ""

            mfe_mae_line = (
                f"During trade: MFE +₹{mfe_during:.2f}{_r(mfe_during)}  |  "
                f"MAE -₹{mae_during:.2f}{_r(mae_during)}"
            )

        else:
            # Index chart: show spot level at entry/exit times instead of option premium.
            try:
                buy_idx      = (df["timestamp"] - buy_time ).abs().idxmin()
                sell_idx     = (df["timestamp"] - sell_time).abs().idxmin()
                spot_at_buy  = float(df.loc[buy_idx,  "close"])
                spot_at_sell = float(df.loc[sell_idx, "close"])

                ax.axhline(spot_at_buy,  color=buy_green, linestyle="--", linewidth=1.3, alpha=0.75, zorder=2)
                ax.axhline(spot_at_sell, color=sell_red,  linestyle="--", linewidth=1.3, alpha=0.75, zorder=2)

                y_tf = ax.get_yaxis_transform()
                ax.annotate(f" BUY {spot_at_buy:.2f} ",
                            xy=(1.0, spot_at_buy), xycoords=y_tf, xytext=(-4, 0),
                            textcoords="offset points", fontsize=9, fontweight="bold",
                            color="white", ha="right", va="center",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=buy_green, edgecolor="none", alpha=0.95),
                            zorder=6)
                ax.annotate(f" SELL {spot_at_sell:.2f} ",
                            xy=(1.0, spot_at_sell), xycoords=y_tf, xytext=(-4, 0),
                            textcoords="offset points", fontsize=9, fontweight="bold",
                            color="white", ha="right", va="center",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=sell_red, edgecolor="none", alpha=0.95),
                            zorder=6)

                triangle_y = df["low"].min()
                ax.scatter(buy_num,  triangle_y, marker="^", s=320, color=buy_green, edgecolor="white", linewidth=1.4, zorder=5, clip_on=False)
                ax.scatter(sell_num, triangle_y, marker="v", s=320, color=sell_red,  edgecolor="white", linewidth=1.4, zorder=5, clip_on=False)

            except Exception as exc:
                log.warning("Could not draw index reference lines for '%s': %s", display_name, exc)

        # ---- P&L badge ----
        badge_color = "#2e7d32" if pnl >= 0 else "#c62828"
        dur_str     = str(end_time - start_time).split(".")[0]
        pnl_sign    = "+" if pnl >= 0 else ""
        trade_type  = "OPTION BUY" if direction == "BUY" else "OPTION SELL"

        line1 = f"{trade_type}  |  P&L: {pnl_sign}₹{pnl:.2f}"
        if pct:
            line1 += f"  ({pct})"
        line1 += f"  |  Duration: {dur_str}"

        total_trades = int(trade_info.get("total_trades") or 1)
        trade_number = int(trade_info.get("trade_number") or 1)
        day_pnl      = trade_info.get("day_pnl")
        badge_lines  = [line1]

        if total_trades > 1 and day_pnl is not None:
            try:
                day_pnl_f = float(day_pnl)
                day_sign  = "+" if day_pnl_f >= 0 else ""
                wins      = int(trade_info.get("day_wins")   or 0)
                losses    = int(trade_info.get("day_losses") or 0)
                badge_lines.append(
                    f"Trade {trade_number} of {total_trades} today  |  "
                    f"Wins: {wins}  Losses: {losses}  |  "
                    f"Day Net: {day_sign}₹{day_pnl_f:.2f}"
                )
            except Exception as exc:
                log.warning("Could not build day-context badge line: %s", exc)

        if has_prices and mfe_mae_line:
            badge_lines.append(mfe_mae_line)

        ax.text(0.01, 0.98, "\n".join(badge_lines), transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="white",
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.6", facecolor=badge_color, edgecolor="none", alpha=0.92),
                zorder=7)

    except Exception as exc:
        log.error("Trade overlay failed for '%s': %s", display_name, exc, exc_info=True)


# ==========================================
# 6. UPLOAD TO IMGBB
# ==========================================
def upload_image(filename: str) -> str | None:
    log.info("Uploading %s to imgbb...", filename)

    if not IMGBB_API_KEY:
        log.error(
            "IMGBB_API_KEY is empty — upload aborted for '%s'. "
            "imgbb will return HTTP 400 with an empty key.", filename
        )
        return None

    if not os.path.exists(filename):
        log.error("Upload aborted: file not found on disk: %s", filename)
        return None

    file_size_kb = os.path.getsize(filename) / 1024
    log.debug("File size: %.1f KB", file_size_kb)

    url = "https://api.imgbb.com/1/upload"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(filename, "rb") as fh:
                resp = requests.post(
                    url,
                    data={"key": IMGBB_API_KEY},
                    files={"image": fh},
                    verify=False,
                    timeout=60,
                )
        except requests.exceptions.Timeout:
            log.error("imgbb upload timeout (attempt %d/%d) for '%s'", attempt, MAX_RETRIES, filename)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            return None
        except requests.exceptions.RequestException as exc:
            log.error("imgbb upload connection error for '%s': %s", filename, exc)
            return None

        if resp.status_code == 400:
            try:
                err = resp.json()
                log.error(
                    "imgbb rejected upload (HTTP 400) for '%s'. "
                    "status_code=%s  error=%s  — "
                    "Most likely cause: IMGBB_API_KEY is wrong or expired.",
                    filename, err.get("status_code"), err.get("error"),
                )
            except Exception:
                log.error(
                    "imgbb rejected upload (HTTP 400) for '%s'. Raw response: %s",
                    filename, resp.text[:300],
                )
            return None

        if resp.status_code in (429, 500, 502, 503, 504):
            log.warning("imgbb HTTP %d (attempt %d/%d) for '%s'", resp.status_code, attempt, MAX_RETRIES, filename)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            return None

        try:
            data = resp.json()
        except Exception as exc:
            log.error(
                "imgbb returned non-JSON response for '%s' [HTTP %d]. "
                "Raw (first 300 chars): %s | Parse error: %s",
                filename, resp.status_code, resp.text[:300], exc,
            )
            return None

        if data.get("success"):
            url_result = data["data"]["url"]
            log.info("Upload successful: %s", url_result)
            return url_result

        log.error(
            "imgbb upload failed for '%s': success=False  "
            "status=%s  error=%s  full_response=%s",
            filename, data.get("status"), data.get("error"), data,
        )
        return None

    return None


# ==========================================
# 7. WRITE HYPERLINK BACK TO SHEET
# ==========================================
def inject_sheet_link(sheet, row_number: int, col: int, url: str, label: str):
    """Update a single cell with a HYPERLINK formula, with error handling."""
    formula = f'=HYPERLINK("{url}", "{label}")'
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            sheet.update_cell(row_number, col, formula)
            log.info("Sheet updated — row %d col %d: %s", row_number, col, label)
            return True
        except gspread.exceptions.APIError as exc:
            log.warning(
                "Google Sheets API error on row %d col %d (attempt %d/%d): %s",
                row_number, col, attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
        except Exception as exc:
            log.error(
                "Unexpected error writing to sheet row %d col %d: %s",
                row_number, col, exc, exc_info=True,
            )
            return False
    log.error("Gave up writing to sheet row %d col %d after %d attempts.", row_number, col, MAX_RETRIES)
    return False


# ==========================================
# 8. MAIN EXECUTION
# ==========================================
def run_eod_journal():
    log.info("=" * 60)
    log.info("EOD Journal Processor — starting")
    log.info("=" * 60)

    validate_config()

    try:
        gc = authenticate_google()
    except Exception:
        log.critical("Cannot proceed without Google authentication. Exiting.")
        sys.exit(1)

    # --- Fetch mStock credentials from sheet ---
    try:
        log.info("Fetching mStock credentials from '%s' sheet", KEYS_TAB)
        settings_sheet = gc.open(SHEET_NAME).worksheet(KEYS_TAB)
        api_key   = settings_sheet.acell("A2").value
        jwt_token = settings_sheet.acell("B2").value
    except gspread.exceptions.SpreadsheetNotFound:
        log.critical("Spreadsheet '%s' not found. Check SHEET_NAME and sharing permissions.", SHEET_NAME)
        sys.exit(1)
    except gspread.exceptions.WorksheetNotFound:
        log.critical("Tab '%s' not found inside spreadsheet '%s'.", KEYS_TAB, SHEET_NAME)
        sys.exit(1)
    except Exception as exc:
        log.critical("Could not read mStock credentials: %s", exc, exc_info=True)
        sys.exit(1)

    if not api_key or not jwt_token:
        log.critical(
            "API key or JWT token is blank in '%s'!A2:B2. "
            "Update the sheet and re-run.", KEYS_TAB
        )
        sys.exit(1)

    log.info("mStock credentials loaded.")

    # --- Load sheet data ---
    try:
        sheet    = gc.open(SHEET_NAME).worksheet(TAB_NAME)
        raw_data = sheet.get_all_values()
    except gspread.exceptions.WorksheetNotFound:
        log.critical("Tab '%s' not found in spreadsheet '%s'.", TAB_NAME, SHEET_NAME)
        sys.exit(1)
    except Exception as exc:
        log.critical("Could not read sheet '%s': %s", TAB_NAME, exc, exc_info=True)
        sys.exit(1)

    # --- Find header row ---
    header_row_index = next(
        (i for i, row in enumerate(raw_data) if "IndexTraded" in row), -1
    )
    if header_row_index == -1:
        log.critical("Could not find 'IndexTraded' column header in the sheet. Aborting.")
        sys.exit(1)

    log.info("Header row found at sheet row %d.", header_row_index + 1)
    headers = raw_data[header_row_index]

    REQUIRED_HEADERS = {
        "TradeStatus", "Chart", "IndexTraded",
        "BuyTime", "SellTime", "BuyPrice", "SellPrice", "Pnl", "Direction",
    }
    try:
        status_idx     = headers.index("TradeStatus")
        chart_idx      = headers.index("Chart")
        symbol_idx     = headers.index("IndexTraded")
        buy_time_idx   = headers.index("BuyTime")
        sell_time_idx  = headers.index("SellTime")
        buy_price_idx  = headers.index("BuyPrice")
        sell_price_idx = headers.index("SellPrice")
        pnl_idx        = headers.index("Pnl")
        direction_idx  = headers.index("Direction")
    except ValueError as exc:
        log.critical("Missing required column header: %s. Found headers: %s", exc, headers)
        sys.exit(1)

    pct_idx  = headers.index("Percentage%") if "Percentage%" in headers else -1
    date_idx = headers.index("Date")        if "Date"        in headers else -1

    if pct_idx  == -1: log.warning("Optional column 'Percentage%%' not found — percentage badge will be skipped.")
    if date_idx == -1: log.warning("Optional column 'Date' not found — will derive date from BuyTime.")

    # --- Build day-context (multi-trade awareness) ---
    data_rows    = raw_data[header_row_index + 1:]
    trade_groups = defaultdict(list)

    for ridx, row in enumerate(data_rows):
        if safe_get(row, status_idx) != "CLOSED":
            continue
        sym = safe_get(row, symbol_idx)
        if not sym:
            continue

        try:
            bt_val = pd.to_datetime(safe_get(row, buy_time_idx))
        except Exception:
            bt_val = pd.Timestamp.min

        date_str = safe_get(row, date_idx) if date_idx >= 0 else ""
        if not date_str:
            try:
                date_str = str(bt_val.date())
            except Exception:
                date_str = "unknown"

        try:
            pnl_val = float(safe_get(row, pnl_idx))
        except Exception:
            pnl_val = 0.0

        trade_groups[(date_str, sym)].append((ridx, bt_val, pnl_val))

    row_context = {}
    for (date_str, sym), trades in trade_groups.items():
        trades_sorted = sorted(trades, key=lambda t: t[1])
        total  = len(trades_sorted)
        day_pnl = sum(t[2] for t in trades_sorted)
        wins    = sum(1 for t in trades_sorted if t[2] > 0)
        losses  = sum(1 for t in trades_sorted if t[2] < 0)
        for rank, (ridx, _, _) in enumerate(trades_sorted, start=1):
            row_context[ridx] = {
                "trade_number": rank, "total_trades": total,
                "day_pnl": day_pnl, "day_wins": wins, "day_losses": losses,
            }

    # --- Process each eligible row ---
    master_df = None
    stats = {"processed": 0, "skipped": 0, "charts_ok": 0, "charts_fail": 0, "uploads_fail": 0, "sheet_fail": 0}

    for idx, row in enumerate(data_rows):
        row_number    = header_row_index + 2 + idx
        status        = safe_get(row, status_idx)
        chart_cell    = safe_get(row, chart_idx)
        custom_symbol = safe_get(row, symbol_idx)

        if not (status == "CLOSED" and chart_cell == "" and custom_symbol):
            stats["skipped"] += 1
            continue

        log.info("-" * 50)
        log.info("Processing row %d: %s", row_number, custom_symbol)
        stats["processed"] += 1

        if master_df is None:
            master_df = load_instrument_master(api_key, jwt_token)
            if master_df is None:
                log.critical("Cannot download Security Master — aborting all remaining rows.")
                break

        token, segment = get_token_from_string(master_df, custom_symbol)
        if not token:
            log.error("Skipping row %d — token lookup failed for '%s'.", row_number, custom_symbol)
            stats["charts_fail"] += 1
            continue

        endpoint_path = f"{segment}/{token}"

        # Build trade_info overlay
        trade_info = None
        try:
            raw_ti = {
                "buy_time":   safe_get(row, buy_time_idx),
                "sell_time":  safe_get(row, sell_time_idx),
                "buy_price":  safe_get(row, buy_price_idx),
                "sell_price": safe_get(row, sell_price_idx),
                "pnl":        safe_get(row, pnl_idx),
                "direction":  safe_get(row, direction_idx),
                "percentage": safe_get(row, pct_idx) if pct_idx >= 0 else "",
            }
            if all([raw_ti["buy_time"], raw_ti["sell_time"],
                    raw_ti["buy_price"], raw_ti["sell_price"]]):
                ctx = row_context.get(idx, {})
                raw_ti.update({
                    "trade_number": ctx.get("trade_number", 1),
                    "total_trades": ctx.get("total_trades", 1),
                    "day_pnl":      ctx.get("day_pnl",      0.0),
                    "day_wins":     ctx.get("day_wins",      0),
                    "day_losses":   ctx.get("day_losses",    0),
                })
                trade_info = raw_ti
            else:
                missing = [k for k in ("buy_time", "sell_time", "buy_price", "sell_price") if not raw_ti[k]]
                log.warning("Trade overlay disabled for row %d — missing fields: %s", row_number, missing)
        except Exception as exc:
            log.warning("Could not build trade_info for row %d: %s", row_number, exc)

        def _process_chart(ep, name, tf, t_info, col_offset, label):
            chart_file = generate_chart(ep, name, api_key, jwt_token, timeframe=tf, trade_info=t_info)
            if not chart_file:
                stats["charts_fail"] += 1
                return
            stats["charts_ok"] += 1
            url = upload_image(chart_file)
            _cleanup(chart_file)
            if not url:
                stats["uploads_fail"] += 1
                log.error("Upload failed for %s (%s) — chart link NOT written to sheet.", name, label)
                return
            ok = inject_sheet_link(sheet, row_number, chart_idx + col_offset, url, f"{name}-{tf}min.png")
            if not ok:
                stats["sheet_fail"] += 1

        # 1. Option 1-min chart
        _process_chart(endpoint_path, custom_symbol, 1, trade_info, 1, "1min")

        # 2. Option 5-min chart
        _process_chart(endpoint_path, custom_symbol, 5, trade_info, 2, "5min")

        # 3. Spot index 5-min chart
        symbol_name = custom_symbol.split("-")[0].upper()
        if symbol_name in INDEX_MAP:
            index_endpoint    = INDEX_MAP[symbol_name]
            index_display     = f"{symbol_name} INDEX"
            index_trade_info  = None
            if trade_info:
                index_trade_info = dict(trade_info)
                index_trade_info["buy_price"]  = float("nan")
                index_trade_info["sell_price"] = float("nan")
            _process_chart(index_endpoint, index_display, 5, index_trade_info, 3, "index-5min")
        else:
            log.warning("Symbol '%s' not in INDEX_MAP — spot index chart skipped.", symbol_name)

        time.sleep(2)

    # ---- RUN SUMMARY ----
    log.info("=" * 60)
    log.info("EOD Journal — DONE")
    log.info("  Rows processed : %d", stats["processed"])
    log.info("  Rows skipped   : %d", stats["skipped"])
    log.info("  Charts OK      : %d", stats["charts_ok"])
    log.info("  Charts FAILED  : %d", stats["charts_fail"])
    log.info("  Uploads FAILED : %d", stats["uploads_fail"])
    log.info("  Sheet writes   : %d failed", stats["sheet_fail"])
    if stats["charts_fail"] or stats["uploads_fail"] or stats["sheet_fail"]:
        log.warning("Some steps failed — review the ERROR lines above for details.")
    else:
        log.info("All steps completed successfully.")
    log.info("=" * 60)


if __name__ == "__main__":
    run_eod_journal()
