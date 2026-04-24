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
import json
import tempfile
import time
import urllib3
from collections import defaultdict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
IMGBB_API_KEY = os.environ.get("IMGBB_API_KEY", "")

SHEET_NAME = "Abhay"
TAB_NAME = "mStock working"
KEYS_TAB = "KeyAndRules" # Pulling directly from your existing tab!

# Mapping the Spot Indices (Exchange/Token)
INDEX_MAP = {
    "NIFTY": "1/26000",       # NSE, Token 26000
    "BANKNIFTY": "1/26009",   # NSE, Token 26009
    "SENSEX": "4/51",         # BSE, Token 51
    "BANKEX": "4/69",         # BSE, Token 69
}

# mStock / Mirae segment codes — used to build the intraday chart URL.
# NFO = NSE F&O (NIFTY, BANKNIFTY options); BFO = BSE F&O (SENSEX, BANKEX options).
EXCHANGE_SEGMENT_MAP = {
    "NSE": "1",
    "NFO": "2",
    "CDS": "3",
    "BSE": "4",
    "BFO": "5",
    "BCD": "6",
    "MCX": "7",
}

# ==========================================
# 🔐 1. AUTHENTICATE SHEETS
# ==========================================

def authenticate_google():
    print("🔐 Connecting to Google Workspace...")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    secret_json = os.environ.get("GOOGLE_SECRET_KEY")
    if secret_json:
        creds_dict = json.loads(secret_json)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    else:
        creds = ServiceAccountCredentials.from_json_keyfile_name("google_secret_key.json", scope)
    
    gc = gspread.authorize(creds)
    return gc
          
# def authenticate_google():
    # print("🔐 Connecting to Google Workspace...")
    # scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # creds = ServiceAccountCredentials.from_json_keyfile_name("google_secret_key.json", scope)
    # gc = gspread.authorize(creds)
    # return gc

# ==========================================
# 📥 2. DOWNLOAD MASTER FILE
# ==========================================
def load_instrument_master(api_key, jwt_token):
    print("📥 Downloading mStock Security Master...")
    url = "https://api.mstock.trade/openapi/typea/instruments/scriptmaster"
    headers = {"X-Mirae-Version": "1", "Authorization": f"token {api_key}:{jwt_token}"}
    
    response = requests.get(url, headers=headers, verify=False)
    
    if response.status_code != 200:
        try:
            err_data = response.json()
            print(f"❌ MASTER API ERROR [{response.status_code}]: {err_data.get('error_type')} - {err_data.get('message')}")
        except:
            print(f"❌ MASTER API ERROR [{response.status_code}]: Failed to fetch master file.")
        return None
        
    df = pd.read_csv(io.StringIO(response.text))
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    return df

# ==========================================
# 🧩 3. PARSE CUSTOM SYMBOL
# ==========================================
def get_token_from_string(df, custom_symbol):
    try:
        parts = custom_symbol.split('-')
        name = parts[0] 
        expiry_date = pd.to_datetime(parts[1]).strftime('%Y-%m-%d')
        strike = float(parts[2]) 
        opt_type = parts[3] 
        
        filtered_df = df[
            (df['name'] == name) & 
            (df['strike'] == strike) & 
            (df['instrument_type'] == opt_type) & 
            (df['expiry'] == expiry_date)
        ]
        
        if not filtered_df.empty:
            row = filtered_df.iloc[0]
            token = row['instrument_token']
            exchange = str(row.get('exchange', '')).upper()
            # SENSEX/BANKEX options sit on BFO (segment 5), NIFTY/BANKNIFTY on NFO (2).
            segment = EXCHANGE_SEGMENT_MAP.get(exchange, "2")
            return token, segment
    except Exception as e:
        print(f"⚠️ Error parsing symbol {custom_symbol}: {e}")
    return None, None

# ==========================================
# 📈 4. FETCH DATA & PLOT CHART (CLEAN PRICE ONLY)
# ==========================================
def generate_chart(endpoint_path, display_name, api_key, jwt_token, timeframe=1, trade_info=None):
    tf_label = f"{timeframe}min"
    print(f"📊 Generating {tf_label} chart for {display_name}...")
    
    # endpoint_path handles both options ("2/token") and indices ("1/26000") dynamically!
    url = f"https://api.mstock.trade/openapi/typea/instruments/intraday/{endpoint_path}/minute"
    headers = {"X-Mirae-Version": "1", "Authorization": f"token {api_key}:{jwt_token}"}
    
    response = requests.get(url, headers=headers, verify=False)
    
    if response.status_code != 200:
        try:
            err_data = response.json()
            print(f"❌ CHART API ERROR [{response.status_code}] for {display_name}: {err_data.get('error_type')} - {err_data.get('message')}")
        except:
            print(f"❌ CHART API ERROR [{response.status_code}] for {display_name}: Server rejected request.")
        return None

    data = response.json()
    
    if data.get("status") != "success":
        print(f"❌ API REJECTED {display_name}: {data.get('message', 'Unknown error')}")
        return None
        
    if "candles" not in data.get("data", {}):
        print(f"⚠️ NO DATA: Exchange returned zero candles for {display_name}.")
        return None

    raw_candles = data["data"]["candles"]
    if not raw_candles:
        print(f"⚠️ EMPTY DATA: Candle list is empty for {display_name}.")
        return None
        
    raw_candles.reverse()
    
    df = pd.DataFrame(raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    df['timestamp'] = df['timestamp'].astype(str).str.split('+').str[0]
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.set_index('timestamp', inplace=True)
    df = df.between_time('09:15', '15:30')

    # Snapshot 1-min data BEFORE resampling. MFE/MAE must use the finest
    # granularity available — a 5-min resample hides intra-5min highs/lows
    # which often contain the true peak favorable / adverse excursion.
    df_1min = df.reset_index().copy()

    if timeframe == 5:
        df = df.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
    df.reset_index(inplace=True)
    
    if df.empty:
        print(f"⚠️ NO MARKET HOURS DATA: {display_name} had no trades between 09:15 and 15:30.")
        return None
        
    green_color, red_color, wick_width = '#26a69a', '#ef5350', 1.0
    
    fig, ax = plt.subplots(figsize=(18, 8), facecolor='white') 
    ax.set_facecolor('white')
    
    df['date_num'] = mdates.date2num(df['timestamp'])
    width = 0.0005 if timeframe == 1 else 0.0025 
    
    up, down = df[df.close >= df.open], df[df.close < df.open]
    
    ax.vlines(up.date_num, up.low, up.high, color=green_color, linewidth=wick_width)
    ax.vlines(down.date_num, down.low, down.high, color=red_color, linewidth=wick_width)
    ax.bar(up.date_num, up.close-up.open, width, bottom=up.open, color=green_color, edgecolor=green_color)
    ax.bar(down.date_num, down.close-down.open, width, bottom=down.open, color=red_color, edgecolor=red_color)
    
    # ==========================================
    # 🎯 TRADE OVERLAY (Entry/Exit markers)
    # ==========================================
    if trade_info:
        try:
            buy_time = pd.to_datetime(trade_info['buy_time'])
            sell_time = pd.to_datetime(trade_info['sell_time'])
            buy_price = float(trade_info['buy_price'])
            sell_price = float(trade_info['sell_price'])
            pnl = float(trade_info['pnl'])
            direction = str(trade_info.get('direction', 'BUY')).upper()
            pct = trade_info.get('percentage', '')

            buy_num = mdates.date2num(buy_time)
            sell_num = mdates.date2num(sell_time)
            has_prices = not (np.isnan(buy_price) or np.isnan(sell_price))

            # Chronological start/end — for SELL (short) trades, sell comes first
            start_num, end_num = (buy_num, sell_num) if buy_num <= sell_num else (sell_num, buy_num)
            start_time, end_time = (buy_time, sell_time) if buy_time <= sell_time else (sell_time, buy_time)

            # --- (2) Shaded duration band, colored by P&L ---
            band_color = '#c8e6c9' if pnl >= 0 else '#ffcdd2'
            ax.axvspan(start_num, end_num, alpha=0.30, color=band_color, zorder=0)

            # --- (1a) Vertical dashed lines at entry/exit ---
            ax.axvline(buy_num,  color='#1b5e20', linestyle='--', linewidth=1.3, alpha=0.85, zorder=3)
            ax.axvline(sell_num, color='#b71c1c', linestyle='--', linewidth=1.3, alpha=0.85, zorder=3)

            # --- (1b) Full-day horizontal price lines + inline price tags ---
            # Horizontal lines span the full session so we can instantly see
            # post-exit behaviour: did price keep running in our direction
            # (RR could have been bigger) or pull back through our entry
            # (good that we exited). Helps tune RR for next time.
            # Tags and direction triangles stay INSIDE the chart to keep the
            # candle area from getting compressed by an outside gutter.
            # Skipped on index charts where the price scale doesn't match.
            if has_prices:
                buy_green = '#1b5e20'
                sell_red  = '#b71c1c'

                # Full-day horizontal dashed lines at entry/exit prices
                ax.axhline(buy_price,  color=buy_green, linestyle='--',
                           linewidth=1.3, alpha=0.75, zorder=2)
                ax.axhline(sell_price, color=sell_red,  linestyle='--',
                           linewidth=1.3, alpha=0.75, zorder=2)

                # Price tags placed INSIDE the chart at the right edge. May
                # clip the final candle occasionally — accepted tradeoff so the
                # candle area isn't squeezed by an outside gutter.
                y_transform = ax.get_yaxis_transform()  # x=axes frac, y=data
                ax.annotate(f' BUY ₹{buy_price:g} ',
                            xy=(1.0, buy_price), xycoords=y_transform,
                            xytext=(-4, 0), textcoords='offset points',
                            fontsize=9, fontweight='bold', color='white',
                            ha='right', va='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=buy_green,
                                      edgecolor='none', alpha=0.95),
                            zorder=6)
                ax.annotate(f' SELL ₹{sell_price:g} ',
                            xy=(1.0, sell_price), xycoords=y_transform,
                            xytext=(-4, 0), textcoords='offset points',
                            fontsize=9, fontweight='bold', color='white',
                            ha='right', va='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=sell_red,
                                      edgecolor='none', alpha=0.95),
                            zorder=6)

                # Direction triangles pinned to the lowest-low of the visible
                # candles — no ylim expansion, no extra whitespace. Occasional
                # wick overlap on the very lowest candle is acceptable.
                triangle_y = df['low'].min()
                ax.scatter(buy_num,  triangle_y, marker='^', s=320,
                           color=buy_green, edgecolor='white', linewidth=1.4,
                           zorder=5, clip_on=False)
                ax.scatter(sell_num, triangle_y, marker='v', s=320,
                           color=sell_red,  edgecolor='white', linewidth=1.4,
                           zorder=5, clip_on=False)

                # --- MFE / MAE analysis on 1-min granularity (during-trade only) ---
                # LONG  (direction='BUY'):  entry=buy_price,  exit=sell_price
                # SHORT (direction='SELL'): entry=sell_price, exit=buy_price
                is_long = (direction == 'BUY')
                entry_price = buy_price if is_long else sell_price
                R = abs(buy_price - sell_price)  # realized risk baseline

                in_trade = df_1min[(df_1min['timestamp'] >= start_time) &
                                   (df_1min['timestamp'] <= end_time)]
                if in_trade.empty:
                    mfe_during = mae_during = 0.0
                else:
                    hi, lo = in_trade['high'].max(), in_trade['low'].min()
                    if is_long:
                        mfe_during = max(hi - entry_price, 0.0)
                        mae_during = max(entry_price - lo, 0.0)
                    else:
                        mfe_during = max(entry_price - lo, 0.0)
                        mae_during = max(hi - entry_price, 0.0)

                def _r(amount):
                    return f" ({amount / R:.2f}R)" if R > 0 else ""

                mfe_mae_line = (f"During trade: MFE +₹{mfe_during:.2f}{_r(mfe_during)}  |  "
                                f"MAE -₹{mae_during:.2f}{_r(mae_during)}")
            else:
                # --- Index chart: reference lines at SPOT level at entry/exit ---
                # Option premium doesn't apply here, but the spot index level at
                # buy_time / sell_time does — shows where the underlying was
                # when we opened and closed the option position. Same visual
                # vocabulary as the option chart: full-day dashed line +
                # inline price tag + bottom direction triangle.
                try:
                    buy_green = '#1b5e20'
                    sell_red  = '#b71c1c'

                    buy_idx  = (df['timestamp'] - buy_time ).abs().idxmin()
                    sell_idx = (df['timestamp'] - sell_time).abs().idxmin()
                    spot_at_buy  = float(df.loc[buy_idx,  'close'])
                    spot_at_sell = float(df.loc[sell_idx, 'close'])

                    ax.axhline(spot_at_buy,  color=buy_green, linestyle='--',
                               linewidth=1.3, alpha=0.75, zorder=2)
                    ax.axhline(spot_at_sell, color=sell_red,  linestyle='--',
                               linewidth=1.3, alpha=0.75, zorder=2)

                    y_transform = ax.get_yaxis_transform()
                    ax.annotate(f' BUY {spot_at_buy:.2f} ',
                                xy=(1.0, spot_at_buy), xycoords=y_transform,
                                xytext=(-4, 0), textcoords='offset points',
                                fontsize=9, fontweight='bold', color='white',
                                ha='right', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=buy_green,
                                          edgecolor='none', alpha=0.95),
                                zorder=6)
                    ax.annotate(f' SELL {spot_at_sell:.2f} ',
                                xy=(1.0, spot_at_sell), xycoords=y_transform,
                                xytext=(-4, 0), textcoords='offset points',
                                fontsize=9, fontweight='bold', color='white',
                                ha='right', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=sell_red,
                                          edgecolor='none', alpha=0.95),
                                zorder=6)

                    triangle_y = df['low'].min()
                    ax.scatter(buy_num,  triangle_y, marker='^', s=320,
                               color=buy_green, edgecolor='white', linewidth=1.4,
                               zorder=5, clip_on=False)
                    ax.scatter(sell_num, triangle_y, marker='v', s=320,
                               color=sell_red,  edgecolor='white', linewidth=1.4,
                               zorder=5, clip_on=False)
                except Exception as e:
                    print(f"⚠️ Could not draw index reference lines for {display_name}: {e}")

            # --- P&L badge in top-left corner ---
            badge_color = '#2e7d32' if pnl >= 0 else '#c62828'
            duration = end_time - start_time
            dur_str = str(duration).split('.')[0]
            pnl_sign = '+' if pnl >= 0 else ''
            trade_type = 'OPTION BUY' if direction == 'BUY' else 'OPTION SELL'
            line1 = f"{trade_type}  |  P&L: {pnl_sign}₹{pnl:.2f}"
            if pct:
                line1 += f"  ({pct})"
            line1 += f"  |  Duration: {dur_str}"

            # --- Day-context line (only when more than one trade on this symbol/date) ---
            total_trades = int(trade_info.get('total_trades') or 1)
            trade_number = int(trade_info.get('trade_number') or 1)
            day_pnl = trade_info.get('day_pnl')
            badge_lines = [line1]
            if total_trades > 1 and day_pnl is not None:
                try:
                    day_pnl_f = float(day_pnl)
                    day_sign = '+' if day_pnl_f >= 0 else ''
                    wins = int(trade_info.get('day_wins') or 0)
                    losses = int(trade_info.get('day_losses') or 0)
                    line2 = (f"Trade {trade_number} of {total_trades} today  |  "
                             f"Wins: {wins}  Losses: {losses}  |  "
                             f"Day Net: {day_sign}₹{day_pnl_f:.2f}")
                    badge_lines.append(line2)
                except Exception:
                    pass

            # Append MFE/MAE during-trade line (only on option charts)
            if has_prices:
                badge_lines.append(mfe_mae_line)

            badge_text = "\n".join(badge_lines)

            ax.text(0.01, 0.98, badge_text, transform=ax.transAxes,
                    fontsize=11, fontweight='bold', color='white',
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=badge_color,
                              edgecolor='none', alpha=0.92),
                    zorder=7)
        except Exception as e:
            print(f"⚠️ Could not draw trade markers for {display_name}: {e}")

    # Smart Grid Intervals
    symbol_name = display_name.split('-')[0].upper()
    if symbol_name in ["BANKNIFTY", "SENSEX", "BANKEX",
                       "BANKNIFTY INDEX", "SENSEX INDEX", "BANKEX INDEX"]:
        interval = 20
    elif symbol_name in ["NIFTY", "NIFTY INDEX"]:
        interval = 10
    else:
        interval = 10
        
    ax.yaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax.yaxis.tick_right()
    
    label_text = 'Spot Index Price (₹)' if 'INDEX' in display_name else 'Premium Price (₹)'
    ax.set_ylabel(label_text, color='black', fontsize=12, fontweight='bold')
    
    ax.tick_params(axis='y', colors='black', labelsize=10)
    ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7)
    ax.set_title(f'{display_name} ({tf_label} Chart)', color='black', fontsize=18, fontweight='bold')
    
    tick_interval = 5 if timeframe == 1 else 15
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, tick_interval)))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='x', rotation=90, colors='black', labelsize=10)

    # Hard-clip the x-axis to the actual candle range so the chart doesn't
    # waste 15 minutes of whitespace before 09:15 (the MinuteLocator would
    # otherwise pull xmin back to a 09:00 tick).
    xmin = df['date_num'].min()
    xmax = df['date_num'].max()
    xpad = (xmax - xmin) * 0.003
    ax.set_xlim(xmin - xpad, xmax + xpad)

    plt.tight_layout() 
    
    filename = f"{display_name}_{tf_label}_chart.png"
    plt.savefig(filename, facecolor=fig.get_facecolor(), dpi=300, bbox_inches='tight') 
    plt.close(fig) 
    return filename

# ==========================================
# ☁️ 5. UPLOAD TO IMGBB
# ==========================================
def upload_image(filename):
    print(f"☁️ Uploading {filename} to Cloud...")
    url = "https://api.imgbb.com/1/upload"
    payload = {"key": IMGBB_API_KEY}
    
    with open(filename, "rb") as file:
        response = requests.post(url, data=payload, files={"image": file}, verify=False)
        
    data = response.json()
    if data.get("success"):
        return data["data"]["url"]
    return None

# ==========================================
# 🚀 6. MAIN EXECUTION
# ==========================================
def run_eod_journal():
    print("🚀 Starting End-of-Day Journal Processor...")
    gc = authenticate_google()
    
    # --- FETCH CREDENTIALS DIRECTLY FROM KEYANDRULES ---
    try:
        print(f"🔑 Fetching mStock credentials from '{KEYS_TAB}' tab...")
        settings_sheet = gc.open(SHEET_NAME).worksheet(KEYS_TAB)
        api_key = settings_sheet.acell('A2').value
        jwt_token = settings_sheet.acell('B2').value
        print("✅ Credentials loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR: Could not read API keys from '{KEYS_TAB}' tab. ({e})")
        return
    # ----------------------------------------------------

    sheet = gc.open(SHEET_NAME).worksheet(TAB_NAME)
    raw_data = sheet.get_all_values() 
    master_df = None 
    
    header_row_index = -1
    for i, row in enumerate(raw_data):
        if 'IndexTraded' in row:
            header_row_index = i
            break
            
    if header_row_index == -1:
        print("❌ ERROR: Could not find 'IndexTraded' anywhere in the sheet!")
        return
        
    print(f"🎯 Found headers at Sheet Row {header_row_index + 1}")
    headers = raw_data[header_row_index]
    
    try:
        status_idx = headers.index('TradeStatus')
        chart_idx = headers.index('Chart')
        symbol_idx = headers.index('IndexTraded')
        buy_time_idx = headers.index('BuyTime')
        sell_time_idx = headers.index('SellTime')
        buy_price_idx = headers.index('BuyPrice')
        sell_price_idx = headers.index('SellPrice')
        pnl_idx = headers.index('Pnl')
        direction_idx = headers.index('Direction')
    except ValueError as e:
        print(f"❌ ERROR: Missing a required header: {e}")
        return

    # Percentage% and Date are optional — fall back if missing
    try:
        pct_idx = headers.index('Percentage%')
    except ValueError:
        pct_idx = -1
    try:
        date_idx = headers.index('Date')
    except ValueError:
        date_idx = -1

    def safe_get(row, i):
        return row[i] if 0 <= i < len(row) else ''

    # ==========================================
    # 📊 BUILD DAY-CONTEXT (multi-trade awareness)
    # ==========================================
    # Group CLOSED rows by (Date, Symbol). For each group, compute:
    #   - total_trades, day_pnl, day_wins, day_losses
    #   - trade_number (chronological rank by BuyTime) for each row
    data_rows = raw_data[header_row_index + 1:]
    trade_groups = defaultdict(list)  # key: (date_str, symbol) -> list of (row_idx, buy_time, pnl_val)

    for ridx, row in enumerate(data_rows):
        if safe_get(row, status_idx) != 'CLOSED':
            continue
        sym = safe_get(row, symbol_idx)
        if not sym:
            continue
        bt_raw = safe_get(row, buy_time_idx)
        try:
            bt_val = pd.to_datetime(bt_raw)
        except Exception:
            bt_val = pd.Timestamp.min

        date_str = safe_get(row, date_idx) if date_idx >= 0 else ''
        if not date_str:
            try:
                date_str = str(bt_val.date())
            except Exception:
                date_str = ''

        try:
            pnl_val = float(safe_get(row, pnl_idx))
        except Exception:
            pnl_val = 0.0

        trade_groups[(date_str, sym)].append((ridx, bt_val, pnl_val))

    row_context = {}  # ridx -> dict of day-context fields
    for key, trades in trade_groups.items():
        trades_sorted = sorted(trades, key=lambda t: t[1])
        total = len(trades_sorted)
        day_pnl = sum(t[2] for t in trades_sorted)
        wins = sum(1 for t in trades_sorted if t[2] > 0)
        losses = sum(1 for t in trades_sorted if t[2] < 0)
        for rank, (ridx, _, _) in enumerate(trades_sorted, start=1):
            row_context[ridx] = {
                'trade_number': rank,
                'total_trades': total,
                'day_pnl': day_pnl,
                'day_wins': wins,
                'day_losses': losses,
            }

    for idx, row in enumerate(data_rows):
        row_number = header_row_index + 2 + idx 
        
        status = safe_get(row, status_idx)
        chart_cell = safe_get(row, chart_idx)
        custom_symbol = safe_get(row, symbol_idx)
        
        if status == 'CLOSED' and chart_cell == '' and custom_symbol != '':
            
            if master_df is None:
                master_df = load_instrument_master(api_key, jwt_token)
                if master_df is None:
                    return 
                
            token, segment = get_token_from_string(master_df, custom_symbol)
            
            if token:
                endpoint_path = f"{segment}/{token}"
                # --- Build trade_info from the row (for entry/exit overlay) ---
                trade_info = None
                try:
                    trade_info = {
                        'buy_time':   safe_get(row, buy_time_idx),
                        'sell_time':  safe_get(row, sell_time_idx),
                        'buy_price':  safe_get(row, buy_price_idx),
                        'sell_price': safe_get(row, sell_price_idx),
                        'pnl':        safe_get(row, pnl_idx),
                        'direction':  safe_get(row, direction_idx),
                        'percentage': safe_get(row, pct_idx) if pct_idx >= 0 else '',
                    }
                    # Skip overlay if any core field is empty
                    if not all([trade_info['buy_time'], trade_info['sell_time'],
                                trade_info['buy_price'], trade_info['sell_price']]):
                        trade_info = None
                    else:
                        # Attach day-context (multi-trade awareness) if available
                        ctx = row_context.get(idx, {})
                        trade_info['trade_number'] = ctx.get('trade_number', 1)
                        trade_info['total_trades'] = ctx.get('total_trades', 1)
                        trade_info['day_pnl']      = ctx.get('day_pnl', 0.0)
                        trade_info['day_wins']     = ctx.get('day_wins', 0)
                        trade_info['day_losses']   = ctx.get('day_losses', 0)
                except Exception as e:
                    print(f"⚠️ Could not build trade_info: {e}")
                    trade_info = None

                # 1. Option 1-Min Chart (Col Y)
                chart_1min = generate_chart(endpoint_path, custom_symbol, api_key, jwt_token,
                                            timeframe=1, trade_info=trade_info)
                if chart_1min:
                    url_1min = upload_image(chart_1min)
                    if url_1min:
                        formula_1 = f'=HYPERLINK("{url_1min}", "{custom_symbol}-1min.png")'
                        sheet.update_cell(row_number, chart_idx + 1, formula_1) 
                        print(f"✅ Injected Option 1-Min Link")
                
                # 2. Option 5-Min Chart (Col Z)
                chart_5min = generate_chart(endpoint_path, custom_symbol, api_key, jwt_token,
                                            timeframe=5, trade_info=trade_info)
                if chart_5min:
                    url_5min = upload_image(chart_5min)
                    if url_5min:
                        formula_5 = f'=HYPERLINK("{url_5min}", "{custom_symbol}-5min.png")'
                        sheet.update_cell(row_number, chart_idx + 2, formula_5) 
                        print(f"✅ Injected Option 5-Min Link")
                
                # 3. Spot Index 5-Min Chart (Col AA) — also overlays trade times for market context
                symbol_name = custom_symbol.split('-')[0].upper()
                if symbol_name in INDEX_MAP:
                    index_endpoint = INDEX_MAP[symbol_name]
                    index_display_name = f"{symbol_name} INDEX"

                    # For the index chart, the buy/sell PRICE markers don't make sense
                    # (they refer to option premium, not index level).
                    # So we pass a stripped-down overlay: just the time band + badge.
                    index_trade_info = None
                    if trade_info:
                        index_trade_info = dict(trade_info)
                        # Hide price markers on index chart by setting prices to NaN
                        # (scatter will just skip NaN points cleanly)
                        index_trade_info['buy_price'] = float('nan')
                        index_trade_info['sell_price'] = float('nan')

                    chart_idx_5min = generate_chart(index_endpoint, index_display_name, api_key, jwt_token,
                                                    timeframe=5, trade_info=index_trade_info)
                    if chart_idx_5min:
                        url_idx_5min = upload_image(chart_idx_5min)
                        if url_idx_5min:
                            formula_idx = f'=HYPERLINK("{url_idx_5min}", "{index_display_name}-5min.png")'
                            sheet.update_cell(row_number, chart_idx + 3, formula_idx) # +3 places it perfectly in Col AA!
                            print(f"✅ Injected Spot Index 5-Min Link")

                time.sleep(2) 

if __name__ == "__main__":
    run_eod_journal()
