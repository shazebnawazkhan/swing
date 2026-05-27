

import pandas as pd
# import growwapi
import talib as ta
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import datetime

# Authentication

from growwapi import GrowwAPI

import pandas as pd
from backtesting import Strategy
from backtesting import Backtest

def SIGNAL():
    return df.TotalSignal


# Groww API Credentials (Replace with your actual credentials)
API_AUTH_TOKEN = "eyJraWQiOiJaTUtjVXciLCJhbGciOiJFUzI1NiJ9.eyJleHAiOjE3NTczNzc4MDAsImlhdCI6MTc1NzM1NTA2NCwibmJmIjoxNzU3MzU1MDY0LCJzdWIiOiJ7XCJ0b2tlblJlZklkXCI6XCIxZTU4MThiMy02NjRhLTQwMDgtYTQzNy1kNzEyZTdkYzkyYjJcIixcInZlbmRvckludGVncmF0aW9uS2V5XCI6XCJlMzFmZjIzYjA4NmI0MDZjODg3NGIyZjZkODQ5NTMxM1wiLFwidXNlckFjY291bnRJZFwiOlwiZjVjNDJmMWUtZTkxYS00NjM0LTg4ZjUtZDdiZmZlYzAyNjA0XCIsXCJkZXZpY2VJZFwiOlwiNjdjNGY3MWEtNmQ1MS01NzY1LWJlNjMtMTU2OTE1MzY5NWVhXCIsXCJzZXNzaW9uSWRcIjpcIjEzNzUxNDQxLThkMjktNDgwNi1hYjRkLTYwZjU4NDJkMDRlNFwiLFwiYWRkaXRpb25hbERhdGFcIjpcIno1NC9NZzltdjE2WXdmb0gvS0EwYk5yQ2l5VDNVOGh0S1RUSlNjQktJb2RSTkczdTlLa2pWZDNoWjU1ZStNZERhWXBOVi9UOUxIRmtQejFFQisybTdRPT1cIixcInJvbGVcIjpcIm9yZGVyLWJhc2ljLGxpdmVfZGF0YS1iYXNpYyxub25fdHJhZGluZy1iYXNpYyxvcmRlcl9yZWFkX29ubHktYmFzaWNcIixcInNvdXJjZUlwQWRkcmVzc1wiOlwiNDkuMzYuMTY5LjEyMywxNzIuNjguMjE0LjUsMzUuMjQxLjIzLjEyM1wiLFwidHdvRmFFeHBpcnlUc1wiOjE3NTczNzc4MDAwMDB9IiwiaXNzIjoiYXBleC1hdXRoLXByb2QtYXBwIn0.LN5QzHhA4ZM_eP8mUYO0Z_tmJAVCl_z-TfjgGESPd6cDH0UHqYX7Mb6mvjmO3WS6-lkbZAv5VTjS7uJX7bKZ8A"

# Initialize Groww API
groww = GrowwAPI(API_AUTH_TOKEN)

import numpy as np
def pointpos(x):
    if x['TotalSignal']==2:
        return x['Low']-1e-3
    elif x['TotalSignal']==1:
        return x['High']+1e-3
    else:
        return np.nan

def ema_signal(df, current_candle, backcandles):
    df_slice = df.reset_index().copy()
    # Get the range of candles to consider
    start = max(0, current_candle - backcandles)
    end = current_candle
    relevant_rows = df_slice.iloc[start:end]

    # Check if all EMA_fast values are below EMA_slow values
    if all(relevant_rows["EMA_fast"] < relevant_rows["EMA_slow"]):
        return 1
    elif all(relevant_rows["EMA_fast"] > relevant_rows["EMA_slow"]):
        return 2
    else:
        return 0


def total_signal(df, current_candle, backcandles):
    if (ema_signal(df, current_candle, backcandles) == 2
            and df.Close[current_candle] <= df['BBANDS_lo'][current_candle]
            # and df.RSI[current_candle]<60
    ):
        return 2
    if (ema_signal(df, current_candle, backcandles) == 1
            and df.Close[current_candle] >= df['BBANDS_up'][current_candle]
            # and df.RSI[current_candle]>40
    ):
        return 1
    return 0


class MyStrat(Strategy):
    mysize = 3000
    slcoef = 1.1
    TPSLRatio = 1.2
    rsi_length = 10

    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)
        # df['RSI']=ta.rsi(df.Close, length=self.rsi_length)

    def next(self):
        super().next()
        slatr = self.slcoef * self.data.ATR[-1]
        TPSLRatio = self.TPSLRatio

        # if len(self.trades)>0:
        #     if self.trades[-1].is_long and self.data.RSI[-1]>=90:
        #         self.trades[-1].close()
        #     elif self.trades[-1].is_short and self.data.RSI[-1]<=10:
        #         self.trades[-1].close()

        if self.signal1 == 2 and len(self.trades) == 0:
            sl1 = self.data.Close[-1] - slatr
            tp1 = self.data.Close[-1] + slatr * TPSLRatio
            self.buy(sl=sl1, tp=tp1, size=self.mysize)

        elif self.signal1 == 1 and len(self.trades) == 0:
            sl1 = self.data.Close[-1] + slatr
            tp1 = self.data.Close[-1] - slatr * TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize)

# holdings_response = groww.get_holdings_for_user(timeout=5)
# print(holdings_response)
#
#
# positions_response = groww.get_positions_for_user(segment=groww.SEGMENT_CASH)
# print(positions_response)

# you can give start time and end time in yyyy-MM-dd HH:mm:ss format.


symbol_list = ["PCJEWELLER", "TBZ"]


def data_pull(symbol):
    end_time = "2025-09-08 15:30:00"
    start_time = "2025-09-08 09:15:00"

    historical_data_response = groww.get_historical_candle_data(
        trading_symbol=symbol,
        exchange=groww.EXCHANGE_NSE,
        segment=groww.SEGMENT_CASH,
        start_time=start_time,
        end_time=end_time,
        interval_in_minutes=1  # Optional: Interval in minutes for the candle data
    )
    # print(historical_data_response)

    data = historical_data_response["candles"]

    df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])

    df['Timestamp']=(pd.to_datetime(df['Timestamp'], unit='s') + pd.Timedelta(hours=5, minutes=30))
    df.set_index("Timestamp", inplace=True)
    return df







df = data_pull('IOLCP')
df["EMA_slow"]=ta.EMA(df.Close, timeperiod=30)
df["EMA_fast"]=ta.EMA(df.Close, timeperiod=10)
df['RSI']=ta.RSI(df.Close, timeperiod=16)
#my_bbands = ta.BBANDS(df.Close, length=15, std=1.5)
df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=7)
df['BBANDS_up']    = ta.BBANDS(df['Close'], timeperiod=15, nbdevup=1.5, nbdevdn=1.5)[0]
df['BBANDS_md']    = ta.BBANDS(df['Close'], timeperiod=15, nbdevup=1.5, nbdevdn=1.5)[1]
df['BBANDS_lo']    = ta.BBANDS(df['Close'], timeperiod=15, nbdevup=1.5, nbdevdn=1.5)[2]


df=df[-10000:-1]
# from tqdm import tqdm
# tqdm.pandas()
df.reset_index(inplace=True)
df['EMASignal'] = df.apply(lambda row: ema_signal(df, row.name, 7) , axis=1) #if row.name >= 20 else 0
df['TotalSignal'] = df.apply(lambda row: total_signal(df, row.name, 7), axis=1)

#print(df[df.TotalSignal == 1].shape)

df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)


#df.to_csv("PCJ.csv", index=False)

#print(df)

#print("Hi")








#df = pd.read_csv("signals.csv")
def generate_html_report(stats, symbol: str, chart_file: str = "backtest_chart.html") -> str:
    """
    Build a standalone HTML backtest report that matches the dashboard theme.
    Returns the output filename.
    """
    import os, math

    out_file = "backtest_report.html"

    def _fmt(val, suffix="", decimals=2):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "—"
        if isinstance(val, float):
            return f"{val:.{decimals}f}{suffix}"
        return f"{val}{suffix}"

    def _signed(val, suffix="%", decimals=2):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "—"
        sign = "+" if val >= 0 else ""
        color = "#15803d" if val >= 0 else "#b91c1c"
        return f'<span style="color:{color};font-weight:700">{sign}{val:.{decimals}f}{suffix}</span>'

    # Key stat cards
    cards = [
        ("Total Return",  _signed(stats.get("Return [%]",         float("nan")))),
        ("Win Rate",      _fmt(stats.get("Win Rate [%]",          float("nan")), "%")),
        ("Max Drawdown",  _signed(stats.get("Max. Drawdown [%]",  float("nan")))),
        ("Sharpe Ratio",  _fmt(stats.get("Sharpe Ratio",          float("nan")))),
        ("# Trades",      _fmt(stats.get("# Trades",              float("nan")), "", 0)),
        ("Profit Factor", _fmt(stats.get("Profit Factor",         float("nan")))),
    ]

    cards_html = "".join(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-val">{val}</div></div>'
        for label, val in cards
    )

    # Full stats table
    skip = {"_strategy", "_equity_curve", "_trades"}
    stat_rows = "".join(
        f"<tr><td>{k}</td><td>{_fmt(v) if isinstance(v, float) else v}</td></tr>"
        for k, v in stats.items()
        if k not in skip
    )

    # Trade log
    trades_df = stats.get("_trades")
    if trades_df is not None and not trades_df.empty:
        thead = "<tr>" + "".join(f"<th>{c}</th>" for c in trades_df.columns) + "</tr>"
        tbody = ""
        for _, row in trades_df.iterrows():
            cells = ""
            for c in trades_df.columns:
                v = row[c]
                if c == "ReturnPct":
                    cells += f"<td>{_signed(float(v) * 100 if not (isinstance(v, float) and math.isnan(v)) else float('nan'))}</td>"
                elif c == "PnL":
                    color = "#15803d" if float(v) >= 0 else "#b91c1c"
                    cells += f'<td style="color:{color};font-weight:600">₹{float(v):,.2f}</td>'
                else:
                    cells += f"<td>{v}</td>"
            tbody += f"<tr>{cells}</tr>"
        trade_table = f"<table><thead>{thead}</thead><tbody>{tbody}</tbody></table>"
    else:
        trade_table = "<p style='color:#6b7280'>No trades recorded.</p>"

    # Chart embed
    chart_section = ""
    if os.path.exists(chart_file):
        chart_section = f"""
        <section>
          <h2>Equity Curve &amp; Trades</h2>
          <iframe src="{chart_file}" style="width:100%;height:620px;border:1.5px solid #c8d0e0;border-radius:8px;background:#fff"></iframe>
        </section>"""

    strategy_name = str(stats.get("_strategy", "MyStrat")).split("(")[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report — {symbol}</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg: #ffffff; --surface: #f8f9ff; --border: #c8d0e0;
  --text: #0f1623; --muted: #3d4a62; --accent: #1d4ed8;
}}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.55; }}
header {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap;
          padding:12px 28px; background:#fff; border-bottom:2px solid var(--border);
          position:sticky; top:0; z-index:100; box-shadow:0 1px 4px rgba(0,0,0,.06); }}
header h1 {{ font-size:16px; font-weight:800; color:var(--accent); letter-spacing:.05em; }}
.badge {{ background:var(--surface); border:1px solid var(--border); border-radius:20px;
          padding:3px 11px; font-size:12px; font-weight:600; color:var(--muted); }}
.back-link {{ margin-left:auto; font-size:13px; font-weight:600; color:var(--accent);
              text-decoration:none; border:1.5px solid var(--border); border-radius:6px;
              padding:5px 13px; transition:border-color .15s; }}
.back-link:hover {{ border-color:var(--accent); }}
main {{ max-width:1280px; margin:0 auto; padding:24px 28px; }}
section {{ margin-bottom:36px; }}
h2 {{ font-size:15px; font-weight:800; color:var(--text); margin-bottom:14px;
       padding-bottom:8px; border-bottom:2px solid var(--border); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:12px;
          margin-bottom:28px; }}
.card {{ background:var(--surface); border:1.5px solid var(--border); border-radius:8px;
          padding:14px 18px; }}
.card-label {{ font-size:11px; font-weight:700; color:var(--muted); text-transform:uppercase;
               letter-spacing:.06em; margin-bottom:6px; }}
.card-val {{ font-size:22px; font-weight:800; color:var(--text); }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ background:var(--surface); font-weight:700; text-align:left;
      padding:8px 10px; border-bottom:2px solid var(--border); color:var(--muted);
      font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
td {{ padding:7px 10px; border-bottom:1px solid #eef0f6; }}
tr:hover td {{ background:#f5f7ff; }}
.stats-table td:first-child {{ font-weight:600; color:var(--muted); width:240px; }}
</style>
</head>
<body>
<header>
  <h1>Backtest Report</h1>
  <span class="badge">{symbol}</span>
  <span class="badge">{strategy_name}</span>
  <a class="back-link" href="dashboard.html">&#8592; Dashboard</a>
</header>
<main>
  <section>
    <h2>Key Metrics</h2>
    <div class="cards">{cards_html}</div>
  </section>
  {chart_section}
  <section>
    <h2>Trade Log</h2>
    {trade_table}
  </section>
  <section>
    <h2>Full Statistics</h2>
    <table class="stats-table">
      <tbody>{stat_rows}</tbody>
    </table>
  </section>
</main>
</body>
</html>"""

    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\nBacktest report written -> {out_file}")
    return out_file


bt = Backtest(df, MyStrat, cash=100000, margin=1 / 30)

stats = bt.run()

print(stats)

bt.plot(filename="backtest_chart.html", open_browser=False)

generate_html_report(stats, symbol="IOLCP")



import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
st=100
dfpl = df[st:st+350]
#dfpl.reset_index(inplace=True)
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close']),

                go.Scatter(x=dfpl.index, y=dfpl['BBANDS_lo'],
                           line=dict(color='red', width=1),
                           name="BBL"),
                go.Scatter(x=dfpl.index, y=dfpl['BBANDS_up'],
                           line=dict(color='green', width=1),
                           name="BBU"),
                go.Scatter(x=dfpl.index, y=dfpl['EMA_fast'],
                           line=dict(color='black', width=1),
                           name="EMA_fast"),
                go.Scatter(x=dfpl.index, y=dfpl['EMA_slow'],
                           line=dict(color='blue', width=1),
                           name="EMA_slow")])

fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="entry")

fig.show()