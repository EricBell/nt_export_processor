"""
nt_export_processor.py

Utilities to load NinjaTrader CSV exports, clean, resample to common intraday bar sizes,
compute indicators (SMA, EMA, RSI, MACD, ATR), plot, and export results.

Assumptions:
- CSV has either a single datetime column (e.g., 'DateTime') or separate 'Date' and 'Time' columns.
- OHLCV columns named 'Open','High','Low','Close','Volume' (case-insensitive).
- Timezone handling: if no tz info provided, treat as local; you can pass tz parameter.

Usage examples at bottom.
"""

from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

pd.options.mode.chained_assignment = None

# ----------------------
# Loading & Cleaning
# ----------------------
def load_ninjatrader_csv(path: str,
                         tz: Optional[str] = None) -> pd.DataFrame:
    """
    Load NinjaTrader exported CSV into a normalized DataFrame with a DateTime index and columns:
    Open, High, Low, Close, Volume
    
    Expected format: 'YYYYMMDD HHMMSS;open;high;low;close;volume'
    Example: '20250612 040100;6062.5;6062.5;6062.25;6062.5;14'
    """
    # Read CSV with semicolon delimiter and no header
    df = pd.read_csv(path, sep=';', header=None, dtype=str)
    print(f"Raw CSV shape: {df.shape}")
    print(f"First few rows:\n{df.head()}")
    
    # Assign column names based on known structure
    df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Parse datetime from 'YYYYMMDD HHMMSS' format
    dt = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S', errors='coerce')
    print(f"Parsed datetime count: {dt.notna().sum()} out of {len(dt)}")
    print(f"Sample parsed dates: {dt.dropna().head()}")
    
    if tz:
        dt = dt.dt.tz_localize(tz)
    
    # Convert OHLCV to numeric
    data = {
        'Open': pd.to_numeric(df['Open'], errors='coerce'),
        'High': pd.to_numeric(df['High'], errors='coerce'),
        'Low': pd.to_numeric(df['Low'], errors='coerce'),
        'Close': pd.to_numeric(df['Close'], errors='coerce'),
        'Volume': pd.to_numeric(df['Volume'], errors='coerce')
    }
    
    out = pd.DataFrame(data, index=dt)
    out.index.name = 'DateTime'
    # Drop rows missing core prices
    out = out.dropna(subset=['Open','High','Low','Close'])
    # Sort index
    out = out.sort_index()
    return out

# ----------------------
# Resampling
# ----------------------
def resample_bars(df: pd.DataFrame, timeframe: str = "1min", how: dict = None) -> pd.DataFrame:
    """
    Resample tick/trade-level or intrabar to OHLCV bars using pandas.resample
    timeframe examples: '1min', '5min', '1H', '1D'
    how default: open=first, high=max, low=min, close=last, volume=sum
    """
    if how is None:
        how = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}

    agg = df.resample(timeframe).agg(how)
    agg = agg.dropna(subset=['Open','High','Low','Close'])
    return agg

# ----------------------
# Indicators
# ----------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr

# ----------------------
# Indicator runner
# ----------------------
def add_indicators(df: pd.DataFrame,
                   sma_windows=(10, 20, 50),
                   ema_windows=(9, 21),
                   rsi_window=14,
                   macd_params=(12, 26, 9),
                   atr_window=14) -> pd.DataFrame:
    dd = df.copy()
    for w in sma_windows:
        dd[f"SMA_{w}"] = sma(dd['Close'], w)
    for w in ema_windows:
        dd[f"EMA_{w}"] = ema(dd['Close'], w)
    dd[f"RSI_{rsi_window}"] = rsi(dd['Close'], rsi_window)
    macd_line, signal_line, hist = macd(dd['Close'], *macd_params)
    dd['MACD'] = macd_line
    dd['MACD_Signal'] = signal_line
    dd['MACD_Hist'] = hist
    dd[f"ATR_{atr_window}"] = atr(dd, atr_window)
    return dd

# ----------------------
# Export helpers
# ----------------------
def export_df(df: pd.DataFrame, path: str, fmt: str = 'parquet'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == 'parquet':
        df.to_parquet(path, index=True)
    elif fmt == 'csv':
        df.to_csv(path, index=True)
    else:
        raise ValueError("fmt must be 'parquet' or 'csv'")

# ----------------------
# Plotting
# ----------------------
def plot_candles_with_indicators(df: pd.DataFrame, title: str = "Chart"):
    # Interactive Plotly candlestick + SMA + EMA + MACD subplot
    fig = make_candle_figure(df, title)
    fig.show()

def make_candle_figure(df: pd.DataFrame, title: str = "Chart"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='Price'))
    # Add SMA/EMA if present (first two)
    for col in df.columns:
        if col.startswith('SMA_') or col.startswith('EMA_'):
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
    # MACD subplot
    if 'MACD' in df.columns:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'],
                                     name='Price'), row=1, col=1)
        for col in df.columns:
            if col.startswith('SMA_') or col.startswith('EMA_'):
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD_Signal'), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD_Hist'), row=2, col=1)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    return fig

# ----------------------
# CLI / example usage
# ----------------------
if __name__ == "__main__":
    import typer

    def main(
        input_file: str = typer.Argument(help="Path to NinjaTrader CSV export"),
        out: str = typer.Option("./output", help="Output base path (dir)"),
        tz: Optional[str] = typer.Option(None, help="Timezone to localize (e.g., 'America/New_York')"),
        resample: str = typer.Option("1min", help="Resample timeframe e.g., '1min','5min','1H'"),
        fmt: str = typer.Option("parquet", help="Export format: parquet or csv")
    ):
        """Process NinjaTrader CSV export."""
        print("Loading:", input_file)
        df = load_ninjatrader_csv(input_file, tz=tz)
        print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
        bars = resample_bars(df, resample)
        bars_ind = add_indicators(bars)
        out_file = os.path.join(out, f"processed_{resample}.{fmt}")
        export_df(bars_ind, out_file, fmt=fmt)
        print("Exported to", out_file)
        # Optionally show chart
        try:
            fig = make_candle_figure(bars_ind, title=f"Processed {os.path.basename(input_file)}")
            fig.show()
        except Exception as e:
            print("Plot failed (headless?). Saved outputs only. Error:", e)

    typer.run(main)