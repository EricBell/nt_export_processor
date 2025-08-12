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
def detect_file_format(path: str) -> tuple[str, str]:
    """
    Detect file format: NT8 CSV, resampled CSV, or Parquet.
    Returns tuple: (format, timeframe) where format is 'nt8', 'resampled', or 'parquet',
    and timeframe is extracted from filename if present.
    """
    # Extract timeframe from filename if present (e.g., "file_3min.csv" -> "3min")
    filename = os.path.basename(path)
    timeframe = None
    if '_' in filename:
        parts = filename.split('_')
        for part in parts:
            # Look for timeframe patterns like "3min", "1H", "5min"
            if any(unit in part.lower() for unit in ['min', 'h', 'd']) and any(c.isdigit() for c in part):
                timeframe = part.split('.')[0]  # Remove extension
                break
    
    # Check if it's a Parquet file
    if path.lower().endswith('.parquet'):
        return 'parquet', timeframe
    
    # For text files, read first line to detect format
    try:
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Check if first line has headers (resampled CSV format)
        if 'DateTime' in first_line or 'Open' in first_line or first_line.count(',') > 3:
            return 'resampled', timeframe
        
        # Check NT8 format: YYYYMMDD HHMMSS;price;price;price;price;volume
        if ';' in first_line and len(first_line.split(';')) == 6:
            datetime_part = first_line.split(';')[0]
            if len(datetime_part) == 15 and datetime_part[8] == ' ':
                return 'nt8', timeframe
    
    except UnicodeDecodeError:
        # If we can't decode as text, it might be a binary format like Parquet
        if path.lower().endswith(('.parquet', '.pqt')):
            return 'parquet', timeframe
    
    return 'unknown', timeframe

def load_resampled_csv(path: str, tz: Optional[str] = None) -> pd.DataFrame:
    """
    Load previously resampled CSV file with DateTime index and OHLCV columns.
    Expected format: CSV with DateTime column and Open,High,Low,Close,Volume columns.
    """
    df = pd.read_csv(path)
    
    # Parse datetime column
    dt = pd.to_datetime(df['DateTime'], errors='coerce')
    
    if tz:
        dt = dt.dt.tz_localize(tz)
    
    # Create DataFrame with OHLCV data
    out = pd.DataFrame({
        'Open': pd.to_numeric(df['Open'], errors='coerce'),
        'High': pd.to_numeric(df['High'], errors='coerce'),
        'Low': pd.to_numeric(df['Low'], errors='coerce'),
        'Close': pd.to_numeric(df['Close'], errors='coerce'),
        'Volume': pd.to_numeric(df['Volume'], errors='coerce')
    })
    
    # Set the datetime index
    out.index = dt
    out.index.name = 'DateTime'
    # Drop rows missing core prices
    out = out.dropna(subset=['Open','High','Low','Close'])
    # Sort index
    out = out.sort_index()
    return out

def load_parquet_file(path: str, tz: Optional[str] = None) -> pd.DataFrame:
    """
    Load Parquet file with DateTime index and OHLCV columns.
    """
    df = pd.read_parquet(path)
    
    # Ensure DateTime is the index
    if 'DateTime' in df.columns:
        df = df.set_index('DateTime')
    
    # Apply timezone if specified
    if tz and df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    
    # Ensure we have the expected OHLCV columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Drop rows missing core prices
    df = df.dropna(subset=['Open','High','Low','Close'])
    # Sort index
    df = df.sort_index()
    return df

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
    
    # Assign column names based on known structure
    df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Parse datetime from 'YYYYMMDD HHMMSS' format
    dt = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S', errors='coerce')
    
    if tz:
        dt = dt.dt.tz_localize(tz)
    
    # Convert OHLCV to numeric and create DataFrame
    out = pd.DataFrame({
        'Open': pd.to_numeric(df['Open'], errors='coerce'),
        'High': pd.to_numeric(df['High'], errors='coerce'),
        'Low': pd.to_numeric(df['Low'], errors='coerce'),
        'Close': pd.to_numeric(df['Close'], errors='coerce'),
        'Volume': pd.to_numeric(df['Volume'], errors='coerce')
    })
    
    # Set the datetime index
    out.index = dt
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

def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF) - measures buying and selling pressure.
    
    CMF = Sum(Money Flow Volume for n periods) / Sum(Volume for n periods)
    Money Flow Volume = Volume * Money Flow Multiplier
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    
    Positive CMF indicates buying pressure, negative indicates selling pressure.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)  # Handle division by zero when high == low
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # CMF = Sum of MFV over window / Sum of Volume over window
    cmf_values = mfv.rolling(window).sum() / volume.rolling(window).sum()
    cmf_values = cmf_values.fillna(0)
    
    return cmf_values

# ----------------------
# Indicator runner
# ----------------------
def add_indicators(df: pd.DataFrame,
                   ema_windows=(9, 21),
                   rsi_window=14,
                   macd_params=(12, 26, 9),
                   atr_window=14,
                   cmf_window=20) -> pd.DataFrame:
    dd = df.copy()
    for w in ema_windows:
        dd[f"EMA_{w}"] = ema(dd['Close'], w)
    dd[f"RSI_{rsi_window}"] = rsi(dd['Close'], rsi_window)
    macd_line, signal_line, hist = macd(dd['Close'], *macd_params)
    dd['MACD'] = macd_line
    dd['MACD_Signal'] = signal_line
    dd['MACD_Hist'] = hist
    dd[f"ATR_{atr_window}"] = atr(dd, atr_window)
    dd[f"CMF_{cmf_window}"] = cmf(dd, cmf_window)
    return dd

# ----------------------
# Export helpers
# ----------------------
def generate_resampled_filename(input_path: str, timeframe: str, output_dir: str) -> str:
    """
    Generate output filename for resampled data.
    Example: input_path='data/MES-1-day-sample.txt', timeframe='3min'
             returns: 'output_dir/MES-1-day-sample_3min.csv'
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{base_name}_{timeframe}.csv"
    return os.path.join(output_dir, output_filename)
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
    # Add EMA if present
    for col in df.columns:
        if col.startswith('EMA_'):
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
    # Multi-panel subplot for MACD and CMF
    if 'MACD' in df.columns or any(col.startswith('CMF_') for col in df.columns):
        from plotly.subplots import make_subplots
        
        # Determine number of subplots needed
        has_macd = 'MACD' in df.columns
        has_cmf = any(col.startswith('CMF_') for col in df.columns)
        
        if has_macd and has_cmf:
            # 3 panels: Price, MACD, CMF
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                vertical_spacing=0.02,
                                row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=('Price & EMAs', 'MACD', 'CMF'))
            cmf_row = 3
        else:
            # 2 panels: Price and either MACD or CMF
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.05,
                                row_heights=[0.7, 0.3])
            cmf_row = 2
        
        # Add main price chart
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'],
                                     name='Price'), row=1, col=1)
        
        # Add EMA lines
        for col in df.columns:
            if col.startswith('EMA_'):
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)), row=1, col=1)
        
        # Add MACD if present
        if has_macd:
            macd_row = 2 if not has_cmf else 2
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=macd_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD_Signal'), row=macd_row, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD_Hist'), row=macd_row, col=1)
        
        # Add CMF if present
        if has_cmf:
            cmf_col = [col for col in df.columns if col.startswith('CMF_')][0]
            fig.add_trace(go.Scatter(x=df.index, y=df[cmf_col], name=cmf_col, 
                                   line=dict(color='orange', width=2)), row=cmf_row, col=1)
            # Add zero line for CMF
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=cmf_row, col=1)
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
        fmt: str = typer.Option("parquet", help="Export format: parquet or csv"),
        no_plot: bool = typer.Option(False, "--no-plot", help="Skip plot display"),
        save_plot: bool = typer.Option(False, "--save-plot", help="Save interactive plot as HTML file"),
        resample_only: bool = typer.Option(False, "--resample-only", help="Only resample data, skip indicators and plotting")
    ):
        """Process NinjaTrader CSV export."""
        print("Loading:", input_file)
        
        # Detect file format and load accordingly
        file_format, detected_timeframe = detect_file_format(input_file)
        if file_format == 'nt8':
            df = load_ninjatrader_csv(input_file, tz=tz)
        elif file_format == 'resampled':
            df = load_resampled_csv(input_file, tz=tz)
            # If resampled file and default resample, suggest better timeframe
            if resample == "1min" and detected_timeframe:
                print(f"Warning: Input file appears to be {detected_timeframe} data. Consider using --resample with a larger timeframe (e.g., 5min, 15min, 1H)")
        elif file_format == 'parquet':
            df = load_parquet_file(input_file, tz=tz)
            # If Parquet file and default resample, suggest better timeframe
            if resample == "1min" and detected_timeframe:
                print(f"Warning: Input file appears to be {detected_timeframe} data. Consider using --resample with a larger timeframe (e.g., 5min, 15min, 1H)")
        else:
            raise ValueError(f"Unknown file format for {input_file}")
        
        print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
        
        # Skip resampling if input timeframe matches target timeframe
        if detected_timeframe and detected_timeframe.lower() == resample.lower():
            print(f"Input data is already in {resample} timeframe. Skipping resample step.")
            bars = df.copy()
        else:
            bars = resample_bars(df, resample)
            if len(bars) != len(df):
                print(f"Resampled from {len(df)} to {len(bars)} bars ({resample} timeframe)")
        
        if resample_only:
            # Resample-only mode: just save resampled data as CSV
            out_file = generate_resampled_filename(input_file, resample, out)
            export_df(bars, out_file, fmt='csv')
            print(f"Resampled data exported to {out_file}")
        else:
            # Full processing mode: indicators + plotting + export
            bars_ind = add_indicators(bars)
            out_file = os.path.join(out, f"processed_{resample}.{fmt}")
            export_df(bars_ind, out_file, fmt=fmt)
            print("Exported to", out_file)
            # Handle plotting
            if not no_plot or save_plot:
                try:
                    fig = make_candle_figure(bars_ind, title=f"Processed {os.path.basename(input_file)}")
                    
                    # Save plot to HTML file if requested
                    if save_plot:
                        plot_filename = os.path.join(out, f"chart_{resample}.html")
                        fig.write_html(plot_filename)
                        print(f"Interactive chart saved to: {plot_filename}")
                    
                    # Show plot if not in headless environment and not disabled
                    if not no_plot:
                        import sys
                        headless = (
                            'DISPLAY' not in os.environ or 
                            os.environ.get('WSL_DISTRO_NAME') is not None or
                            sys.platform.startswith('linux')
                        )
                        
                        if headless:
                            if not save_plot:
                                print("Headless/WSL environment detected. Use --save-plot to save chart as HTML.")
                        else:
                            fig.show()
                            
                except Exception as e:
                    print(f"Plot handling failed. Error: {e}")
            
        print("Processing complete.")

    typer.run(main)