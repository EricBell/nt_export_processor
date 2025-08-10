#!/usr/bin/env python3
"""
resample_1m_to_3m.py

Command-line tool (Typer) to load a NinjaTrader 1-minute export in the format:
  date time;open;hi;low;close;volume
(e.g. 20250612 040100;6062.5;6062.5;6062.25;6062.5;14)

It resamples to 3-minute OHLCV bars and writes output to CSV or Parquet.

Usage examples:
  python resample_1m_to_3m.py --input data/mes_1m_noheader.csv --output output/mes_3m.parquet --tz America/New_York
  python resample_1m_to_3m.py -i data/mes_1m.csv -o output/mes_3m.csv --out-format csv --datetime-format "%Y%m%d %H%M%S"

Dependencies:
  pip install typer[all] pandas pyarrow

Author: MARI (adapted for EricTheRed)
"""
from __future__ import annotations
import sys
from typing import Optional
import pathlib
import pandas as pd
import typer
from datetime import datetime

app = typer.Typer(add_completion=False)

DEFAULT_DT_FORMAT = "%Y%m%d %H%M%S"  # matches your sample: 20250612 040100

def load_no_header_semicolon(path: pathlib.Path,
                             datetime_format: Optional[str] = None,
                             tz: Optional[str] = None,
                             cols: list[str] | None = None) -> pd.DataFrame:
    """
    Load semicolon-delimited file with no header and columns in order:
      date time;open;hi;low;close;volume

    Returns DataFrame indexed by DateTime with columns Open,High,Low,Close,Volume
    """
    if cols is None:
        cols = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.read_csv(path, sep=';', header=None, names=cols, dtype=str)
    # Parse datetime
    if datetime_format:
        dt = pd.to_datetime(df['DateTime'], format=datetime_format, errors='coerce')
    else:
        dt = pd.to_datetime(df['DateTime'], errors='coerce', infer_datetime_format=True)
    if dt.isnull().any():
        bad = df[dt.isnull()].head(5)
        raise ValueError(f"Found unparsable datetime values (first few shown):\n{bad[['DateTime']].to_string(index=False)}\n"
                         f"Try specifying --datetime-format to match your file (default {DEFAULT_DT_FORMAT})")
    # timezone handling: localize naive datetimes to tz if provided
    if tz:
        # localize naive datetimes to tz
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(tz)
        else:
            dt = dt.dt.tz_convert(tz)
    df.index = dt
    df.index.name = "DateTime"
    # numeric conversion
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df.columns:
            # remove thousand separators if any
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df = df.sort_index()
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def resample_ohlcv(df: pd.DataFrame,
                   rule: str = "3T",
                   drop_empty: bool = True,
                   fill_method: Optional[str] = None) -> pd.DataFrame:
    """
    Resample dataframe (with DateTime index) to OHLCV using pandas resample.
    rule: pandas offset alias (3T = 3 minutes)
    drop_empty: drop bars where OHLC are NaN
    fill_method: None or one of 'ffill' (fill missing price), 'bfill'
    """
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    bars = df.resample(rule).agg(agg)
    if fill_method:
        # avoid changing volume; only fill price columns, then keep volume as-is
        price_cols = ['Open', 'High', 'Low', 'Close']
        bars[price_cols] = bars[price_cols].fillna(method=fill_method)
    if drop_empty:
        bars = bars.dropna(subset=['Open', 'High', 'Low', 'Close'])
    return bars

@app.command()
def main(
    input: pathlib.Path = typer.Option(..., help="Path to input 1-minute file (semicolon-delimited, no header)"),
    output: pathlib.Path = typer.Option(..., help="Path to output file (CSV or Parquet). Use .csv or .parquet extension"),
    datetime_format: Optional[str] = typer.Option(None, help=f"Datetime format strptime string (default {DEFAULT_DT_FORMAT})"),
    tz: Optional[str] = typer.Option("America/New_York", help="Timezone to localize naive datetimes (e.g., America/New_York). Use empty string to keep naive"),
    resample_rule: str = typer.Option("3T", help="Pandas resample rule, default '3T' for 3-minute"),
    drop_empty: bool = typer.Option(True, help="Drop empty/partial bars after resampling"),
    fill_method: Optional[str] = typer.Option(None, help="If set, fill empty price bars with 'ffill' or 'bfill' before keeping bars"),
    out_format: Optional[str] = typer.Option(None, help="Output format override: 'csv' or 'parquet' (auto by extension if omitted)"),
    preview: bool = typer.Option(False, help="Print a short preview of the resampled output"),
):
    """
    CLI entry: resample 1-minute NinjaTrader export (no header) to 3-minute OHLCV.
    """
    try:
        if not input.exists():
            typer.echo(f"Input file not found: {input}", err=True)
            raise typer.Exit(code=1)

        dt_fmt = datetime_format if datetime_format else DEFAULT_DT_FORMAT

        # load
        df1 = load_no_header_semicolon(input, datetime_format=dt_fmt, tz=tz if tz else None)

        typer.echo(f"Loaded rows: {len(df1)}; start: {df1.index.min()} end: {df1.index.max()}")

        # resample
        bars = resample_ohlcv(df1, rule=resample_rule, drop_empty=drop_empty, fill_method=fill_method)

        typer.echo(f"Resampled to {resample_rule} bars: {len(bars)}; start: {bars.index.min()} end: {bars.index.max()}")

        # determine output format
        if out_format:
            fmt = out_format.lower()
        else:
            suffix = output.suffix.lower()
            if suffix in ['.csv']:
                fmt = 'csv'
            elif suffix in ['.parquet', '.parq', '.pq']:
                fmt = 'parquet'
            else:
                # default to parquet
                fmt = 'parquet'
        # save
        output.parent.mkdir(parents=True, exist_ok=True)
        if fmt == 'csv':
            bars.to_csv(output, index=True)
        else:
            bars.to_parquet(output, index=True)
        typer.echo(f"Wrote {len(bars)} bars to {output} as {fmt.upper()}")

        if preview:
            typer.echo("\nPreview (last 10 bars):")
            typer.echo(bars.tail(10).to_string())

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

if __name__ == "__main__":
    import sys
    # If user ran without args, show help text
    if len(sys.argv) == 1:
        try:
            from typer.main import get_command
            import click
            cmd = get_command(app)
            click.echo(cmd.get_help(click.Context(cmd)))
        except Exception as e:
            # Fallback: run app normally (will show Click/Typer help on --help)
            app()
    else:
        app()