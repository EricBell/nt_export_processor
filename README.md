# nt_export_processor

A Python utility to process exported historical data from NinjaTrader 8 (NT8). This script loads NT8 CSV exports, cleans the data, resamples to common intraday bar sizes, computes technical indicators, and provides plotting and export capabilities.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- pandas - Data manipulation and analysis
- numpy - Numerical computing
- matplotlib - Basic plotting
- plotly - Interactive charting
- typer - CLI framework
- pyarrow - Parquet file support

## Usage

### Command Line Interface

```bash
python nt_export_processor.py [INPUT_FILE] [OPTIONS]
```

#### Arguments
- `INPUT_FILE` - Path to NinjaTrader CSV export file (required)

#### Options
- `--out` - Output directory (default: ./output)
- `--tz` - Timezone to localize data (e.g., 'America/New_York')
- `--resample` - Resample timeframe (default: 1min). Examples: '1min', '5min', '1H', '1D'
- `--fmt` - Export format: 'parquet' or 'csv' (default: parquet)
- `--no-plot` - Skip interactive plot display
- `--resample-only` - Only resample data, skip indicators and plotting (outputs CSV)

#### Examples

Basic usage:
```bash
python nt_export_processor.py data/MES-1-day-sample.txt
```

With custom options:
```bash
python nt_export_processor.py data/MES-1-day-sample.txt --out ./results --resample 5min --tz America/New_York --fmt csv
```

Skip plotting (useful for headless environments):
```bash
python nt_export_processor.py data/MES-1-day-sample.txt --no-plot
```

Resample-only mode (for preprocessing):
```bash
python nt_export_processor.py data/MES-1-day-sample.txt --resample 3min --resample-only
```

Process previously resampled file:
```bash
python nt_export_processor.py output/MES-1-day-sample_3min.csv --resample 15min --resample-only
```

### Input Data Format

The script accepts two input formats:

**Raw NinjaTrader CSV exports:**
```
YYYYMMDD HHMMSS;open;high;low;close;volume
20250612 040100;6062.5;6062.5;6062.25;6062.5;14
```

**Previously resampled CSV files:**
```
DateTime,Open,High,Low,Close,Volume
2025-06-12 04:01:00,6062.5,6064.25,6062.25,6063.75,29
```

The script automatically detects the format and loads accordingly.

### Features

- **Flexible Data Loading**: 
  - Auto-detects NT8 CSV exports vs previously resampled files
  - Handles both raw semicolon-delimited and standard CSV formats
- **Data Cleaning**: Handles missing values and sorts by timestamp
- **Resampling**: Converts tick/bar data to OHLCV bars at specified intervals
- **Resample-Only Mode**: Efficient preprocessing for large datasets
  - Outputs resampled data as `{input_name}_{timeframe}.csv`
  - Can chain resample operations (e.g., 1min → 3min → 15min)
  - Skips indicators and plotting for faster processing
- **Technical Indicators**:
  - Simple Moving Average (SMA): 10, 20, 50 periods
  - Exponential Moving Average (EMA): 9, 21 periods
  - Relative Strength Index (RSI): 14 periods
  - MACD: 12/26/9 parameters
  - Average True Range (ATR): 14 periods
- **Interactive Plotting**: Candlestick charts with indicators using Plotly
- **Export Options**: Save processed data as Parquet or CSV files

### Python API

You can also use the functions directly in Python:

```python
from nt_export_processor import (
    detect_file_format, load_ninjatrader_csv, load_resampled_csv, 
    resample_bars, add_indicators, generate_resampled_filename
)

# Auto-detect and load data
file_format = detect_file_format('data/export.csv')
if file_format == 'nt8':
    df = load_ninjatrader_csv('data/export.csv', tz='America/New_York')
elif file_format == 'resampled':
    df = load_resampled_csv('data/export.csv', tz='America/New_York')

# Resample to 5-minute bars
bars = resample_bars(df, '5min')

# For full processing: add technical indicators
bars_with_indicators = add_indicators(bars)

# For resample-only: generate output filename
output_file = generate_resampled_filename('data/export.csv', '5min', './output')
```

## Output

### Full Processing Mode
- OHLCV price data with technical indicators (SMA, EMA, RSI, MACD, ATR)
- Interactive charts (if plotting enabled)
- Exported files in specified format (Parquet or CSV)

### Resample-Only Mode
- Pure OHLCV resampled data (no indicators)
- CSV format with descriptive filename: `{input_name}_{timeframe}.csv`
- Example: `MES-1-day-sample.txt` → `MES-1-day-sample_3min.csv`
- Optimized for preprocessing large datasets

To Do
- Remove SMA 10, 20, 30 and EMA 21