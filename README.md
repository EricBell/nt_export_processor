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
- `--save-plot` - Save interactive plot as HTML file (works in headless environments)
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

Full analysis with HTML plot (ideal for WSL/headless):
```bash
python nt_export_processor.py data/processed_3min.parquet --resample 15min --save-plot
```

Analysis without resampling (when timeframes match):
```bash
python nt_export_processor.py data/processed_3min.parquet --resample 3min --save-plot
```

### Input Data Format

The script accepts three input formats:

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

**Parquet files (processed data):**
- Binary format with DateTime index and OHLCV columns
- Faster loading and smaller file sizes
- Preserves data types and timezone information

The script automatically detects the format and loads accordingly.

### Features

- **Flexible Data Loading**: 
  - Auto-detects NT8 CSV, resampled CSV, and Parquet formats
  - Handles semicolon-delimited, standard CSV, and binary formats
  - Extracts timeframe from filename for intelligent processing
- **Data Cleaning**: Handles missing values and sorts by timestamp
- **Smart Resampling**: 
  - Automatically skips resampling when input/target timeframes match
  - Converts tick/bar data to OHLCV bars at specified intervals
  - Shows clear feedback about resampling operations
- **Resample-Only Mode**: Efficient preprocessing for large datasets
  - Outputs resampled data as `{input_name}_{timeframe}.csv`
  - Can chain resample operations (e.g., 1min → 3min → 15min)
  - Skips indicators and plotting for faster processing
- **Technical Indicators**:
  - Exponential Moving Average (EMA): 9 periods
  - MACD: 12/26/9 parameters
  - Average True Range (ATR): 14 periods
  - Chaikin Money Flow (CMF): 20 periods
- **Interactive Plotting**: 
  - Candlestick charts with technical indicators using Plotly
  - Multi-panel layout with MACD and CMF subplots
  - Save as HTML files for viewing in any browser
  - Works in headless/WSL environments
- **Export Options**: Save processed data as Parquet or CSV files

### Python API

You can also use the functions directly in Python:

```python
from nt_export_processor import (
    detect_file_format, load_ninjatrader_csv, load_resampled_csv, 
    load_parquet_file, resample_bars, add_indicators, generate_resampled_filename
)

# Auto-detect and load data
file_format, detected_timeframe = detect_file_format('data/processed_3min.parquet')
if file_format == 'nt8':
    df = load_ninjatrader_csv('data/export.csv', tz='America/New_York')
elif file_format == 'resampled':
    df = load_resampled_csv('data/export.csv', tz='America/New_York')
elif file_format == 'parquet':
    df = load_parquet_file('data/processed_3min.parquet', tz='America/New_York')

# Smart resampling (skips if timeframes match)
target_timeframe = '5min'
if detected_timeframe and detected_timeframe.lower() == target_timeframe.lower():
    bars = df.copy()  # Skip resampling
else:
    bars = resample_bars(df, target_timeframe)

# For full processing: add technical indicators
bars_with_indicators = add_indicators(bars)
```

## Output

### Full Processing Mode
- OHLCV price data with technical indicators (EMA, MACD, ATR, CMF)
- Interactive charts saved as HTML files (`chart_{timeframe}.html`)
- Exported processed data in specified format (Parquet or CSV)

### Resample-Only Mode
- Pure OHLCV resampled data (no indicators)
- CSV format with descriptive filename: `{input_name}_{timeframe}.csv`
- Example: `MES-1-day-sample.txt` → `MES-1-day-sample_3min.csv`
- Optimized for preprocessing large datasets

## Interactive Chart Features

The HTML charts generated with `--save-plot` offer:

### Navigation
- **Zoom**: Click and drag to select area
- **Pan**: Shift + click and drag to move around
- **Reset**: Double-click to reset to full view
- **Scroll wheel**: Zoom in/out at cursor

### Toolbar Controls
- 📷 Download as PNG
- 🔍 Zoom tools
- ↔️ Pan mode
- 🏠 Reset view
- ⚙️ Auto-scale

### Interactive Features
- **Hover data**: Move cursor over candlesticks for exact values
- **Toggle indicators**: Click legend items to show/hide lines
- **Multi-panel**: Main chart + MACD and CMF subplots with linked zooming
- **Crosshair**: Precise time/price coordinates

## Technical Indicator Details

### Chaikin Money Flow (CMF)
CMF measures buying and selling pressure by combining price and volume:
- **Positive values**: Buying pressure (accumulation)
- **Negative values**: Selling pressure (distribution)
- **Range**: Typically between -1.0 and +1.0
- **Zero line**: Acts as neutral territory
- **Formula**: Sum of Money Flow Volume / Sum of Volume over 20 periods

CMF is particularly useful for:
- Confirming price trends with volume
- Identifying potential reversals when diverging from price
- Spotting accumulation/distribution phases