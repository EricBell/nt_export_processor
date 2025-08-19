# NinjaTrader Export Processing Toolkit

A comprehensive Python toolkit for processing and analyzing NinjaTrader 8 (NT8) exported data. This suite includes three specialized tools for data processing, resampling, and trading signal generation.

## Overview

This toolkit consists of three main components:

1. **`nt_export_processor.py`** - Main data processor with technical indicators and charting
2. **`resample_1m_to_3m.py`** - Specialized resampling utility with time gap analysis  
3. **`trifecta_signals.py`** - Trading signal generator using multi-factor analysis

## Installation

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

## 1. Main Processor (`nt_export_processor.py`)

The primary tool for processing NinjaTrader exports with full technical analysis capabilities.

### Usage

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
python nt_export_processor.py data/MES\ 09-25.Last.txt
```

With custom options:
```bash
python nt_export_processor.py "data/MES 09-25.Last.txt" --out ./results --resample 5min --tz America/New_York --fmt csv
```

Skip plotting (useful for headless environments):
```bash
python nt_export_processor.py "data/MES 09-25.Last.txt" --no-plot
```

Resample-only mode (for preprocessing):
```bash
python nt_export_processor.py "data/MES 09-25.Last.txt" --resample 3min --resample-only
```

Full analysis with HTML plot (ideal for WSL/headless):
```bash
python nt_export_processor.py data/processed_3min.parquet --resample 15min --save-plot
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

### Output

#### Full Processing Mode
- OHLCV price data with technical indicators (EMA, MACD, ATR, CMF)
- Interactive charts saved as HTML files (`chart_{timeframe}.html`)
- Exported processed data in specified format (Parquet or CSV)

#### Resample-Only Mode
- Pure OHLCV resampled data (no indicators)
- CSV format with descriptive filename: `{input_name}_{timeframe}.csv`
- Example: `MES 09-25.Last.txt` → `MES-09-25.Last_3min.csv`
- Optimized for preprocessing large datasets

## 2. Resampling Tool (`resample_1m_to_3m.py`)

Specialized command-line tool for efficient data resampling and time gap analysis.

### Usage

```bash
python resample_1m_to_3m.py --input [INPUT_FILE] [OPTIONS]
```

#### Features
- **Resample Feature**: Convert 1-minute data to any timeframe (3min default)
- **Deltas Feature**: Analyze time gaps and frequency in processed data
- **Timezone Support**: Convert between timezones
- **Format Support**: CSV and Parquet input/output

#### Examples

Resample to 3-minute bars:
```bash
python resample_1m_to_3m.py --input "data/MES 09-25.Last.txt" --output output/mes_3m.parquet --tz America/New_York
```

Resample to different timeframe:
```bash
python resample_1m_to_3m.py --input "data/MES 09-25.Last.txt" --output output/mes_5m.csv --resample-rule 5T --out-format csv
```

Analyze time gaps in processed data:
```bash
python resample_1m_to_3m.py --input output/mes_3m.parquet --feature deltas
```

#### Options
- `--input` - Input file path (required)
- `--output` - Output file path (required for resample feature)
- `--feature` - 'resample' (default) or 'deltas'
- `--tz` - Timezone (default: America/New_York)
- `--resample-rule` - Pandas rule like '3T', '5T', '1H' (default: 3T)
- `--out-format` - 'csv' or 'parquet' (auto-detected from extension)
- `--preview` - Show preview of resampled data
- `--datetime-format` - Custom datetime parsing format

## 3. Trading Signals (`trifecta_signals.py`)

Generate trading signals using the "trifecta" strategy that combines multiple technical factors.

### Python API Usage

```python
from trifecta_signals import generate_trifecta_signals
import pandas as pd

# Load your OHLCV data
df = pd.read_parquet('data/processed_3min.parquet')

# Generate trifecta signals
signals_df = generate_trifecta_signals(df)

# View signals
buy_signals = signals_df[signals_df['trifecta_signal'] == 1]
sell_signals = signals_df[signals_df['trifecta_signal'] == -1]

print(f"Generated {len(buy_signals)} buy signals and {len(sell_signals)} sell signals")
```

### Trifecta Strategy Components

The strategy requires all four conditions to align:

1. **Trend Line (EMA 9)**: Price direction confirmation
   - Bull: EMA slope positive over 3 periods
   - Bear: EMA slope negative over 3 periods

2. **MACD Momentum**: Momentum confirmation
   - Bull: MACD histogram > 0
   - Bear: MACD histogram < 0

3. **Money Flow**: Volume-price analysis
   - **CMF Mode**: Chaikin Money Flow > 0 (bull) or < 0 (bear)
   - **ADL Mode**: Accumulation/Distribution Line increasing (bull) or decreasing (bear)

4. **Volume Confirmation**: Significant volume required
   - Current volume > 1.25x 20-period average

### Signal Output

The function returns a DataFrame with additional columns:
- `TL`, `TL_slope` - Trend line and slope
- `MACD`, `MACD_signal`, `MACD_hist` - MACD components  
- `MoneyFlow` - CMF or ADL values
- `VolumeConfirm` - Boolean volume condition
- `trifecta_signal` - 1 (bull), -1 (bear), 0 (no signal)
- `trifecta_reason` - Human-readable signal explanation

### Example Integration

```python
# Complete workflow example
from nt_export_processor import load_ninjatrader_csv, resample_bars, add_indicators
from trifecta_signals import generate_trifecta_signals

# 1. Load and process data
df = load_ninjatrader_csv('data/MES 09-25.Last.txt', tz='America/New_York')
bars_3min = resample_bars(df, '3min')

# 2. Generate signals
signals = generate_trifecta_signals(bars_3min)

# 3. Analyze results
active_signals = signals[signals['trifecta_signal'] != 0]
print(f"Found {len(active_signals)} trifecta signals:")
for idx, row in active_signals.tail(5).iterrows():
    print(f"{idx}: {row['trifecta_reason']}")
```

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

## Sample Workflows

### 1. Full Analysis Pipeline
```bash
# Step 1: Process NT8 export to 3min with indicators and chart
python nt_export_processor.py "data/MES 09-25.Last.txt" --resample 3min --save-plot

# Step 2: Generate trading signals
python -c "
from nt_export_processor import load_parquet_file
from trifecta_signals import generate_trifecta_signals
df = load_parquet_file('output/processed_3min.parquet')
signals = generate_trifecta_signals(df)
print('Signals:', len(signals[signals['trifecta_signal'] != 0]))
"
```

### 2. Multi-Timeframe Analysis
```bash
# Generate multiple timeframes for analysis
python nt_export_processor.py "data/MES 09-25.Last.txt" --resample 1min --resample-only
python nt_export_processor.py "output/MES-09-25.Last_1min.csv" --resample 3min --resample-only  
python nt_export_processor.py "output/MES-09-25.Last_3min.csv" --resample 15min --save-plot
```

### 3. Headless Processing (WSL/Server)
```bash
# Process data without display, save charts as HTML
python nt_export_processor.py "data/MES 09-25.Last.txt" --resample 5min --no-plot --save-plot
# Open output/chart_5min.html in any browser
```

## File Structure

```
nt_export_processor/
├── nt_export_processor.py    # Main processor
├── resample_1m_to_3m.py      # Resampling utility
├── trifecta_signals.py       # Trading signals
├── requirements.txt          # Dependencies
├── data/                     # Input data
│   ├── MES 09-25.Last.txt   # Sample NT8 export
│   └── sample               # Sample data snippet
└── output/                   # Generated outputs
    ├── processed_*.parquet  # Processed data
    ├── chart_*.html        # Interactive charts
    └── *_resampled.csv     # Resampled data
```

## To-Do Items

- Remove SMA 10, 20, 30 and EMA 21 (completed - now uses EMA 9 only)