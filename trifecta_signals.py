# trifecta_signals.py
import pandas as pd
import numpy as np

def ema(series: pd.Series, window: int):
    return series.ewm(span=window, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def chaikin_money_flow(df: pd.DataFrame, window: int = 20):
    """
    CMF = sum( ( (close - low) - (high - close) ) / (high - low) * volume ) / sum(volume)
    over 'window'
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    vol = df['Volume'].replace(0, np.nan)
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
    mf_volume = mf_multiplier * vol
    cmf = mf_volume.rolling(window).sum() / vol.rolling(window).sum()
    return cmf.fillna(0)

def accumulation_distribution_line(df: pd.DataFrame):
    high = df['High']
    low = df['Low']
    close = df['Close']
    vol = df['Volume']
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    ad = (mfm * vol).cumsum()
    return ad

def atr(df: pd.DataFrame, window: int = 14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean()

def volume_confirm(df: pd.DataFrame, vol_window: int = 20, vol_multiplier: float = 1.25):
    vol_sma = df['Volume'].rolling(vol_window).mean().fillna(0)
    return df['Volume'] >= (vol_sma * vol_multiplier)

def tl_slope(df: pd.DataFrame, tl_window: int = 9, slope_period: int = 3):
    tl = ema(df['Close'], tl_window)
    slope = tl - tl.shift(slope_period)
    return slope

def generate_trifecta_signals(df: pd.DataFrame,
                              tl_window: int = 9,
                              tl_slope_period: int = 3,
                              macd_params=(12,26,9),
                              money_flow_type: str = 'cmf',  # 'cmf' or 'adl'
                              cmf_window: int = 20,
                              vol_window: int = 20,
                              vol_multiplier: float = 1.25,
                              require_all_in_same_bar: bool = True):
    """
    Returns DataFrame with columns:
      - TL (9ema)
      - TL_slope
      - MACD, MACD_signal, MACD_hist
      - MoneyFlow (CMF or ADL)
      - VolumeConfirm (bool)
      - trifecta_signal: 1 for bull, -1 for bear, 0 for none
      - trifecta_reason: short text for debugging
    """
    df = df.copy()
    df['TL'] = ema(df['Close'], tl_window)
    df['TL_slope'] = df['TL'] - df['TL'].shift(tl_slope_period)

    macd_line, macd_signal, macd_hist = macd(df['Close'], *macd_params)
    df['MACD'] = macd_line
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist

    if money_flow_type.lower() == 'cmf':
        df['MoneyFlow'] = chaikin_money_flow(df, cmf_window)
    else:
        df['MoneyFlow'] = accumulation_distribution_line(df)

    df['VolumeConfirm'] = volume_confirm(df, vol_window, vol_multiplier)

    # Conditions
    df['TL_bull'] = df['TL_slope'] > 0
    df['TL_bear'] = df['TL_slope'] < 0
    df['MACD_bull'] = df['MACD_hist'] > 0
    df['MACD_bear'] = df['MACD_hist'] < 0
    if money_flow_type.lower() == 'cmf':
        df['MF_bull'] = df['MoneyFlow'] > 0
        df['MF_bear'] = df['MoneyFlow'] < 0
    else:  # ADL slope
        df['MF_bull'] = df['MoneyFlow'] - df['MoneyFlow'].shift(1) > 0
        df['MF_bear'] = df['MoneyFlow'] - df['MoneyFlow'].shift(1) < 0

    # Compose final signal
    def compute_row(row):
        bull = row['TL_bull'] and row['MACD_bull'] and row['MF_bull'] and row['VolumeConfirm']
        bear = row['TL_bear'] and row['MACD_bear'] and row['MF_bear'] and row['VolumeConfirm']
        if bull:
            return 1, "Bull: TL+,MACD+,MF+,Vol+"
        if bear:
            return -1, "Bear: TL-,MACD-,MF-,Vol+"
        return 0, ""
    computed = df.apply(compute_row, axis=1, result_type='expand')
    df['trifecta_signal'] = computed[0]
    df['trifecta_reason'] = computed[1]

    return df