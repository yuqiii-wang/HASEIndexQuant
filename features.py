import pandas as pd
import numpy as np
import ta

SHORT_MA = 15
LONG_MA = 30

roc_windows_to_compare = [5, 10, 15, 20, 60]  # e.g., 5-min, 10-min, 20-min, 1-hour ROC

def smooth_roc_for_day(daily_raw_roc_series, smoothing_window, col_name_to_assign):
    if daily_raw_roc_series.dropna().empty or len(daily_raw_roc_series.dropna()) < smoothing_window:
        return pd.Series(np.nan, index=daily_raw_roc_series.index, name=col_name_to_assign)
    # Use EMA for smoothing. adjust=False is common for financial data.
    # min_periods ensures we get values even if shorter than window at start
    return daily_raw_roc_series.ewm(span=smoothing_window, adjust=False, min_periods=max(1,smoothing_window//2+1)).mean().rename(col_name_to_assign)


# --- Configuration ---
PRICE_COL = 'close' # Assuming 'Close' price is used for decisions and P&L

# --- 1. Feature Engineering ---
def add_features(hsi_df: pd.DataFrame, 
                hsfi_df: pd.DataFrame,
                hscat_df: pd.DataFrame,
                sma_short_window=SHORT_MA, sma_long_window=LONG_MA, rsi_window=14, roc_window=10) -> pd.DataFrame:
    
    try:
        if not isinstance(hsi_df.index, pd.DatetimeIndex):
            hsi_df.index = pd.to_datetime(hsi_df.index)
        if not isinstance(hsfi_df.index, pd.DatetimeIndex):
            hsfi_df.index = pd.to_datetime(hsfi_df.index)
        if not isinstance(hscat_df.index, pd.DatetimeIndex):
            hscat_df.index = pd.to_datetime(hscat_df.index)
    except Exception as e:
        raise ValueError(f"One or more DataFrame indices could not be converted to DatetimeIndex: {e}")
    
    common_index = hsi_df.index.intersection(hsfi_df.index).intersection(hscat_df.index)
    
    df_feat = hsi_df.loc[common_index].copy()
    hsfi_aligned = hsfi_df.loc[common_index].copy()
    hscat_aligned = hscat_df.loc[common_index].copy()
    
    # Ensure datetime index
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        try:
            df_feat.index = pd.to_datetime(df_feat.index)
        except:
            raise ValueError("Index must be convertible to DatetimeIndex.")

    # Price features (assuming 'Open', 'High', 'Low', 'Close', 'Volume' exist)
    for window in roc_windows_to_compare: # Example ROC windows
        df_feat.loc[:, f'hsi_roc_{window}'] = df_feat.loc[:, PRICE_COL].pct_change(periods=window).fillna(0) * 100

    # Moving Averages
    for window in roc_windows_to_compare:
        df_feat.loc[:, f'ma_{window}'] = ta.trend.SMAIndicator(df_feat.loc[:, PRICE_COL], window=window).sma_indicator().bfill()
        df_feat.loc[:, f'price_div_ma_{window}'] = df_feat.loc[:, PRICE_COL] / (df_feat.loc[:, f'ma_{window}'] + 1e-9) # Add epsilon for stability
    
    if 'ma_10' in df_feat and 'ma_50' in df_feat:
        df_feat.loc[:, 'ma_10_div_ma_50'] = df_feat.loc[:, 'ma_10'] / (df_feat.loc[:, 'ma_50'] + 1e-9)

    # Volatility
    df_feat.loc[:, 'hsi_volatility_20'] = ta.volatility.BollingerBands(df_feat.loc[:, PRICE_COL], window=20, window_dev=2).bollinger_wband().bfill()
    df_feat.loc[:, 'hsi_atr_14'] = ta.volatility.AverageTrueRange(df_feat.loc[:, 'high'], df_feat.loc[:, 'low'], df_feat.loc[:, PRICE_COL], window=14).average_true_range().bfill()

    # Momentum
    df_feat.loc[:, 'hsi_rsi_14'] = ta.momentum.RSIIndicator(df_feat.loc[:, PRICE_COL], window=14).rsi().fillna(50) # Fill NaN with neutral 50
    stoch = ta.momentum.StochasticOscillator(df_feat.loc[:, 'high'], df_feat.loc[:, 'low'], df_feat.loc[:, PRICE_COL], window=14, smooth_window=3)
    df_feat.loc[:, 'stoch_k'] = stoch.stoch().fillna(50)
    df_feat.loc[:, 'stoch_d'] = stoch.stoch_signal().fillna(50)
    
    macd = ta.trend.MACD(df_feat.loc[:, PRICE_COL])
    df_feat.loc[:, 'macd'] = macd.macd().fillna(0)
    df_feat.loc[:, 'macd_signal'] = macd.macd_signal().fillna(0)
    df_feat.loc[:, 'macd_hist'] = macd.macd_diff().fillna(0)
    
    # Calculate time since market open (assuming market opens at 9:30 for this example)
    # This needs to be adjusted based on actual market hours in your data
    market_open_time = pd.to_datetime('09:45:00').time()
    df_feat.loc[:, 'time_since_open_min'] = df_feat.index.to_series().apply(
        lambda dt: (dt.hour * 60 + dt.minute) - (market_open_time.hour * 60 + market_open_time.minute)
        if dt.time() >= market_open_time else -1 # Or some other handling for pre-market
    )
    df_feat.loc[df_feat.loc[:, 'time_since_open_min'] < 0, 'time_since_open_min'] = 0 # Cap at 0 for simplicity


    df_feat['hsi_hsfi_spread'] = df_feat[PRICE_COL] - hsfi_aligned[PRICE_COL]
    df_feat['hsi_hsfi_ratio'] = df_feat[PRICE_COL] / (hsfi_aligned[PRICE_COL] + 1e-9)
    df_feat['hsi_hsfi_spread_norm'] = df_feat['hsi_hsfi_spread'] / (df_feat['hsi_atr_14'] + 1e-9)
    df_feat['hsfi_roc_10'] = ta.momentum.ROCIndicator(hsfi_aligned[PRICE_COL], window=10).roc().fillna(0)
    df_feat['hsfi_rsi_14'] = ta.momentum.RSIIndicator(hsfi_aligned[PRICE_COL], window=14).rsi().fillna(50)
    df_feat['hsfi_roc_10_lag1'] = df_feat['hsfi_roc_10'].shift(1).fillna(0)
    df_feat['hsfi_close_lag1'] = hsfi_aligned[PRICE_COL].shift(1) # Lagged price itself

    df_feat['hsi_hscat_spread'] = df_feat[PRICE_COL] - hscat_aligned[PRICE_COL]
    df_feat['hsi_hscat_ratio'] = df_feat[PRICE_COL] / (hscat_aligned[PRICE_COL] + 1e-9)
    df_feat['hsi_hscat_spread_norm'] = df_feat['hsi_hscat_spread'] / (df_feat['hsi_atr_14'] + 1e-9)
    df_feat['hscat_roc_10'] = ta.momentum.ROCIndicator(hscat_aligned[PRICE_COL], window=10).roc().fillna(0)
    df_feat['hscat_rsi_14'] = ta.momentum.RSIIndicator(hscat_aligned[PRICE_COL], window=14).rsi().fillna(50)
    df_feat['rsi_diff_hsi_hscat'] = df_feat['hsi_rsi_14'] - df_feat['hscat_rsi_14']
    df_feat['roc_diff_hsi_hscat_10'] = df_feat['hsi_roc_10'] - df_feat['hscat_roc_10']

    # Daily High/Low context (be careful with lookahead bias if not done correctly)
    # This typically requires grouping by day and then expanding.
    # For simplicity, we'll use expanding max/min for each day.
    # This will be done more carefully in the target generation and simulation.

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.ffill().bfill().fillna(0) # Robust fillna

    return df_feat
