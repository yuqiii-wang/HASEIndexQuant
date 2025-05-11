import pandas as pd
import numpy as np
import datetime

SHORT_MA = 15
LONG_MA = 30
GBDT_PREDICTION_HORIZON_MINUTES = 5 # Predict price 5 minutes ahead
GBDT_TARGET_COLUMN = f'target_price_ratio_h{GBDT_PREDICTION_HORIZON_MINUTES}'
GBDT_FEATURE_COLUMNS = ['SMA_short', 'SMA_long', 'RSI', 'ROC', 
                        'Volatility', 'Momentum', 'Upper_Bollinger', 'Lower_Bollinger']

def add_features(df, sma_short_window=SHORT_MA, sma_long_window=LONG_MA, rsi_window=14, roc_window=10):
    df_feat = df.copy()

    if 'close' not in df_feat.columns or df_feat['close'].dtype == 'object':
        df_feat.loc[:, 'close'] = pd.to_numeric(df_feat['close'], errors='coerce')

    if df_feat['close'].isnull().all():
        print("Warning in add_features: 'close' column is all NaNs. Feature values will be based on fillna logic.")

    df_feat.loc[:, 'SMA_short'] = df_feat['close'].rolling(window=sma_short_window, min_periods=1).mean()
    df_feat.loc[:, 'SMA_long'] = df_feat['close'].rolling(window=sma_long_window, min_periods=1).mean()

    # Momentum
    df_feat['Momentum'] = df_feat['close'] - df_feat['close'].shift(4)

    # Volume-Weighted Features (if volume data exists)
    if 'volume' in df_feat.columns:
        df_feat['VWAP'] = (df_feat['volume'] * df_feat['close']).cumsum() / df_feat['volume'].cumsum()

    delta = df_feat['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
    
    rs = gain / loss.replace(0, 0.000001) # Avoid division by zero
    df_feat.loc[:, 'RSI'] = 100 - (100 / (1 + rs))

    df_feat.loc[:, 'ROC'] = df_feat['close'].pct_change(periods=roc_window) * 100 
    df_feat.loc[:, 'Volatility'] = df_feat['close'].pct_change().rolling(window=sma_long_window, min_periods=1).std() * np.sqrt(sma_long_window)
    df_feat.loc[:, 'Price_Pred_Simple'] = df_feat['close'].shift(1) + (df_feat['close'].shift(1) - df_feat['close'].shift(2))
    df_feat.loc[:, GBDT_TARGET_COLUMN] = df_feat['close'].shift(-GBDT_PREDICTION_HORIZON_MINUTES) / df_feat['close']
    
    # Bollinger Bands
    df_feat['Upper_Bollinger'] = df_feat['SMA_long'] + 2*df_feat['Volatility']
    df_feat['Lower_Bollinger'] = df_feat['SMA_long'] - 2*df_feat['Volatility']

    cols_to_process = GBDT_FEATURE_COLUMNS + ['Price_Pred_Simple', GBDT_TARGET_COLUMN]
    for col in cols_to_process:
        if col in df_feat.columns: # Check if column was actually created
            df_feat.loc[:, col] = pd.to_numeric(df_feat[col], errors='coerce')

    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True) # inplace=True on the whole df_feat is fine here
    
    if 'RSI' in df_feat.columns:
        df_feat.loc[:, 'RSI'] = df_feat['RSI'].fillna(50) # Specific fill for RSI

    df_feat = df_feat.bfill().ffill() # df.bfill() preferred over fillna(method='bfill')
    return df_feat

