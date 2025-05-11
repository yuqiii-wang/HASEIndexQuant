import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


GBDT_PREDICTION_HORIZON_MINUTES = 5 # Predict price 5 minutes ahead
GBDT_TARGET_COLUMN = f'target_price_ratio_h{GBDT_PREDICTION_HORIZON_MINUTES}'
GBDT_FEATURE_COLUMNS = ['SMA_short', 'SMA_long', 'RSI', 'ROC', 
                        'Volatility', 'Momentum', 'Upper_Bollinger', 'Lower_Bollinger']

# --- New Helper Function for GBDT Target ---
def add_max_future_price_target(df, lookahead_periods, price_col='close', new_target_col_name_template='max_price_next_{}m'):
    """
    Calculates the maximum price in the next 'lookahead_periods' minutes (exclusive of current minute).
    The target for current time t is max(price[t+1], price[t+2], ..., price[t+lookahead_periods]).

    Args:
        df (pd.DataFrame): DataFrame with time series data, must have 'price_col'.
        lookahead_periods (int): Number of future periods (minutes) to look into.
        price_col (str): Name of the column containing the price.
        new_target_col_name_template (str): String template for the new target column name.

    Returns:
        pd.DataFrame: DataFrame with the new target column added.
        str: The name of the newly added target column.
    """
    if lookahead_periods <= 0:
        raise ValueError("lookahead_periods must be positive.")
    
    actual_new_target_col_name = new_target_col_name_template.format(lookahead_periods)
    
    # Calculate max price in future window [t+1, t+lookahead_periods]
    # .shift(-lookahead_periods) aligns the end of the future window to the current time.
    # .rolling(window=lookahead_periods, min_periods=1).max() then takes the max over that window.
    df[actual_new_target_col_name] = df[price_col].shift(-lookahead_periods) \
                                           .rolling(window=lookahead_periods, min_periods=1) \
                                           .max()
    print(f"Added target column: {actual_new_target_col_name}")
    return df, actual_new_target_col_name

# --- GBDT Model Training and Evaluation ---
def train_evaluate_gbdt_model(df_full_features, train_dates_list, validation_dates_list):
    # This function will now use the globally defined GBDT_TARGET_COLUMN,
    # which should be the new profit-based target (e.g., 'max_price_next_10m')
    # and the globally defined GBDT_FEATURE_COLUMNS.
    print("\n--- Training GBDT Model ---")
    
    if GBDT_TARGET_COLUMN not in df_full_features.columns:
        print(f"Error: Target column '{GBDT_TARGET_COLUMN}' not found in DataFrame. Ensure it's generated and GBDT_TARGET_COLUMN is set correctly.")
        return None
        
    data_for_gbdt = df_full_features.dropna(subset=[GBDT_TARGET_COLUMN] + GBDT_FEATURE_COLUMNS).copy()

    if data_for_gbdt.empty:
        print(f"Error: No data available for GBDT training (target: {GBDT_TARGET_COLUMN}) after NaN drop.")
        return None

    train_dates_ts = pd.to_datetime(train_dates_list)
    validation_dates_ts = pd.to_datetime(validation_dates_list)

    train_mask = data_for_gbdt.index.normalize().isin(train_dates_ts)
    validation_mask = data_for_gbdt.index.normalize().isin(validation_dates_ts)

    X = data_for_gbdt[GBDT_FEATURE_COLUMNS]
    y = data_for_gbdt[GBDT_TARGET_COLUMN]

    X_train, y_train = X[train_mask], y[train_mask]
    X_validation, y_validation = X[validation_mask], y[validation_mask]
    
    if X_train.empty or y_train.empty:
        print(f"Error: Training data for GBDT is empty ({X_train.shape[0]} samples) using target {GBDT_TARGET_COLUMN}. Check date splitting and data availability on train_dates.")
        return None

    print(f"GBDT Training samples (minute-level): {X_train.shape[0]} from {len(train_dates_list)} days for target '{GBDT_TARGET_COLUMN}'")
    print(f"GBDT Validation samples (minute-level): {X_validation.shape[0]} from {len(validation_dates_list)} days")

    gbdt_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    # The model will minimize MSE: (true_max_future_price - predicted_max_future_price)^2
    # This aids in maximizing profit if decisions are based on this prediction.
    gbdt_model.fit(X_train, y_train)
    print("GBDT Model trained.")

    if not X_validation.empty and not y_validation.empty:
        y_pred_validation = gbdt_model.predict(X_validation)
        mse = mean_squared_error(y_validation, y_pred_validation)
        r2 = r2_score(y_validation, y_pred_validation)
        print(f"GBDT Model Evaluation on Validation Days (predicting {GBDT_TARGET_COLUMN}):")
        print(f"  Mean Squared Error (MSE): {mse:.6f}")
        print(f"  R-squared (R2 Score): {r2:.4f}")
    else:
        print("GBDT Model: No validation data for evaluation or validation set was empty.")
    return gbdt_model
