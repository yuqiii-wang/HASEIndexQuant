import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import find_peaks
import ta

from utils import load_data # Technical Analysis library
from features import PRICE_COL, add_features

sell_units_map = [40, 30, 20, 5, 3, 2]

# --- 2. Target Variable Definition & Ideal Cash Calculation ---
def generate_target_variable_and_ideal_cash(df_full_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df_full_features.copy()
    df['ideal_units_to_sell'] = 0
    daily_target_cash = {} # Store target cash for each day

    for date, day_data in df.groupby(df.index.date):
        if day_data.empty:
            continue

        # Find local maxima in price for the day
        # distance parameter helps avoid too many small peaks close together
        # prominence might be better, but distance is simpler here
        peaks_indices, properties = find_peaks(day_data[PRICE_COL], distance=5, prominence=0.001 * day_data[PRICE_COL].mean()) 
                                            # Prominence relative to mean price to make it adaptive
        
        if len(peaks_indices) == 0:
            daily_target_cash[date] = 0
            continue

        peak_info = []
        for i in peaks_indices:
            idx_loc = day_data.index[i]
            peak_info.append({
                'time': idx_loc,
                'price': day_data.loc[idx_loc, PRICE_COL],
                'hour': idx_loc.hour
            })
        
        # Sort by price descending to pick highest peaks first
        peak_info_sorted_by_price = sorted(peak_info, key=lambda x: x['price'], reverse=True)
        
        selected_ideal_sells = []
        sold_hours_ideal = set()
        day_target_cash_value = 0
        
        for peak in peak_info_sorted_by_price:
            if len(selected_ideal_sells) < len(sell_units_map) and peak['hour'] not in sold_hours_ideal:
                units_to_assign = sell_units_map[len(selected_ideal_sells)]
                selected_ideal_sells.append({**peak, 'units': units_to_assign})
                sold_hours_ideal.add(peak['hour'])
                day_target_cash_value += peak['price'] * units_to_assign
        
        daily_target_cash[date] = day_target_cash_value
        
        # Assign to the main dataframe
        for sell_event in selected_ideal_sells:
            df.loc[sell_event['time'], 'ideal_units_to_sell'] = sell_event['units']
            
    df_target_cash = pd.Series(daily_target_cash, name='target_cash').sort_index()
    return df, df_target_cash

# --- 3. Model Training and 4. Custom Evaluation ---
def train_evaluate_gbdt_model(df_with_targets:pd.DataFrame, 
                              df_target_cash_daily:pd.Series,
                              train_dates:pd.DatetimeIndex, 
                              validation_dates:pd.DatetimeIndex,
                              feature_cols: list) -> GradientBoostingRegressor:
    
    df_train = df_with_targets[df_with_targets.index.normalize().isin(train_dates.normalize())]
    df_validation = df_with_targets[df_with_targets.index.normalize().isin(validation_dates.normalize())]

    X_train = df_train[feature_cols]
    y_train = df_train['ideal_units_to_sell']
    
    X_validation = df_validation[feature_cols]
    y_validation_ideal_units = df_validation['ideal_units_to_sell'] # For GBDT's own metrics

    print(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
    print(f"Validation data shape: {X_validation.shape}, Target shape: {y_validation_ideal_units.shape}")

    if X_train.empty or y_train.empty:
        print("Training data is empty. Skipping training.")
        return None, {}

    # --- Train GBDT Model ---
    gbdt_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, 
                                           max_depth=8, random_state=42,
                                           subsample=0.8) # Added subsample
    gbdt_model.fit(X_train, y_train)

    # --- GBDT's own performance metrics ---
    y_pred_validation_raw_gbdt = gbdt_model.predict(X_validation)
    # The GBDT predicts "ideal units". We can round/clip for interpretation if needed
    # y_pred_validation_units_gbdt = np.round(y_pred_validation_raw_gbdt).clip(0, 4).astype(int)
    
    mse = mean_squared_error(y_validation_ideal_units, y_pred_validation_raw_gbdt)
    r2 = r2_score(y_validation_ideal_units, y_pred_validation_raw_gbdt)
    print(f"\nGBDT Model Performance on predicting 'ideal_units_to_sell':")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  R-squared (R2): {r2:.4f}")

    # --- Custom Evaluation: Simulate Trading Strategy based on Model Predictions ---
    # The model's raw output is a "sell desirability score".
    df_validation_sim = df_validation.copy()
    df_validation_sim['model_score'] = y_pred_validation_raw_gbdt
    
    daily_actual_cash_model = {}
    daily_model_sells_info = {} # For debugging/inspection
    
    for val_date_dt, day_data_sim in df_validation_sim.groupby(df_validation_sim.index.date):
        val_date = pd.Timestamp(val_date_dt) # Ensure it's a Timestamp for dict key consistency
        if day_data_sim.empty:
            daily_actual_cash_model[val_date] = 0
            daily_model_sells_info[val_date] = []
            continue

        # Find local maxima of the *model's scores* for the day
        # We need to be careful with peaks at the very start/end if using simple comparison
        # `find_peaks` is more robust. Use a small distance to allow frequent signals if model is spiky.
        # Prominence for scores might be absolute (e.g. score > 0.5) or relative
        # Let's assume higher score is better. A score of 0 means "don't sell".
        # So, we only consider scores > 0 (or some threshold like 0.25 if model rarely predicts exact 0s)
        
        potential_peak_indices, _ = find_peaks(day_data_sim['model_score'], 
                                               height=0.1, # Minimum score to be considered a peak
                                               distance=5) # Min separation between score peaks
        
        if len(potential_peak_indices) == 0:
            daily_actual_cash_model[val_date] = 0
            daily_model_sells_info[val_date] = []
            continue

        model_peak_info = []
        for i in potential_peak_indices:
            idx_loc = day_data_sim.index[i]
            model_peak_info.append({
                'time': idx_loc,
                'price_at_signal': day_data_sim.loc[idx_loc, PRICE_COL],
                'model_score': day_data_sim.loc[idx_loc, 'model_score'],
                'hour': idx_loc.hour
            })
        
        # Sort by model_score descending to pick signals the model rated highest
        model_peak_info_sorted_by_score = sorted(model_peak_info, key=lambda x: x['model_score'], reverse=True)
        
        selected_model_sells = []
        sold_hours_model = set()
        day_actual_cash_value = 0
        
        for peak_candidate in model_peak_info_sorted_by_score:
            if len(selected_model_sells) < len(sell_units_map) and peak_candidate['hour'] not in sold_hours_model:
                units_to_sell_model = sell_units_map[len(selected_model_sells)]
                
                # Record the sell
                sell_details = {
                    'time': peak_candidate['time'],
                    'price': peak_candidate['price_at_signal'], # Sell at price when signal occurred
                    'units': units_to_sell_model,
                    'hour': peak_candidate['hour'],
                    'model_score': peak_candidate['model_score']
                }
                selected_model_sells.append(sell_details)
                
                sold_hours_model.add(peak_candidate['hour'])
                day_actual_cash_value += peak_candidate['price_at_signal'] * units_to_sell_model
        
        daily_actual_cash_model[val_date] = day_actual_cash_value
        daily_model_sells_info[val_date] = selected_model_sells

    df_actual_cash_model = pd.Series(daily_actual_cash_model, name='actual_cash_model').sort_index()
    
    df_target_cash_validation = df_target_cash_daily[df_target_cash_daily.index.isin(validation_dates.date)]

    comparison_df = pd.DataFrame({
        'target_cash': df_target_cash_validation,
        'actual_cash_model': df_actual_cash_model
    }).dropna() # Ensure we only compare days present in both

    if not comparison_df.empty:
        comparison_df['loss (target - actual)'] = comparison_df['target_cash'] - comparison_df['actual_cash_model']
        
        avg_loss = comparison_df['loss (target - actual)'].mean()
        total_target_cash = comparison_df['target_cash'].sum()
        total_actual_cash = comparison_df['actual_cash_model'].sum()
        
        print(f"\nCustom Trading Strategy Performance (Validation Set):")
        print(f"  Total Target Cash (Ideal): {total_target_cash:.2f}")
        print(f"  Total Actual Cash (Model): {total_actual_cash:.2f}")
        print(f"  Average Daily Loss (Target - Actual): {avg_loss:.2f}")
        if total_target_cash > 0:
             print(f"  Model Capture Rate: {total_actual_cash / total_target_cash * 100:.2f}%")
        else:
             print(f"  Model Capture Rate: N/A (Total Target Cash is zero)")

        # Display some daily details for inspection
        print("\nSample Daily Comparison:")
        print(comparison_df.head())
        
        # Inspection of model's sell decisions for a sample day
        if daily_model_sells_info:
            sample_day_key = list(daily_model_sells_info.keys())[0]
            print(f"\nModel sells for sample day {sample_day_key.strftime('%Y-%m-%d')}:")
            for s_info in daily_model_sells_info[sample_day_key]:
                print(f"  Time: {s_info['time']}, Price: {s_info['price']:.2f}, Units: {s_info['units']}, Score: {s_info['model_score']:.2f}")

    else:
        print("\nCustom Trading Strategy Performance (Validation Set):")
        print("  No common dates for comparison or no trades made/targets available.")

    results = {
        'gbdt_mse': mse,
        'gbdt_r2': r2,
        'avg_custom_loss': avg_loss if not comparison_df.empty else np.nan,
        'total_target_cash': total_target_cash if not comparison_df.empty else 0,
        'total_actual_cash': total_actual_cash if not comparison_df.empty else 0,
        'comparison_df': comparison_df,
        'model_sells_info': daily_model_sells_info
    }
    
    return gbdt_model

def prepare_then_get_gbdt_model(df_full_features:pd.DataFrame, 
                              train_dates_list:list[pd.DatetimeIndex],
                              validation_dates_list:list[pd.DatetimeIndex]) -> GradientBoostingRegressor:
    
    train_dates_idx = pd.to_datetime(train_dates_list)
    validation_dates_idx = pd.to_datetime(validation_dates_list)

    print(f"Feature columns: {df_full_features.columns}")


    # 2. Generate target variable and ideal cash
    df_with_targets, df_target_cash_daily = generate_target_variable_and_ideal_cash(df_full_features)
    print("\nData with targets ('ideal_units_to_sell'):")
    print(df_with_targets[df_with_targets['ideal_units_to_sell'] > 0].head())
    print("\nDaily target cash:")
    print(df_target_cash_daily.head())
    
    # Define feature columns (exclude target and original price/OHLC if not transformed)
    # All ROCs, MAs, technical indicators, time features etc.
    feature_columns = [col for col in df_with_targets.columns if col not in 
                       ['ideal_units_to_sell', 'target_cash']] # Targets
    
    # Filter out any remaining NaN/inf just in case, particularly in feature_columns if new ones were added
    original_len = len(df_with_targets)
    df_with_targets = df_with_targets.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
    if len(df_with_targets) < original_len:
        print(f"Dropped {original_len - len(df_with_targets)} rows due to NaN/inf in features.")


    print(f"\nTraining on {len(train_dates_idx)} days, Validating on {len(validation_dates_idx)} days.")

    # 3 & 4. Train and Evaluate
    model = None
    if not df_with_targets.empty:
        model = train_evaluate_gbdt_model(df_with_targets, 
                                        df_target_cash_daily,
                                        train_dates_idx, 
                                        validation_dates_idx,
                                        feature_columns)
        if model:
            print("\nTraining and evaluation complete.")
            # You can inspect results dict here
            # print(results['comparison_df'])
        else:
            print("\nModel training failed or was skipped.")
    else:
        print("Not enough data or dates to proceed with training and validation.")

    return model

# --- Example Usage ---
if __name__ == '__main__':
    hsi_df = load_hsi_data()
