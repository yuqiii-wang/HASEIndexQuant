import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # For date formatting on plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf # Import acf and pacf
from scipy.fft import fft, fftfreq
from mplfinance.original_flavor import candlestick_ohlc
from scipy.signal import find_peaks

SHORT_MA = 15
LONG_MA = 30
GBDT_PREDICTION_HORIZON_MINUTES = 5 # Predict price 5 minutes ahead
GBDT_TARGET_COLUMN = f'target_price_ratio_h{GBDT_PREDICTION_HORIZON_MINUTES}'
GBDT_FEATURE_COLUMNS = ['SMA_short', 'SMA_long', 'RSI', 'ROC', 
                        'Volatility', 'Momentum', 'Upper_Bollinger', 'Lower_Bollinger']

# --- 1. Data Loading and Preparation ---
def load_hsi_data(file_path="dataset/HSI_min_20250201-20250430.xlsx"):
    """Loads HSI data from Excel and sets datetime index."""
    hsi_df = pd.read_excel(file_path, index_col="index")
    hsi_df['datetime'] = pd.to_datetime(hsi_df['date'] + ' ' + hsi_df['time'])
    hsi_df = hsi_df.set_index('datetime')

    if 'close' in hsi_df.columns:
        hsi_df['close'] = pd.to_numeric(hsi_df['close'], errors='coerce')
        if hsi_df['close'].isnull().all():
            print("Warning: 'close' column is all NaNs after loading and pd.to_numeric(errors='coerce'). Check Excel file.")
    else:
        print("Warning: 'close' column not found. Feature calculation will likely fail.")
        hsi_df['close'] = np.nan 
    return hsi_df

# --- Plotting function (largely unchanged from your snippet, with minor robusteness enhancements) ---
def plot_detailed_day_simulation(day_data_full_resolution,
                                 all_trades_on_day_dict,
                                 date_obj,
                                 total_units_target_for_day,
                                 price_column='close'):
    
    num_algos = len(all_trades_on_day_dict)
    if num_algos == 0:
        print(f"No trades to plot for {date_obj.strftime('%Y-%m-%d')}.")
        return

    ncols = 2 if num_algos > 1 else 1
    nrows = (num_algos + ncols - 1) // ncols # Calculate number of rows needed
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=(7 * ncols, 5 * nrows), 
                             sharex=True, 
                             squeeze=False) # squeeze=False ensures axes is always 2D array
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    day_start_dt = datetime.datetime.combine(date_obj, datetime.time.min)
    trading_session_start_dt = day_start_dt.replace(hour=9, minute=30) 
    trading_session_end_dt = day_start_dt.replace(hour=16, minute=0)
    
    # Define key time markers based on typical trading session structure and assumptions
    morning_tradeable_start_time = day_start_dt.replace(hour=9, minute=45)
    morning_tradeable_end_time = day_start_dt.replace(hour=11, minute=45) # Approx, based on typical get_tradeable_data
    lunch_break_start_time = day_start_dt.replace(hour=12, minute=0) 
    afternoon_session_open_time = day_start_dt.replace(hour=13, minute=0)
    afternoon_tradeable_start_time = day_start_dt.replace(hour=13, minute=15) # Approx, based on typical get_tradeable_data
    afternoon_tradeable_end_time = day_start_dt.replace(hour=15, minute=45) # Approx
    
    mid_morning_time = morning_tradeable_start_time + (morning_tradeable_end_time - morning_tradeable_start_time) / 2
    mid_afternoon_time = afternoon_tradeable_start_time + (afternoon_tradeable_end_time - afternoon_tradeable_start_time) / 2

    non_trade_shade_alpha = 0.15
    plot_count = 0

    for i, (strategy_name, trades_df) in enumerate(all_trades_on_day_dict.items()):
        ax = axes[i]
        plot_count += 1

        ax.plot(day_data_full_resolution.index, day_data_full_resolution[price_column], 
                label=f'Market {price_column}', color='darkgrey', alpha=0.9, zorder=1, linewidth=1.0)

        # Shading for non-tradeable/restricted periods
        ax.axvspan(trading_session_start_dt, morning_tradeable_start_time, 
                   facecolor='lightyellow', alpha=non_trade_shade_alpha, zorder=0)
        ax.axvspan(morning_tradeable_end_time, lunch_break_start_time, 
                   facecolor='lightyellow', alpha=non_trade_shade_alpha, zorder=0)
        ax.axvspan(lunch_break_start_time, afternoon_session_open_time, 
                   facecolor='lightblue', alpha=non_trade_shade_alpha, zorder=0) # Lunch
        ax.axvspan(afternoon_session_open_time, afternoon_tradeable_start_time,
                   facecolor='lightyellow', alpha=non_trade_shade_alpha, zorder=0)
        ax.axvspan(afternoon_tradeable_end_time, trading_session_end_dt, 
                   facecolor='lightyellow', alpha=non_trade_shade_alpha, zorder=0)

        units_sold_on_day = 0.0
        avg_sell_price_eod = 0.0
        if trades_df is not None and not trades_df.empty and 'units_sold' in trades_df.columns and 'close' in trades_df.columns:
            trades_df['units_sold'] = pd.to_numeric(trades_df['units_sold'], errors='coerce')
            trades_df.dropna(subset=['units_sold'], inplace=True) # Ensure units_sold is numeric

            units_sold_on_day = trades_df['units_sold'].sum()
            # Adjust scatter size dynamically based on total units, ensuring visibility
            size_factor = 400 / total_units_target_for_day if total_units_target_for_day > 0 else 4
            sizes = np.clip(trades_df['units_sold'] * size_factor, 15, 250)
            
            color_map = {'LongShortMovingAverage': 'dodgerblue', 'MovingAverage': 'seagreen', 'Momentum': 'crimson', 
                         'LSTM_Proxy': 'darkorchid', 'GBDT_Strategy': 'darkorange', 'RSI_Strategy': 'teal'}
            plot_color = color_map.get(strategy_name, 'red') # Default to red if strategy not in map

            ax.scatter(trades_df['timestamp'], trades_df['close'], 
                       s=sizes, label=f'{strategy_name} Sells', 
                       color=plot_color, alpha=0.8, marker='v', edgecolors='black', linewidths=0.5, zorder=2)
            
            if units_sold_on_day > 1e-6: # Avoid division by zero
                avg_sell_price_eod = (trades_df['value'].sum() / units_sold_on_day)
                ax.axhline(avg_sell_price_eod, color=plot_color, linestyle='--', linewidth=1.2, 
                           label=f'Avg Px EOD: {avg_sell_price_eod:.2f}', zorder=1)

            # --- Annotations for key time markers ---
            time_markers_info = {
                "MidMorn": mid_morning_time, "MornEnd": morning_tradeable_end_time,
                "MidAft": mid_afternoon_time, "AftEnd": afternoon_tradeable_end_time
            }
            y_min_plot, y_max_plot = ax.get_ylim() # Get current y-axis limits for text placement

            for label, marker_time_dt in time_markers_info.items():
                # Ensure marker_time_dt is within the overall trading session for relevance
                if not (trading_session_start_dt <= marker_time_dt <= trading_session_end_dt):
                    continue
                
                trades_up_to_marker = trades_df[trades_df['timestamp'] <= marker_time_dt]
                cum_units_sold_marker = 0.0
                avg_px_marker = 0.0
                
                if not trades_up_to_marker.empty:
                    current_cum_units = trades_up_to_marker['units_sold'].sum()
                    if current_cum_units > 1e-6:
                        cum_units_sold_marker = current_cum_units
                        avg_px_marker = trades_up_to_marker['value'].sum() / cum_units_sold_marker
                
                remaining_units_marker = total_units_target_for_day - cum_units_sold_marker
                ax.axvline(marker_time_dt, color='dimgray', linestyle=':', linewidth=0.9, zorder=0)
                
                text_y_pos_offset = (y_max_plot - y_min_plot) * 0.05 
                if "Morn" in label: text_y_pos = y_max_plot - text_y_pos_offset
                elif "MidAft" in label: text_y_pos = y_min_plot + text_y_pos_offset * 1.5
                else: text_y_pos = y_min_plot + text_y_pos_offset * 2.5 # AftEnd
                
                ha = 'right' if marker_time_dt.hour < 12 or (marker_time_dt.hour == 12 and marker_time_dt.minute == 0) else 'left'
                x_offset_factor = -0.015 if ha == 'right' else 0.015 # Slightly more offset
                
                # --- CORRECTED SECTION FOR X-POSITIONING AND COMPARISON ---
                current_plot_xlim_nums = ax.get_xlim()  # These are float numbers (matplotlib internal date representation)
                plot_min_x_num = current_plot_xlim_nums[0]
                plot_max_x_num = current_plot_xlim_nums[1]
                
                marker_time_as_num = mdates.date2num(marker_time_dt) # Convert marker_time (datetime) to Matplotlib number
                
                # Calculate the numerical x-position for the text annotation
                text_x_pos_as_num = marker_time_as_num + (plot_max_x_num - plot_min_x_num) * x_offset_factor
                
                # Convert this numerical position back to datetime for ax.text(), as ax.text can handle datetime for date axes
                text_x_pos_as_dt_for_plot = mdates.num2date(text_x_pos_as_num)
                # --- END CORRECTION DETAIL ---

                annotation_text = f"AvgPx: {avg_px_marker:.0f}\nRem: {remaining_units_marker:.0f}"
                if label == "AftEnd" and units_sold_on_day > 1e-6: # EOD uses overall results
                     annotation_text = f"AvgPx: {avg_sell_price_eod:.0f}\nRem: {total_units_target_for_day - units_sold_on_day:.0f}"

                # Condition to plot text: compare numerical x-positions with numerical plot limits
                if plot_min_x_num <= text_x_pos_as_num <= plot_max_x_num:
                    ax.text(text_x_pos_as_dt_for_plot, text_y_pos, annotation_text, # Use datetime for plotting text
                            fontsize=7, color='black', ha=ha, 
                            va=('top' if "Morn" in label else "bottom"),
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.75), zorder=3)
        
        units_remaining_at_eod = total_units_target_for_day - units_sold_on_day
        ax.set_title(f'{strategy_name} (EOD Left: {units_remaining_at_eod:.1f})', fontsize=10, pad=3)
        ax.set_ylabel('Price', fontsize=9)
        
        ax.set_xlim(trading_session_start_dt, trading_session_end_dt) # Set x-limits using datetime objects
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1)) 
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,15,30,45])) # More minor ticks
        
        ax.legend(loc='best', fontsize=7, frameon=True, framealpha=0.8) # 'best' can be slow, consider 'upper left' etc.
        ax.grid(True, linestyle=':', alpha=0.5)

        # Manage x-axis labels for subplots
        if nrows > 1 and i < num_algos - ncols : # If not in the last row of subplots
            ax.set_xlabel('')
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel('Time (HKT)', fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Hide any unused subplots if num_algos doesn't perfectly fill the grid
    for j in range(plot_count, nrows * ncols):
        fig.delaxes(axes[j]) # More robust than set_visible(False)

    fig.suptitle(f'HSI Selling Simulation: {date_obj.strftime("%Y-%m-%d")} (Target: {total_units_target_for_day:.0f} units)', 
                 fontsize=14, y=0.99) # Adjusted y for suptitle
    plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # rect to make space for suptitle and xlabels
    plt.show()

def plot_ave_returns_and_volatility(df, column_name='close',
                                    sampling_str="15min"):
    # Calculate 1-minute returns
    df['returns'] = df[column_name].pct_change()
    df.dropna(inplace=True) # Remove NaN from first return

    # Ensure the index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df['time_slot'] = df.index.floor(sampling_str).time # Gets the start time of the 30-min interval

    avg_return_by_slot = df.groupby('time_slot')['returns'].mean()
    std_return_by_slot = df.groupby('time_slot')['returns'].std()
    count_by_slot = df.groupby('time_slot')['returns'].count()

    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.bar(range(len(avg_return_by_slot)), avg_return_by_slot, 
            color='dodgerblue', alpha=0.7, label='Mean Return')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Mean 1-min Return', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8) # Zero line for reference

    # Set x-ticks to be the time slot labels
    time_slot_labels = [ts.strftime('%H:%M') for ts in avg_return_by_slot.index]
    x_positions = range(len(avg_return_by_slot))
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(time_slot_labels, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(range(len(std_return_by_slot)), std_return_by_slot, color='forestgreen',
            marker='o', linestyle='--', label='Std Dev of Return (Volatility)')
    ax2.set_ylabel('Std Dev of 1-min Return', color='forestgreen')
    ax2.tick_params(axis='y', labelcolor='forestgreen')

    plt.title('Average Return and Volatility by 15 Min of Day')
    fig.tight_layout()
    plt.grid(True, axis='x')
    plt.show()

def plot_intraday_three_curves(hsi_df, hsfi_df, hscat100_df,
                                afternoon_start = ' 13:15:00',
                                morning_start = ' 09:45:00',
                                morning_end = ' 11:45:00',
                                morning_early_end = ' 11:25:00',
                                afternoon_end = ' 15:45:00',
                                afternoon_early_end = ' 14:55:00'):
    # 1) find the top‐N turnover days
    daily_turn = hsi_df['turnover'].resample('D').sum()
    top_n = 6
    top_dates = daily_turn.sort_values(ascending=False).head(top_n).index.date

    # 2) prepare subplot grid
    ncols = 2
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(14, 4*nrows),
                            constrained_layout=True)
    axes = axes.flatten()

    # width of each 5‑min bar, in days (for matplotlib.dates)
    bar_width = 4 / (24 * 60)

    for ax, date in zip(axes, top_dates):
        # slice out that one day
        hsi_df_day = hsi_df[hsi_df.index.date == date]
        hsfi_df_day = hsfi_df[hsfi_df.index.date == date]
        hscat100_df_day = hscat100_df[hscat100_df.index.date == date]

        hsfi_df_day_morning = hsfi_df_day.between_time(morning_start.strip(), morning_end.strip())
        hsfi_df_day_afternoon = hsfi_df_day.between_time(afternoon_start.strip(), afternoon_end.strip())
        hscat100_df_day_morning = hscat100_df_day.between_time(morning_start.strip(), morning_early_end.strip())
        hscat100_df_day_afternoon = hscat100_df_day.between_time(afternoon_start.strip(), afternoon_early_end.strip())
        
        # resample to 5‑min OHLC
        ohlc = hsi_df_day[['open','high','low','close']].resample('5min').agg({
            'open':  'first',
            'high':  'max',
            'low':   'min',
            'close': 'last'
        }).dropna()
        
        # resample turnover
        turnover = hsi_df_day['turnover'].resample('5min').sum().loc[ohlc.index]
        
        # build the sequence of (date_num, o,h,l,c)
        data_ohlc = ohlc.copy()
        data_ohlc['date_num'] = mdates.date2num(data_ohlc.index.to_pydatetime())
        quotes = [
            (row.date_num, row.open, row.high, row.low, row.close)
            for row in data_ohlc.itertuples()
        ]
        
        # 3) plot candles
        candlestick_ohlc(ax, quotes, width=bar_width,
                        colorup='forestgreen', colordown='firebrick',
                        alpha=0.8)
        
        # 4) overlay turnover on a twin‐y
        ax2 = ax.twinx()
        ax2.bar(mdates.date2num(turnover.index.to_pydatetime()),
                turnover.values,
                width=bar_width,
                alpha=0.4,
                color='dodgerblue',
                label='Turnover')
        
        # Plot HSFI Closing Price (Deep Green)
        ax3 = ax.twinx()
        ax3.plot(hsfi_df_day_morning.index, hsfi_df_day_morning['close'], 
                color='#006400', 
                linewidth=1,
                alpha=0.5,
                label='HSFI Closing Price')
        ax3.plot(hsfi_df_day_afternoon.index, hsfi_df_day_afternoon['close'], 
                color='#006400', 
                linewidth=1,
                alpha=0.5,)
        ax3.yaxis.set_visible(False)  # Disable y-axis for ax3

        # Plot HSCAT100 Closing Price (Deep Yellow)
        ax4 = ax.twinx()
        ax4.plot(hscat100_df_day_morning.index, hscat100_df_day_morning['close'], 
                color='#FFD700', 
                linewidth=1,
                alpha=0.7,
                label='HSCAT100 Closing Price')
        ax4.plot(hscat100_df_day_afternoon.index, hscat100_df_day_afternoon['close'], 
                color='#FFD700', 
                linewidth=1,
                alpha=0.7,)
        ax4.yaxis.set_visible(False)  # Disable y-axis for ax4

        # styling
        ax.set_title(f'{date}  (Total turnover: {daily_turn[str(date)]:.0f})')
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_ylabel('Price')
        ax2.set_ylabel('Turnover')

        # Combine legends
        lines = []
        labels = []
        for this_ax in [ax, ax2, ax3, ax4]:
            ax_lines, ax_labels = this_ax.get_legend_handles_labels()
            lines.extend(ax_lines)
            labels.extend(ax_labels)
        ax.legend(lines, labels, loc='upper center')
        
    # hide any unused axes if top_n < nrows*ncols
    for ax in axes[top_n:]:
        ax.set_visible(False)

    plt.show()


def plot_fft_analysis(df, column_name='close', T_sampling=1.0,
                      short_ma_period_suggestion=25,
                      long_ma_period_suggestion=75,
                      plot_period_min=2, plot_period_max=240):
    """
    Performs FFT analysis on a given column of a DataFrame and plots:
    1. Amplitude Spectrum vs. Frequency
    Adds vertical lines to suggest good short and long moving average window lengths based on the frequency curve.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        column_name (str): Name of the column in df to analyze (e.g., 'close').
        T_sampling (float): Sampling interval in time units (e.g., 1.0 for 1 minute).
                            The frequency unit will be cycles per this time unit.
        plot_period_min (float): Minimum period to focus on in the plot (transforms to max frequency).
        plot_period_max (float): Maximum period to focus on in the plot (transforms to min frequency).
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        return None, (None, None)
        
    prices = df[column_name].values
    N = len(prices)  # FFT size
    freq_cutoff = 256

    # Apply FFT
    yf = fft(prices)
    xf = fftfreq(N, T_sampling)
    xf *= N
    amplitude_spectrum = np.abs(yf[:freq_cutoff])  # Take the first N points

    # Extract positive frequencies (excluding DC component)
    xf_positive = xf[:freq_cutoff]
    amplitude_positive = amplitude_spectrum[:freq_cutoff]

    # Find peaks in the positive amplitude spectrum
    peaks_indices, _ = find_peaks(amplitude_positive)
    peak_freqs = xf_positive[peaks_indices]
    peak_periods = 1.0 / peak_freqs

    # Filter peaks based on the valid period range
    valid_mask = (peak_periods >= plot_period_min) & (peak_periods <= plot_period_max)
    valid_peak_periods = peak_periods[valid_mask]
    valid_peak_amplitudes = amplitude_positive[peaks_indices][valid_mask]

    # Determine suggested periods
    if len(valid_peak_periods) >= 2:
        # Sort peaks by amplitude (descending)
        sorted_indices = np.argsort(-valid_peak_amplitudes)
        top_two_periods = valid_peak_periods[sorted_indices[:2]]
        short_ma_period = min(top_two_periods)
        long_ma_period = max(top_two_periods)
    elif len(valid_peak_periods) == 1:
        short_ma_period = valid_peak_periods[0]
        long_ma_period = valid_peak_periods[0]
    else:
        # Default values if no peaks are found
        short_ma_period = short_ma_period_suggestion
        long_ma_period = long_ma_period_suggestion

    short_ma_period_suggestion = int(round(short_ma_period))
    long_ma_period_suggestion = int(round(long_ma_period))

    # Create plot
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 10))

    # Plot Amplitude Spectrum
    ax1.plot(xf_positive, amplitude_positive, color='dodgerblue')
    ax1.set_title(f'Amplitude Spectrum for "{column_name}"')
    ax1.set_xlabel('Frequency (cycles per time unit)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.set_yscale('log')

    # Add vertical lines for suggested MA periods
    legend_handles = []
    labels = []

    if short_ma_period_suggestion > 0:
        freq_short = 1.0 / short_ma_period_suggestion
        line_short = ax1.axvline(freq_short, color='grey', linestyle='--', lw=1.5)
        legend_handles.append(line_short)
        labels.append(f'Short MA ({short_ma_period_suggestion} periods)')

    if long_ma_period_suggestion > 0:
        freq_long = 1.0 / long_ma_period_suggestion
        line_long = ax1.axvline(freq_long, color='darkgrey', linestyle=':', lw=1.5)
        legend_handles.append(line_long)
        labels.append(f'Long MA ({long_ma_period_suggestion} periods)')

    if legend_handles:
        ax1.legend(legend_handles, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'FFT Analysis: {column_name} Price Series', fontsize=16)
    plt.show()

    return fig, (ax1)

# --- Helper Function for Plotting ACF (Zoomed with Annotations) ---
def plot_zoomed_acf_with_annotations(series, series_name, nlags=60, lags_to_ignore_for_ylim=5):
    """
    Plots a "zoomed-in" ACF for a given series, with annotations for initial lags
    including the 0-th lag.
    """
    if series.empty or len(series.dropna()) <= nlags:
        print(f"Series '{series_name}' is empty or too short. Cannot plot ACF.")
        return

    data_to_plot = series.dropna()

    # Calculate ACF values including lag 0
    # `plot_acf` internally calls `acf` but doesn't show lag 0. We need it for annotation.
    # `acf_values` from `statsmodels.tsa.stattools.acf` INCLUDES lag 0.
    acf_values, acf_confint = acf(data_to_plot, nlags=nlags, alpha=0.05, fft=False) # fft=False for consistency with plot_acf's default for smaller series

    fig_zoom_acf, ax_zoom_acf = plt.subplots(figsize=(12, 6)) # Increased height a bit

    # `plot_acf` itself does not plot lag 0. The x-axis for its bars starts from lag 1.
    # So the bar for acf_values[1] is at x=1, acf_values[2] at x=2 etc.
    plot_acf(data_to_plot, lags=nlags, ax=ax_zoom_acf, title=f'ACF of {series_name} (Zoomed - Y-lim from lag {lags_to_ignore_for_ylim+1})')
    ax_zoom_acf.set_xlabel('Lag')
    ax_zoom_acf.set_ylabel('Autocorrelation')
    ax_zoom_acf.grid(True)

    # Set y-limits for zoomed view (based on lags *after* lags_to_ignore_for_ylim)
    # Note: acf_values[0] is lag 0, acf_values[1] is lag 1, etc.
    # So we consider acf_values starting from index (lags_to_ignore_for_ylim + 1)
    if len(acf_values) > lags_to_ignore_for_ylim + 1: # Check if there are enough lags
        relevant_acf_values_for_ylim = acf_values[lags_to_ignore_for_ylim + 1:]
        # acf_confint also starts from lag 0, so same indexing applies
        relevant_confint_lower_abs = acf_confint[lags_to_ignore_for_ylim + 1:, 0]
        relevant_confint_upper_abs = acf_confint[lags_to_ignore_for_ylim + 1:, 1]

        if len(relevant_acf_values_for_ylim) > 0:
            min_val = min(relevant_acf_values_for_ylim.min(), relevant_confint_lower_abs.min())
            max_val = max(relevant_acf_values_for_ylim.max(), relevant_confint_upper_abs.max())
            padding = abs(max_val - min_val) * 0.1 + 0.01
            ax_zoom_acf.set_ylim(min_val - padding, max_val + padding)
        else:
            print(f"Not enough lags in {series_name} to create a zoomed y-limit beyond lag {lags_to_ignore_for_ylim}.")
    else:
        print(f"Not enough ACF values for {series_name} (less than {lags_to_ignore_for_ylim + 1} total lags) to create a zoomed y-limit.")

    # Add text annotations for the first 'lags_to_ignore_for_ylim' (AND lag 0)
    y_plot_min, y_plot_max = ax_zoom_acf.get_ylim()
    y_plot_range = y_plot_max - y_plot_min

    # Annotate Lag 0 (Self-correlation = 1)
    # The x-coordinate for lag 0 on the plot_acf output is indeed 0.
    # plot_acf from statsmodels.graphics.tsaplots.plot_acf sets xlim starting from 0
    # but doesn't draw a bar at 0. We will add the text annotation there.
    val_lag0 = acf_values[0] # Should always be 1.0
    text_y_lag0 = y_plot_max - (y_plot_range * 0.03) # Place at top of zoomed y-axis
    ax_zoom_acf.text(0, text_y_lag0, f"↑ Lag 0: {val_lag0:.2f}", ha='center', va='top', fontsize=8, color='blue',
                     bbox=dict(facecolor='white', alpha=0.7, pad=0.5, boxstyle='round,pad=0.2'))
    ax_zoom_acf.axvline(0, color='blue', linestyle=':', linewidth=0.8, alpha=0.5) # Optional: mark x=0

    # Annotate Lags 1 to lags_to_ignore_for_ylim
    # The x-coordinate for these lags on the plot_acf output corresponds to their lag number.
    for k in range(1, min(lags_to_ignore_for_ylim + 1, len(acf_values))):
        val = acf_values[k]
        prefix = ""
        y_text_offset_factor = 0.03

        if val > y_plot_max:
            text_y = y_plot_max - (y_plot_range * y_text_offset_factor)
            va = 'top'
            prefix = "↑ "
        elif val < y_plot_min:
            text_y = y_plot_min + (y_plot_range * y_text_offset_factor)
            va = 'bottom'
            prefix = "↓ "
        else:
            text_y = val + np.sign(val) * (y_plot_range * 0.015) if val != 0 else (y_plot_range * 0.015)
            if np.sign(val) >= 0: va = 'bottom' if text_y < y_plot_max else 'top'
            else: va = 'top' if text_y > y_plot_min else 'bottom'

        ax_zoom_acf.text(k, text_y, f"{prefix}Lag {k}: {val:.3f}", ha='center', va=va, fontsize=7, color='red',
                         bbox=dict(facecolor='white', alpha=0.6, pad=0.5, boxstyle='round,pad=0.2'))

    plt.suptitle(f"Zoomed ACF for {series_name} (Y-axis from lag {lags_to_ignore_for_ylim+1}, initial lags annotated)", y=1.03, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()
