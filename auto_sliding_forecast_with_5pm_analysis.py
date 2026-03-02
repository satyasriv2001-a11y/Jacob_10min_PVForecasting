import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from data.data_utils import preprocess_features

# Try importing PyTorch and models (optional for sklearn-only usage)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available - only sklearn models will work")

try:
    from models.tcn import TCNModel
    TCN_AVAILABLE = True
except ImportError:
    TCN_AVAILABLE = False


def parse_model_name(model_path):
    """
    Parse model filename to extract configuration.
    
    Supports formats like:
    - "best_model_LSTM_high_PV+HW_72h_noTE.pt"
    - "186_XGB_high_PV+NWP_24h_noTE.json"
    - "MultiCH_high_PV+NWP+_24h_TE"
    - "Transformer_low_NWP_TE.pt"
    
    Returns configuration dict with all necessary parameters.
    """
    filename = model_path.split('/')[-1]
    
    # Extract model type
    model_match = re.search(r'(LSTM|GRU|Transformer|TCN|MultiChannelLSTM|MultiCH|XGB|LGBM|RF|Linear|LR)', filename, re.IGNORECASE)
    if model_match:
        model_type = model_match.group(1)
        if 'MultiCH' in model_type:
            model_type = 'MultiChannelLSTM'
    else:
        model_type = 'LSTM'  # Default
    
    # Extract complexity
    complexity = 'high' if '_high_' in filename or 'high' in filename.lower() else 'low'
    
    # Extract lookback hours
    hours_match = re.search(r'_(\d+)h_', filename)
    if hours_match:
        past_hours = int(hours_match.group(1))
    else:
        # Check for NWP-only (no lookback)
        if re.search(r'_NWP[+]?_', filename) and 'PV' not in filename:
            past_hours = 0
        else:
            past_hours = 24  # Default
    
    # Extract time encoding
    use_time_encoding = '_TE' in filename and 'noTE' not in filename
    
    # Parse feature configuration
    # Patterns: PV, PV+HW, PV+NWP, PV+NWP+, NWP, NWP+
    use_pv = 'PV' in filename and not re.search(r'\bNWP(?!\+)', filename) or 'PV+' in filename
    use_hist_weather = 'PV+HW' in filename
    use_forecast = 'NWP' in filename
    use_ideal_nwp = 'NWP+' in filename
    
    # Handle NWP-only case
    if re.search(r'\bNWP[+]?\b', filename) and 'PV' not in filename:
        use_pv = False
        use_hist_weather = False
        use_forecast = True
    
    # Build config
    config = {
        'model_type': model_type,
        'model_complexity': complexity,
        'past_hours': past_hours,
        'future_hours': 24,  # Always 24h forecast
        'use_pv': use_pv,
        'use_hist_weather': use_hist_weather,
        'use_forecast': use_forecast,
        'use_ideal_nwp': use_ideal_nwp,
        'use_time_encoding': use_time_encoding,
        'weather_category': 'all_weather',
        'start_date': '2022-01-01',
        'end_date': '2024-09-28',
        'no_hist_power': past_hours == 0,
    }
    
    # Add model params based on complexity
    if complexity == 'low':
        config['model_params'] = {
            'd_model': 32,
            'hidden_dim': 16,
            'num_heads': 2,
            'num_layers': 1,
            'dropout': 0.1,
            'tcn_channels': [16, 32],
            'kernel_size': 3,
            'te_dim': 8,
        }
    else:
        config['model_params'] = {
            'd_model': 64,
            'hidden_dim': 32,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'tcn_channels': [32, 64],
            'kernel_size': 3,
            'te_dim': 8,
        }
    
    return config


def compute_TE_for_timestep(df_clean, idx, hist_steps, fut_steps, TE_COLS):
    """
    Compute time encoding features for a single timestep.
    Returns flattened TE array for historical + forecast window.
    """
    hist_start = idx - hist_steps
    fut_end = idx + fut_steps
    
    # Check if TE columns exist
    missing = [c for c in TE_COLS if c not in df_clean.columns]
    if missing:
        return None
    
    te_hist = df_clean.iloc[hist_start:idx][TE_COLS].values
    te_fut = df_clean.iloc[idx:fut_end][TE_COLS].values
    
    te = np.concatenate([te_hist, te_fut], axis=0).reshape(-1)
    return te


def auto_sliding_forecast(
    model,
    model_name,
    df_raw,
    midnight_index=None,
    show_individual_plots=False,
    save_to_csv=True,
    output_dir="forecast_results",
    plot_every_n=24  # Plot every N forecasts to avoid clutter (24 = every 4 hours)
):
    """
    Universal sliding 24h forecast every 10 minutes.
    
    Works with:
    - Deep Learning models (LSTM, GRU, Transformer, TCN, MultiChannelLSTM)
    - Machine Learning models (XGBoost, Linear Regression, etc.)
    
    Args:
        model: Trained model (PyTorch module or sklearn model)
               - DL: torch.nn.Module or loaded with torch.load()
               - ML: sklearn model with .predict() method
        model_name: Model filename for config parsing
                   Examples: "LSTM_high_PV+HW_72h_noTE.pt"
                            "186_XGB_high_PV+NWP_24h_noTE.json"
        df_raw: Raw dataframe with DateTime column
        midnight_index: Which midnight to start from (None = random)
        show_individual_plots: Whether to show individual forecast plots
        save_to_csv: Whether to save results to CSV files
        output_dir: Directory to save CSV files
        plot_every_n: Show every Nth forecast in combined plot (default 24 = every 4 hours)
    
    Returns:
        predictions: List of 288 prediction arrays (one every 10 minutes)
        ground_truth: 48h ground truth array
        config: Parsed configuration dict
        csv_paths: Dict of paths to saved CSV files (if save_to_csv=True)
    
    Loading Examples:
        # PyTorch models
        model = torch.load('model.pt')
        
        # Sklearn models (pickled)
        import pickle
        with open('model.json', 'rb') as f:
            model = pickle.load(f)
    """
    
    # Parse configuration from model name
    config = parse_model_name(model_name)
    
    print(f"[CONFIG] Model: {config['model_type']}")
    print(f"[CONFIG] Complexity: {config['model_complexity']}")
    print(f"[CONFIG] Lookback: {config['past_hours']}h")
    print(f"[CONFIG] Features: PV={config['use_pv']}, HW={config['use_hist_weather']}, "
          f"NWP={config['use_forecast']}, NWP+={config['use_ideal_nwp']}")
    print(f"[CONFIG] Time Encoding: {config['use_time_encoding']}")
    
    # Preprocess data
    df_raw_copy = df_raw.copy()
    
    # Ensure Datetime column exists with robust parsing
    if 'Datetime' not in df_raw_copy.columns:
        # Find the datetime column
        datetime_col = None
        for col in ['date', 'DateTime', 'datetime', 'Date', 'timestamp', 'Timestamp']:
            if col in df_raw_copy.columns:
                datetime_col = col
                break
        
        if datetime_col is None:
            raise ValueError("No datetime column found in dataframe. Expected columns: 'date', 'DateTime', 'Datetime', etc.")
        
        # Robust datetime parsing
        date_series = df_raw_copy[datetime_col]
        
        # Check if it's numeric (timestamp)
        if pd.api.types.is_numeric_dtype(date_series):
            # Determine if it's seconds, milliseconds, or nanoseconds
            sample_val = date_series.dropna().iloc[0]
            if sample_val > 1e15:  # Likely nanoseconds or milliseconds
                if sample_val > 1e18:  # Nanoseconds
                    df_raw_copy['Datetime'] = pd.to_datetime(date_series, unit='ns')
                else:  # Milliseconds
                    df_raw_copy['Datetime'] = pd.to_datetime(date_series, unit='ms')
            else:  # Seconds
                df_raw_copy['Datetime'] = pd.to_datetime(date_series, unit='s')
        else:
            # Try mixed format parsing for strings
            try:
                df_raw_copy['Datetime'] = pd.to_datetime(date_series, format='mixed', errors='coerce')
            except:
                # Fallback to default parsing
                df_raw_copy['Datetime'] = pd.to_datetime(date_series, errors='coerce')
        
        # Remove any rows with invalid dates
        invalid_dates = df_raw_copy['Datetime'].isna().sum()
        if invalid_dates > 0:
            print(f"[WARNING] Removed {invalid_dates} rows with invalid dates")
            df_raw_copy = df_raw_copy.dropna(subset=['Datetime'])
    
    # Ensure timezone-naive
    if df_raw_copy['Datetime'].dt.tz is not None:
        df_raw_copy['Datetime'] = df_raw_copy['Datetime'].dt.tz_localize(None)
    
    df_clean, hist_feats, fcst_feats, sh, sf, st, no_hist_power = \
        preprocess_features(df_raw_copy, config)
    
    print(f"[INFO] Historical features ({len(hist_feats)}): {hist_feats[:5]}...")
    print(f"[INFO] Forecast features ({len(fcst_feats)}): {fcst_feats[:5]}...")
    
    # Find midnight indices
    df_midnight = df_clean[df_clean['Datetime'].dt.hour == 0]
    midnight_indices = df_midnight.index.tolist()
    
    if midnight_index is None:
        midnight_index = np.random.choice(len(midnight_indices))
    
    start_idx = midnight_indices[midnight_index]
    
    print(f"[INFO] Midnight sample #{midnight_index} → df index {start_idx}")
    print(f"[INFO] Timestamp = {df_clean.loc[start_idx, 'Datetime']}")
    
    # Window sizes
    hist_steps = config['past_hours'] * 6
    fut_steps = config['future_hours'] * 6
    
    # Detect target column
    target_col = [c for c in df_clean.columns if 'cap' in c.lower()][0]
    y_full_48h = df_clean[target_col].iloc[start_idx : start_idx + 288].values
    
    # Determine model type (DL vs ML)
    is_pytorch = TORCH_AVAILABLE and isinstance(model, torch.nn.Module)
    is_tcn = TCN_AVAILABLE and isinstance(model, TCNModel)
    is_sklearn = hasattr(model, 'predict') and not is_pytorch
    is_multichannel = config['model_type'] == 'MultiChannelLSTM'
    
    print(f"[INFO] Model type detected: PyTorch={is_pytorch}, sklearn={is_sklearn}, TCN={is_tcn}, MultiChannel={is_multichannel}")
    
    # Time encoding columns
    TE_COLS = [
        "month_cos", "month_sin",
        "hour_cos", "hour_sin",
        "daypos_cos", "daypos_sin",
        "hourpos_sin", "hourpos_cos"
    ]
    
    # Set model to eval mode if PyTorch
    if is_pytorch:
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"[INFO] Using device: {device}")
    
    # Generate predictions (every 10 minutes = every timestep)
    print(f"\n[INFO] Generating 288 sliding forecasts (every 10 minutes)...")
    preds_all = []
    
    # Loop through each 10-minute timestep (288 total for 48 hours)
    for timestep in range(288):
        idx = start_idx + timestep
        
        # Extract features
        X_h = df_clean[hist_feats].iloc[idx - hist_steps : idx].values
        
        if len(fcst_feats) > 0:
            X_f = df_clean[fcst_feats].iloc[idx : idx + fut_steps].values
        else:
            X_f = np.zeros((fut_steps, 0))
        
        # =====================================================================
        # SKLEARN MODELS (ML)
        # =====================================================================
        if is_sklearn:
            # Flatten and concatenate
            X_h_flat = X_h.flatten()
            X_f_flat = X_f.flatten()
            
            if len(X_f_flat) > 0:
                X_combined = np.concatenate([X_h_flat, X_f_flat]).reshape(1, -1)
            else:
                X_combined = X_h_flat.reshape(1, -1)
            
            pred = model.predict(X_combined).reshape(-1)
        
        # =====================================================================
        # PYTORCH MODELS (DL)
        # =====================================================================
        elif is_pytorch:
            with torch.no_grad():
                
                # -----------------------------------------------------------
                # MultiChannelLSTM (requires TE)
                # -----------------------------------------------------------
                if is_multichannel:
                    # Compute TE if needed
                    if config['use_time_encoding']:
                        te = compute_TE_for_timestep(df_clean, idx, hist_steps, fut_steps, TE_COLS)
                        if te is None:
                            raise ValueError("Time encoding required but TE columns not found")
                        te_tensor = torch.tensor(te, dtype=torch.float32).unsqueeze(0).to(device)
                    else:
                        # Create dummy TE
                        te_size = (hist_steps + fut_steps) * 8
                        te_tensor = torch.zeros(1, te_size, dtype=torch.float32).to(device)
                    
                    X_h_tensor = torch.tensor(X_h, dtype=torch.float32).unsqueeze(0).to(device)
                    X_f_tensor = torch.tensor(X_f, dtype=torch.float32).unsqueeze(0).to(device) if X_f.shape[1] > 0 else torch.zeros(1, fut_steps, 0).to(device)
                    
                    pred = model(X_h_tensor, X_f_tensor, te_tensor).cpu().numpy().reshape(-1)
                
                # -----------------------------------------------------------
                # TCN
                # -----------------------------------------------------------
                elif is_tcn:
                    # TCN expects (batch, channels, time)
                    X_h_tensor = torch.tensor(X_h, dtype=torch.float32).T.unsqueeze(0).to(device)
                    
                    if X_f.shape[1] > 0:
                        X_f_tensor = torch.tensor(X_f, dtype=torch.float32).T.unsqueeze(0).to(device)
                        pred = model(X_h_tensor, X_f_tensor).cpu().numpy().reshape(-1)
                    else:
                        pred = model(X_h_tensor).cpu().numpy().reshape(-1)
                
                # -----------------------------------------------------------
                # Standard RNN/Transformer (LSTM, GRU, Transformer)
                # -----------------------------------------------------------
                else:
                    # Standard expects (batch, time, features)
                    X_h_tensor = torch.tensor(X_h, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    if X_f.shape[1] > 0:
                        X_f_tensor = torch.tensor(X_f, dtype=torch.float32).unsqueeze(0).to(device)
                        pred = model(X_h_tensor, X_f_tensor).cpu().numpy().reshape(-1)
                    else:
                        pred = model(X_h_tensor).cpu().numpy().reshape(-1)
        
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        preds_all.append(np.clip(pred, 0, None))
    
    # Build 48h datetime axis
    dt0 = df_clean.loc[start_idx, 'Datetime']
    times_48h = [dt0 + pd.Timedelta(minutes=10 * i) for i in range(288)]
    time_labels_48h = [t.strftime('%H:%M') for t in times_48h]
    
    # Create timestep axis (0 to 287)
    timesteps = np.arange(288)
    
    # Combined sliding plot (plotting every Nth forecast to avoid clutter)
    plt.figure(figsize=(20, 10))
    
    plt.plot(
        timesteps, y_full_48h,
        color='black', linewidth=3, marker='o', markersize=3,
        label='Ground Truth (48h)', zorder=10
    )
    
    print(f"[INFO] Plotting every {plot_every_n}th forecast ({288//plot_every_n} total)...")
    for timestep in range(0, 288, plot_every_n):
        pred = preds_all[timestep]
        
        # Calculate which 24h segment this forecast covers
        timestep_segment = timesteps[timestep : min(timestep + 144, 288)]
        pred_segment = pred[:len(timestep_segment)]
        gt_segment = y_full_48h[timestep : min(timestep + 144, 288)]
        
        if len(gt_segment) == len(pred_segment):
            rmse = np.sqrt(mean_squared_error(gt_segment, pred_segment))
        else:
            rmse = np.nan
        
        hours_offset = timestep / 6
        
        plt.plot(
            timestep_segment,
            pred_segment,
            linewidth=1.2,
            marker='o', markersize=2,
            alpha=0.5,
            label=f'+{hours_offset:.1f}h (RMSE={rmse:.2f})' if not np.isnan(rmse) else f'+{hours_offset:.1f}h',
        )
    
    # Set x-ticks every 30 timesteps (every 5 hours)
    tick_spacing = 30
    plt.xticks(np.arange(0, 288, tick_spacing), [f'{i}' for i in range(0, 288, tick_spacing)])
    
    # plt.legend(ncol=4, fontsize=8, loc='upper left')
    plt.grid(alpha=0.3)
    
    # Build title with config details
    title = f"Sliding 24h Forecasts (every 10 min) - {config['model_type']} ({config['model_complexity']})\n"
    feat_str = []
    if config['use_pv']:
        feat_str.append('PV')
    if config['use_hist_weather']:
        feat_str.append('HW')
    if config['use_forecast']:
        feat_str.append('NWP+' if config['use_ideal_nwp'] else 'NWP')
    title += f"Features: {'+'.join(feat_str)}, {config['past_hours']}h lookback, "
    title += "TE" if config['use_time_encoding'] else "noTE"
    title += f" | Showing every {plot_every_n}th forecast"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Capacity Factor', fontsize=12)
    plt.xlabel('Timestep (10-minute intervals)', fontsize=12)
    plt.tight_layout()
    
    # Save plot BEFORE showing (critical for non-interactive backends)
    if save_to_csv:
        plot_filename = f"{model_name.replace('.pt', '').replace('.json', '').replace('.pkl', '')}_combined_plot_{dt0.strftime('%Y%m%d_%H%M')}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Combined plot → {plot_path}")
    
    plt.show()
    
    # Individual plots (optional, for every N forecasts)
    if show_individual_plots:
        print(f"\n[INFO] Generating individual forecast plots (every {plot_every_n}th)...")
        for timestep in range(0, 288, plot_every_n):
            plt.figure(figsize=(12, 5))
            
            pred = preds_all[timestep]
            
            # Calculate segment
            time_segment = times_48h[timestep : min(timestep + 144, 288)]
            pred_segment = pred[:len(time_segment)]
            gt_segment = y_full_48h[timestep : min(timestep + 144, 288)]
            
            if len(gt_segment) == len(pred_segment):
                rmse = np.sqrt(mean_squared_error(gt_segment, pred_segment))
            else:
                rmse = np.nan
            
            hours_offset = timestep / 6
            
            plt.plot(time_segment, gt_segment, linewidth=3.5, marker='o', markersize=3.5, 
                    label='Ground Truth', color='black')
            plt.plot(time_segment, pred_segment, linewidth=2, marker='o', markersize=2,
                    label=f'Prediction +{hours_offset:.1f}h (RMSE={rmse:.2f})' if not np.isnan(rmse) else f'Prediction +{hours_offset:.1f}h', 
                    alpha=0.7)
            
            plt.xticks(time_segment[::6], [t.strftime('%H:%M') for t in time_segment[::6]], rotation=45)
            plt.grid(alpha=0.3)
            plt.legend()
            
            title = f"24h Forecast Starting +{hours_offset:.1f}h - {config['model_type']} ({config['model_complexity']})"
            plt.title(title)
            plt.ylabel('Capacity Factor')
            plt.xlabel('Time (HH:MM)')
            plt.tight_layout()
            plt.show()
    
    print(f"\n[COMPLETE] Sliding forecast finished! Generated {len(preds_all)} forecasts.")
    
    # ========================================================================
    # ANALYZE ACCURACY AS WE APPROACH 5PM (INTERVAL END - 1 HOUR)
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYZING ACCURACY AS WE APPROACH 5PM")
    print("="*70)
    
    # Find the first 5pm (17:00) in the 48h window
    first_5pm_idx = None
    for i, t in enumerate(times_48h):
        if t.hour == 17 and t.minute == 0:
            first_5pm_idx = i
            break
    
    if first_5pm_idx is not None:
        print(f"[INFO] First 5pm found at timestep {first_5pm_idx}")
        print(f"[INFO] Timestamp: {times_48h[first_5pm_idx]}")
        
        # Calculate accuracy metrics for forecasts that predict values within 10am-6pm window
        # As we get closer to 5pm, track the RMSE for the 10am-6pm portion of each forecast
        approach_analysis = []
        
        # We'll look at forecasts starting from midnight up until 5pm
        # For each forecast, calculate RMSE only for predictions that fall in 10am-6pm
        for timestep in range(min(first_5pm_idx + 1, len(preds_all))):
            pred = preds_all[timestep]
            forecast_start_time = times_48h[timestep]
            
            # Collect predictions that fall within 10am-6pm window
            daytime_predictions = []
            daytime_ground_truth = []
            
            for i in range(len(pred)):
                pred_timestep = timestep + i
                if pred_timestep >= len(times_48h):
                    break
                    
                pred_time = times_48h[pred_timestep]
                
                # Check if this prediction falls in 10am-6pm window
                if 10 <= pred_time.hour < 18:
                    daytime_predictions.append(pred[i])
                    daytime_ground_truth.append(y_full_48h[pred_timestep])
            
            # Calculate RMSE for daytime predictions only
            if len(daytime_predictions) > 0:
                daytime_rmse = np.sqrt(mean_squared_error(daytime_ground_truth, daytime_predictions))
                daytime_mae = np.mean(np.abs(np.array(daytime_ground_truth) - np.array(daytime_predictions)))
                
                # Time until 5pm
                time_until_5pm = (times_48h[first_5pm_idx] - forecast_start_time).total_seconds() / 3600
                
                approach_analysis.append({
                    'timestep': timestep,
                    'forecast_start': forecast_start_time,
                    'forecast_start_hour': forecast_start_time.hour + forecast_start_time.minute / 60,
                    'hours_until_5pm': time_until_5pm,
                    'daytime_rmse': daytime_rmse,
                    'daytime_mae': daytime_mae,
                    'n_daytime_predictions': len(daytime_predictions)
                })
        
        df_approach = pd.DataFrame(approach_analysis)
        
        # Plot accuracy as we approach 5pm
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: RMSE vs Time Until 5pm
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df_approach['hours_until_5pm'], df_approach['daytime_rmse'], 
                   alpha=0.6, s=50, c=df_approach['forecast_start_hour'], cmap='viridis')
        ax1.plot(df_approach['hours_until_5pm'], df_approach['daytime_rmse'], 
                alpha=0.3, linewidth=1)
        ax1.set_xlabel('Hours Until 5pm', fontsize=12)
        ax1.set_ylabel('RMSE (10am-6pm predictions only)', fontsize=12)
        ax1.set_title('Forecast Accuracy as We Approach 5pm', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.invert_xaxis()  # So time flows left to right (approaching 5pm)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Forecast Start Hour', fontsize=10)
        
        # Plot 2: RMSE by Forecast Start Hour (for forecasts before 5pm)
        ax2 = axes[0, 1]
        before_5pm = df_approach[df_approach['forecast_start_hour'] < 17]
        scatter2 = ax2.scatter(before_5pm['forecast_start_hour'], before_5pm['daytime_rmse'], 
                   alpha=0.6, s=50, c=before_5pm['hours_until_5pm'], cmap='coolwarm')
        ax2.set_xlabel('Forecast Start Hour', fontsize=12)
        ax2.set_ylabel('RMSE (10am-6pm predictions only)', fontsize=12)
        ax2.set_title('RMSE by Forecast Start Time', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.axvspan(10, 17, alpha=0.1, color='yellow', label='Daytime Window')
        ax2.axvline(17, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5pm Target')
        ax2.legend()
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Hours Until 5pm', fontsize=10)
        
        # Plot 3: Number of Daytime Predictions per Forecast
        ax3 = axes[1, 0]
        ax3.plot(df_approach['forecast_start_hour'], df_approach['n_daytime_predictions'], 
                marker='o', markersize=4, linewidth=2)
        ax3.set_xlabel('Forecast Start Hour', fontsize=12)
        ax3.set_ylabel('# of Predictions in 10am-6pm', fontsize=12)
        ax3.set_title('Coverage of Daytime Window', fontsize=13, fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.axvspan(10, 17, alpha=0.1, color='yellow')
        
        # Plot 4: Rolling average RMSE
        ax4 = axes[1, 1]
        # Group by hour and calculate mean
        hourly_rmse = df_approach.groupby(df_approach['forecast_start_hour'].round())['daytime_rmse'].agg(['mean', 'std'])
        ax4.errorbar(hourly_rmse.index, hourly_rmse['mean'], yerr=hourly_rmse['std'], 
                    marker='o', markersize=8, capsize=5, linewidth=2, label='Mean ± Std')
        ax4.set_xlabel('Forecast Start Hour', fontsize=12)
        ax4.set_ylabel('Mean RMSE (10am-6pm predictions)', fontsize=12)
        ax4.set_title('Hourly Average RMSE', fontsize=13, fontweight='bold')
        ax4.grid(alpha=0.3)
        ax4.axvspan(10, 17, alpha=0.1, color='yellow', label='Daytime Window')
        ax4.axvline(17, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5pm Target')
        ax4.legend()
        
        plt.suptitle(f'Daytime Forecast Accuracy Analysis - Approaching 5pm\n{config["model_type"]} Model', 
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\nKey Statistics:")
        print(f"  Forecasts analyzed: {len(df_approach)}")
        print(f"  Time range: {df_approach['forecast_start'].min()} to {df_approach['forecast_start'].max()}")
        print(f"\nRMSE Summary (10am-6pm predictions only):")
        print(f"  Overall mean: {df_approach['daytime_rmse'].mean():.4f}")
        print(f"  Overall std:  {df_approach['daytime_rmse'].std():.4f}")
        
        # Compare early vs late forecasts
        early_forecasts = df_approach[df_approach['forecast_start_hour'] < 12]
        late_forecasts = df_approach[(df_approach['forecast_start_hour'] >= 12) & 
                                     (df_approach['forecast_start_hour'] < 17)]
        
        if len(early_forecasts) > 0:
            print(f"\nEarly forecasts (before noon):")
            print(f"  Mean RMSE: {early_forecasts['daytime_rmse'].mean():.4f}")
            print(f"  Std RMSE:  {early_forecasts['daytime_rmse'].std():.4f}")
        
        if len(late_forecasts) > 0:
            print(f"\nLate forecasts (noon-5pm):")
            print(f"  Mean RMSE: {late_forecasts['daytime_rmse'].mean():.4f}")
            print(f"  Std RMSE:  {late_forecasts['daytime_rmse'].std():.4f}")
        
        # Find best/worst forecast times
        best_forecast = df_approach.loc[df_approach['daytime_rmse'].idxmin()]
        worst_forecast = df_approach.loc[df_approach['daytime_rmse'].idxmax()]
        
        print(f"\nBest forecast for 10am-6pm window:")
        print(f"  Start time: {best_forecast['forecast_start']}")
        print(f"  RMSE: {best_forecast['daytime_rmse']:.4f}")
        print(f"  Hours until 5pm: {best_forecast['hours_until_5pm']:.1f}")
        
        print(f"\nWorst forecast for 10am-6pm window:")
        print(f"  Start time: {worst_forecast['forecast_start']}")
        print(f"  RMSE: {worst_forecast['daytime_rmse']:.4f}")
        print(f"  Hours until 5pm: {worst_forecast['hours_until_5pm']:.1f}")
        
    else:
        print("[WARNING] No 5pm timestamp found in the 48h window")
        df_approach = pd.DataFrame()
    
    # ========================================================================
    # SAVE TO CSV
    # ========================================================================
    csv_paths = {}
    
    if save_to_csv:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create base filename from model name
        base_name = model_name.replace('.pt', '').replace('.json', '').replace('.npz', '').replace('.pkl', '').replace('best_model_', '')
        timestamp_str = df_clean.loc[start_idx, 'Datetime'].strftime('%Y%m%d_%H%M')
        
        # ====================================================================
        # CSV 1: Combined 48h data with ALL forecasts (288 columns)
        # ====================================================================
        print(f"[INFO] Saving combined CSV with {len(preds_all)} forecast columns...")
        combined_data = {
            'datetime': times_48h,
            'ground_truth': y_full_48h,
        }
        
        # Add each forecast as a column (shifted to align with its prediction window)
        for timestep in range(len(preds_all)):
            # Create a column with NaN padding
            forecast_col = np.full(288, np.nan)
            pred = preds_all[timestep]
            end_idx = min(timestep + 144, 288)
            forecast_col[timestep:end_idx] = pred[:end_idx - timestep]
            
            # Column name shows the timestep offset
            hours_offset = timestep / 6
            combined_data[f'forecast_+{hours_offset:.2f}h'] = forecast_col
        
        df_combined = pd.DataFrame(combined_data)
        combined_path = os.path.join(output_dir, f"{base_name}_combined_10min_{timestamp_str}.csv")
        df_combined.to_csv(combined_path, index=False)
        csv_paths['combined'] = combined_path
        print(f"[SAVED] Combined 48h data → {combined_path}")
        
        # ====================================================================
        # CSV 2: Forecast metrics (one row per forecast - 288 rows)
        # ====================================================================
        print(f"[INFO] Calculating metrics for all {len(preds_all)} forecasts...")
        metrics_data = []
        for timestep in range(len(preds_all)):
            pred = preds_all[timestep]
            
            # Calculate segment
            end_idx = min(timestep + 144, 288)
            gt_segment = y_full_48h[timestep:end_idx]
            pred_segment = pred[:end_idx - timestep]
            
            if len(gt_segment) > 0 and len(pred_segment) == len(gt_segment):
                rmse = np.sqrt(mean_squared_error(gt_segment, pred_segment))
                mae = np.mean(np.abs(gt_segment - pred_segment))
            else:
                rmse = np.nan
                mae = np.nan
            
            hours_offset = timestep / 6
            
            metrics_data.append({
                'timestep': timestep,
                'forecast_offset_hours': hours_offset,
                'forecast_start_datetime': times_48h[timestep],
                'forecast_end_datetime': times_48h[end_idx - 1] if end_idx > timestep else times_48h[timestep],
                'forecast_length_timesteps': len(pred_segment),
                'rmse': rmse,
                'mae': mae,
                'mean_prediction': pred_segment.mean() if len(pred_segment) > 0 else np.nan,
                'mean_ground_truth': gt_segment.mean() if len(gt_segment) > 0 else np.nan,
                'max_prediction': pred_segment.max() if len(pred_segment) > 0 else np.nan,
                'max_ground_truth': gt_segment.max() if len(gt_segment) > 0 else np.nan,
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics_10min_{timestamp_str}.csv")
        df_metrics.to_csv(metrics_path, index=False)
        csv_paths['metrics'] = metrics_path
        print(f"[SAVED] Forecast metrics → {metrics_path}")
        
        # ====================================================================
        # CSV 3: Long-format predictions (one row per prediction point)
        # ====================================================================
        print(f"[INFO] Creating long-format dataset...")
        long_data = []
        for timestep in range(len(preds_all)):
            pred = preds_all[timestep]
            
            end_idx = min(timestep + 144, 288)
            hours_offset = timestep / 6
            
            for i in range(end_idx - timestep):
                time_idx = timestep + i
                time = times_48h[time_idx]
                pred_val = pred[i]
                gt_val = y_full_48h[time_idx]
                
                long_data.append({
                    'forecast_timestep': timestep,
                    'forecast_offset_hours': hours_offset,
                    'datetime': time,
                    'timestep_within_forecast': i,
                    'prediction': pred_val,
                    'ground_truth': gt_val,
                    'error': pred_val - gt_val,
                    'absolute_error': abs(pred_val - gt_val),
                })
        
        df_long = pd.DataFrame(long_data)
        long_path = os.path.join(output_dir, f"{base_name}_long_format_10min_{timestamp_str}.csv")
        df_long.to_csv(long_path, index=False)
        csv_paths['long_format'] = long_path
        print(f"[SAVED] Long-format data → {long_path}")
        
        # ====================================================================
        # CSV 4: Model configuration
        # ====================================================================
        config_df = pd.DataFrame([{
            'model_name': model_name,
            'model_type': config['model_type'],
            'complexity': config['model_complexity'],
            'past_hours': config['past_hours'],
            'future_hours': config['future_hours'],
            'use_pv': config['use_pv'],
            'use_hist_weather': config['use_hist_weather'],
            'use_forecast': config['use_forecast'],
            'use_ideal_nwp': config['use_ideal_nwp'],
            'use_time_encoding': config['use_time_encoding'],
            'forecast_start_datetime': times_48h[0],
            'midnight_index': midnight_index,
            'df_index': start_idx,
            'total_forecasts': len(preds_all),
            'forecast_frequency': '10 minutes',
        }])
        config_path = os.path.join(output_dir, f"{base_name}_config_10min_{timestamp_str}.csv")
        config_df.to_csv(config_path, index=False)
        csv_paths['config'] = config_path
        print(f"[SAVED] Model config → {config_path}")
        
        # ====================================================================
        # CSV 5: 5pm Approach Analysis
        # ====================================================================
        if not df_approach.empty:
            approach_path = os.path.join(output_dir, f"{base_name}_5pm_approach_analysis_{timestamp_str}.csv")
            df_approach.to_csv(approach_path, index=False)
            csv_paths['5pm_approach'] = approach_path
            print(f"[SAVED] 5pm approach analysis → {approach_path}")
        
        print(f"\n[CSV SUMMARY] Saved {len(csv_paths)} files to '{output_dir}/'")
    else:
        csv_paths = None
    
    return preds_all, y_full_48h, config, csv_paths