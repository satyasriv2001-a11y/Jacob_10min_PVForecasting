#!/usr/bin/env python3
"""
Project 186 Side Analysis
Modified from train186_with_sliding.py to generate 20 sliding forecast plots
Saves models and test splits in side_analysis/ folder
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
import pickle

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_ml import train_ml_model

# Import the sliding forecast function
from auto_sliding_forecast_with_5pm_analysis import auto_sliding_forecast


def safe_parse_datetime(col):
    """Parse datetime column with various formats"""
    s = col.copy()
    
    if np.issubdtype(s.dtype, np.number):
        s_abs = s.abs().median()
        if s_abs > 1e14:
            return pd.to_datetime(s, unit='ns')
        elif s_abs > 1e11:
            return pd.to_datetime(s, unit='ms')
        else:
            return pd.to_datetime(s, unit='s')
    
    try:
        return pd.to_datetime(s, format='mixed', utc=False)
    except:
        pass
    
    try:
        return pd.to_datetime(s, errors='coerce')
    except:
        return s


def generate_ml_configs():
    """Generate configs for XGB and Linear models"""
    configs = []
    
    base_config = {
        'model_complexity': 'high',
        'past_hours': 24,
        'future_hours': 24,
        'use_pv': True,
        'use_hist_weather': False,
        'use_forecast': True,
        'use_ideal_nwp': False,
        'use_time_encoding': False,
        'weather_category': 'all_weather',
        'start_date': '2022-01-01',
        'end_date': '2024-09-28',
        'no_hist_power': False,
        'train_params': {
            'epochs': 35,
            'batch_size': 64,
            'learning_rate': 0.001
        }
    }
    
    # XGB config
    xgb_config = base_config.copy()
    xgb_config['model'] = 'XGB'
    xgb_config['experiment_name'] = 'XGB_high_PV+NWP_24h_noTE'
    xgb_config['model_params'] = {
        'n_estimators': 75,
        'max_depth': 8,
        'learning_rate': 0.03,
        'verbosity': 0
    }
    configs.append(xgb_config)
    
    # Linear config
    linear_config = base_config.copy()
    linear_config['model'] = 'Linear'
    linear_config['experiment_name'] = 'Linear_high_PV+NWP_24h_noTE'
    linear_config['model_params'] = {}
    configs.append(linear_config)
    
    return configs


def run_sliding_forecasts(model, model_name, df_clean, config, 
                         hist_feats, fcst_feats, scaler_target,
                         n_samples=20, output_dir='sliding_forecasts'):
    """
    Run sliding forecasts on 20 midnight samples.
    Saves ONLY the combined plots, deletes CSV files.
    """
    import matplotlib
    # Set to non-interactive backend to prevent plt.show() from clearing figures
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print(f"  Running {n_samples} sliding forecast samples...")
    
    # Find test set midnights
    df_midnight = df_clean[df_clean['Datetime'].dt.hour == 0]
    midnight_indices = df_midnight.index.tolist()
    
    total_len = len(df_clean)
    test_start_idx = int(total_len * 0.9)
    test_midnights = [idx for idx in midnight_indices if idx >= test_start_idx]
    
    # Select n_samples evenly spaced
    if len(test_midnights) < n_samples:
        selected_midnights = test_midnights
    else:
        step = len(test_midnights) // n_samples
        selected_midnights = test_midnights[::step][:n_samples]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save midnight indices list
    midnight_list_path = os.path.join(output_dir, 'midnight_indices.txt')
    with open(midnight_list_path, 'w') as f:
        f.write("Midnight Indices for Sliding Forecasts\n")
        f.write("="*50 + "\n\n")
        for i, idx in enumerate(selected_midnights, 1):
            # Find position in full midnight list
            midnight_position = midnight_indices.index(idx)
            dt = df_clean.loc[idx, 'Datetime']
            f.write(f"{i:2d}. Index {idx:6d} (position {midnight_position}) - {dt}\n")
    
    # Run sliding forecast for each sample
    for i, midnight_idx in enumerate(selected_midnights, 1):
        # Find position in full midnight list
        midnight_position = midnight_indices.index(midnight_idx)
        
        try:
            sample_dir = os.path.join(output_dir, f'sample_{i:02d}_midnight_{midnight_idx}')
            os.makedirs(sample_dir, exist_ok=True)
            
            print(f"    Sample {i}/{len(selected_midnights)}: midnight index {midnight_idx}")
            
            # NOTE: auto_sliding_forecast generates plots but doesn't save them!
            # We need to modify it or capture the plots manually
            # For now, we'll save CSVs and generate our own plot
            
            # Run sliding forecast - generates data
            preds, gt, cfg, paths = auto_sliding_forecast(
                model=model,
                model_name=model_name,
                df_raw=df_clean,
                midnight_index=midnight_position,
                show_individual_plots=False,
                save_to_csv=True,
                output_dir=sample_dir,
                plot_every_n=24
            )
            
            # The plots were generated but not saved by auto_sliding_forecast
            # Check if any PNG files exist
            png_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
            
            if len(png_files) == 0:
                print(f"      ⚠ No plots saved by auto_sliding_forecast, generating manually...")
                
                # Load the combined CSV and regenerate the plot
                combined_csv = paths.get('combined')
                if combined_csv and os.path.exists(combined_csv):
                    df_combined = pd.read_csv(combined_csv)
                    
                    # Create the combined sliding plot
                    fig, ax = plt.subplots(figsize=(20, 10))
                    
                    # Create x-axis as timesteps (0 to 287)
                    timesteps = np.arange(len(df_combined))
                    
                    # Plot ground truth
                    ax.plot(timesteps, df_combined['ground_truth'],
                           color='black', linewidth=2.5, marker='o', markersize=3,
                           label='Ground Truth', zorder=10)
                    
                    # Plot every 24th forecast (every 4 hours)
                    forecast_cols = [col for col in df_combined.columns if col.startswith('forecast_+')]
                    for j, col in enumerate(forecast_cols[::24]):  # Every 24th
                        ax.plot(timesteps, df_combined[col],
                               linewidth=1.2, marker='o', markersize=2, alpha=0.5,
                               label=col.replace('forecast_+', '+').replace('h', 'h'))
                    
                    ax.grid(alpha=0.3)
                    ax.set_xlabel('Timestep (10-minute intervals)', fontsize=12)
                    ax.set_ylabel('Capacity Factor', fontsize=12)
                    ax.set_title(f'Sliding Forecasts - Midnight Index {midnight_idx}', fontsize=14, fontweight='bold')
                    
                    # Set x-ticks every 30 timesteps (every 5 hours)
                    tick_spacing = 30
                    ax.set_xticks(np.arange(0, len(timesteps), tick_spacing))
                    ax.set_xticklabels([f'{i}' for i in range(0, len(timesteps), tick_spacing)])
                    
                    # Add legend (but limit to avoid clutter)
                    handles, labels = ax.get_legend_handles_labels()
                    if len(handles) > 15:
                        # Only show first 15 items in legend
                        ax.legend(handles[:15], labels[:15], ncol=3, fontsize=8, loc='upper right')
                    else:
                        ax.legend(ncol=3, fontsize=8, loc='upper right')
                    
                    plt.tight_layout()
                    
                    # Save the plot
                    plot_path = os.path.join(sample_dir, f'{model_name}_combined_plot_midnight_{midnight_idx}.png')
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"      ✓ Saved plot: {os.path.basename(plot_path)}")
            else:
                print(f"      ✓ Found existing plot: {png_files[0]}")
            
            # Keep ONLY the plot, delete all CSVs
            files_to_remove = ['combined', 'metrics', 'long_format', 'config', '5pm_approach']
            for key in files_to_remove:
                if key in paths and os.path.exists(paths[key]):
                    try:
                        os.remove(paths[key])
                    except:
                        pass
            
            if i % 5 == 0:
                print(f"    ✓ Completed {i}/{len(selected_midnights)} samples")
            
        except Exception as e:
            print(f"    ✗ Sample {i} (midnight {midnight_idx}) failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"    ✓ All {len(selected_midnights)} plots saved, CSVs deleted")
    return selected_midnights


def main():
    """Main training pipeline for Project 186 with 20 forecast plots"""
    
    print("="*70)
    print("PROJECT 186 SIDE ANALYSIS - 20 FORECAST PLOTS")
    print("="*70)
    
    # Configuration - ONLY PROJECT 186
    csv_path = "/content/drive/MyDrive/Solar PV electricity/data/Project1140.csv"
    project_id = 186
    n_plot_samples = 20
    output_base_dir = "side_analysis"  # Separate folder
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_base_dir, f"project186_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n[INFO] CSV: {csv_path}")
    print(f"[INFO] Output: {run_dir}")
    print(f"[INFO] Forecast plots per model: {n_plot_samples}\n")
    
    # Load data
    print("[STEP 1] Loading data...")
    df = pd.read_csv(csv_path)
    df['Datetime'] = safe_parse_datetime(df['DateTime'])
    df['Datetime'] = df['Datetime'].dt.tz_localize(None)
    
    # Filter from 2022-01-01
    start_date = pd.Timestamp('2022-01-01')
    df = df[df['Datetime'] >= start_date].reset_index(drop=True)
    
    print(f"  ✓ Loaded {len(df)} rows")
    print(f"  ✓ Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    
    # Generate configs
    configs = generate_ml_configs()
    print(f"\n[STEP 2] Training {len(configs)} models")
    
    # Train each model
    for idx, config in enumerate(configs, 1):
        model_name = config['model']
        
        print("\n" + "="*70)
        print(f"[MODEL {idx}/{len(configs)}] {model_name}")
        print("="*70)
        
        try:
            # Preprocess
            print("\n[STEP 3] Preprocessing...")
            df_clean, hist_feats, fcst_feats, sh, sf, st, no_hist = \
                preprocess_features(df.copy(), config)
            
            print(f"  ✓ Historical features: {len(hist_feats)}")
            print(f"  ✓ Forecast features: {len(fcst_feats)}")
            
            # Create windows
            print("\n[STEP 4] Creating sliding windows...")
            X_hist, X_fcst, y, hours, dates = create_sliding_windows(
                df_clean, config['past_hours'], config['future_hours'],
                hist_feats, fcst_feats, no_hist
            )
            
            print(f"  ✓ Total samples: {len(X_hist)}")
            
            # Split data
            total = len(X_hist)
            train_end = int(total * 0.8)
            test_start = int(total * 0.9)
            
            Xh_train = X_hist[:train_end]
            Xf_train = X_fcst[:train_end] if X_fcst is not None else None
            y_train = y[:train_end]
            dates_train = [dates[i] for i in range(train_end)]
            
            Xh_test = X_hist[test_start:]
            Xf_test = X_fcst[test_start:] if X_fcst is not None else None
            y_test = y[test_start:]
            dates_test = [dates[i] for i in range(test_start, total)]
            
            print(f"  ✓ Train: {len(Xh_train)}, Test: {len(Xh_test)}")
            
            # Create model directory
            model_dir = os.path.join(run_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Train model
            print("\n[STEP 5] Training model...")
            model, metrics = train_ml_model(
                config=config,
                Xh_train=Xh_train,
                Xf_train=Xf_train,
                y_train=y_train,
                Xh_test=Xh_test,
                Xf_test=Xf_test,
                y_test=y_test,
                dates_test=dates_test,
                scaler_target=st,
                save_model=False,
                project_id=project_id
            )
            
            print(f"\n  ✓ Test RMSE: {metrics['rmse']:.4f}")
            print(f"  ✓ Daytime RMSE: {metrics['daytime_rmse']:.4f}")
            
            # Save model
            print("\n[STEP 6] Saving model...")
            if model_name == 'XGB':
                model_path = os.path.join(model_dir, f'{project_id}_XGB_model.pkl')
                # XGB is wrapped in MultiOutputRegressor, save with pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            else:
                model_path = os.path.join(model_dir, f'{project_id}_Linear_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            print(f"  ✓ {model_path}")
            
            # Save scalers
            scaler_path = os.path.join(model_dir, 'scalers.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump({'scaler_hist': sh, 'scaler_fcst': sf, 'scaler_target': st}, f)
            
            # Save config (needed by auto_sliding_forecast)
            config_path = os.path.join(model_dir, 'config.pkl')
            with open(config_path, 'wb') as f:
                pickle.dump({
                    'config': config,
                    'hist_feats': hist_feats,
                    'fcst_feats': fcst_feats
                }, f)
            print(f"  ✓ Saved config and features")
            
            # Save split info
            split_path = os.path.join(model_dir, 'train_test_split.pkl')
            with open(split_path, 'wb') as f:
                pickle.dump({
                    'train_indices': list(range(train_end)),
                    'test_indices': list(range(test_start, total)),
                    'dates_train': dates_train,
                    'dates_test': dates_test
                }, f)
            
            # Save test data
            np.savez_compressed(
                os.path.join(model_dir, 'test_data.npz'),
                Xh_test=Xh_test,
                Xf_test=Xf_test if Xf_test is not None else np.array([]),
                y_test=y_test
            )
            
            # Generate sliding forecasts (288 forecasts per sample)
            print(f"\n[STEP 7] Generating {n_plot_samples} sliding forecast samples...")
            print("  (Each sample contains 288 forecasts - one every 10 minutes)")
            
            # Save a config file that auto_sliding_forecast expects
            # It will try to load this based on model_name
            model_config_dir = os.path.join('models', model_name)
            os.makedirs(model_config_dir, exist_ok=True)
            
            # Save model in expected location
            temp_model_path = os.path.join(model_config_dir, f'{project_id}_{config["experiment_name"]}.pkl')
            with open(temp_model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save config in expected format
            temp_config_path = os.path.join(model_config_dir, f'{project_id}_{config["experiment_name"]}_config.pkl')
            with open(temp_config_path, 'wb') as f:
                pickle.dump({
                    'config': config,
                    'hist_feats': hist_feats,
                    'fcst_feats': fcst_feats,
                    'scaler_target': st
                }, f)
            
            sliding_dir = os.path.join(model_dir, 'sliding_forecasts')
            midnight_indices = run_sliding_forecasts(
                model=model,
                model_name=f'{project_id}_{config["experiment_name"]}',
                df_clean=df_clean,
                config=config,
                hist_feats=hist_feats,
                fcst_feats=fcst_feats,
                scaler_target=st,
                n_samples=n_plot_samples,
                output_dir=sliding_dir
            )
            
            # Clean up temp files
            import shutil
            if os.path.exists(model_config_dir):
                shutil.rmtree(model_config_dir)
            
            # Save metrics
            with open(os.path.join(model_dir, 'metrics.txt'), 'w') as f:
                f.write(f"PROJECT 186 - {model_name}\n" + "="*70 + "\n\n")
                f.write(f"TEST RMSE:         {metrics['rmse']:.4f}\n")
                f.write(f"TEST R²:           {metrics['r2']:.4f}\n")
                f.write(f"DAYTIME RMSE:      {metrics['daytime_rmse']:.4f}\n")
                f.write(f"DAYTIME R²:        {metrics['daytime_r2']:.4f}\n")
            
            print(f"\n✓ {model_name} complete!")
            
        except Exception as e:
            print(f"\n✗ {model_name} failed: {str(e)}")
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print(f"Results: {run_dir}")
    print("="*70)


if __name__ == "__main__":
    main()