#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized ML Experiment Runner
Single Configuration: high_PV+NWP_24h_noTE
Models: XGB + Linear Regression
Features:
- Processes all CSVs in specified directory
- Saves results incrementally to Google Drive
- Auto-resume from last completed experiment
- Saves ALL trained models (format: projectID_model_experimentname)
- XGBoost saved as JSON, Linear Regression as compressed numpy arrays
- Extract project IDs from CSV filenames (ProjectXXXX.csv format)
"""

import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==== CONFIGURATION ====
CSV_DIR = "/content/drive/MyDrive/Solar PV electricity/data"
RESULTS_DIR = "/content/drive/MyDrive/TimeBasedResults"
RESULTS_FILENAME = "400run.csv"
MODELS_DIR = "/content/drive/MyDrive/SavedModels"  # Models saved here

# Ensure working directory is correct
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

print(f"[INFO] Working directory: {os.getcwd()}")
print(f"[INFO] CSV directory: {CSV_DIR}")
print(f"[INFO] Results directory: {RESULTS_DIR}")
print(f"[INFO] Models directory: {MODELS_DIR}")

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_ml import train_ml_model


# ---------------------------------------------------------
# CONFIG GENERATION - SINGLE OPTIMIZED CONFIG
# ---------------------------------------------------------
def generate_ml_configs():
    """
    Generate configs for ONLY:
    - Models: XGB, Linear
    - Config: high_PV+NWP_noTE
    - Lookback: 24h
    """
    configs = []
    models = ['XGB']
    
    # Model-specific parameters for HIGH complexity
    DEFAULT_ML_PARAMS = {
        'XGB': {
            'n_estimators': 400, 
            'max_depth': 8,  
            'learning_rate': 0.03, 
            'verbosity': 0, 
            'random_state': 42
        },
        'Linear': {}  # No hyperparameters needed
    }
    
    # Dummy train_params (required by pipeline but not used for ML)
    train_params = {"epochs": 1, "batch_size": 64, "learning_rate": 0.001}
    
    # Single feature combo: PV+NWP
    for model in models:
        cfg = {
            'model': model,
            'model_complexity': 'high',
            'past_hours': 24,  # 24h lookback
            'future_hours': 24,
            'use_pv': True,
            'use_hist_weather': False,
            'use_forecast': True,
            'use_ideal_nwp': False,
            'use_time_encoding': False,  # noTE
            'weather_category': 'all_weather',
            'start_date': '2022-01-01',
            'end_date': '2024-09-28',
            'experiment_name': f"{model}_high_PV+NWP_24h_noTE",
            'train_params': train_params,
            'model_params': DEFAULT_ML_PARAMS[model],
        }
        configs.append(cfg)
    
    logger.info(f"[CONFIG] Generated {len(configs)} ML configurations (high_PV+NWP_24h_noTE only)")
    return configs


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_project_id(csv_filename):
    """Extract project ID from CSV filename."""
    try:
        name = csv_filename.replace('.csv', '').replace('Project', '')
        import re
        match = re.match(r'^(\d+)', name)
        if match:
            return int(match.group(1))
        return int(name)
    except:
        logger.warning(f"Could not extract project_id from '{csv_filename}', using hash")
        return abs(hash(csv_filename)) % 100000


def safe_parse_datetime(col):
    """Parse datetime column with multiple format handling."""
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


# ---------------------------------------------------------
# RESUME FUNCTIONALITY
# ---------------------------------------------------------
def get_output_filepath():
    """Generate output file path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, RESULTS_FILENAME)


def get_last_completed_experiment(output_file):
    """Determine last completed experiment number."""
    if not os.path.exists(output_file):
        return 0
    
    try:
        df = pd.read_csv(output_file)
        if len(df) == 0:
            return 0
        completed = df[df['mae'].notna()]['global_exp_num'].max()
        return int(completed) if not pd.isna(completed) else 0
    except Exception as e:
        logger.warning(f"Could not read existing results: {e}")
        return 0


def initialize_results_file(output_file):
    """Create results file with headers."""
    if not os.path.exists(output_file):
        pd.DataFrame(columns=[
            'global_exp_num', 'project_id', 'csv_file', 'csv_index', 'config_index',
            'experiment_name', 'model', 
            'mae', 'rmse', 'r2', 'nrmse', 'smape',
            'daytime_mae', 'daytime_rmse', 'daytime_r2', 'daytime_nrmse', 'daytime_smape', 'daytime_samples',
            'forecast_mean_daytime_rmse', 'forecast_median_daytime_rmse', 'forecast_samples_collected',
            'model_saved', 'timestamp'
        ]).to_csv(output_file, index=False)
        logger.info(f"Created new results file: {output_file}")
    else:
        logger.info(f"Using existing results file: {output_file}")


def save_result(output_file, result_dict):
    """Append result to CSV."""
    pd.DataFrame([result_dict]).to_csv(output_file, mode='a', header=False, index=False)
    logger.info(f"Saved experiment {result_dict['global_exp_num']} → {output_file}")


# ---------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------
def run_ml_experiments_multi_file(start_from_exp=None):
    """
    Run XGB and Linear Regression experiments across all CSVs.
    Config: high_PV+NWP_24h_noTE only
    Saves ALL models with format: projectID_model_experimentname
    """
    print("=" * 110)
    print("[START] XGB + Linear Regression: high_PV+NWP_24h_noTE across all CSVs")
    print("=" * 110)

    # Setup
    csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith(".csv")])
    logger.info(f"Found {len(csv_files)} CSV files")
    
    configs = generate_ml_configs()
    total_csvs = len(csv_files)
    total_configs = len(configs)  # Should be 2 (XGB + Linear)
    total_global_exps = total_csvs * total_configs

    logger.info(f"Total CSVs: {total_csvs}")
    logger.info(f"Configs per CSV: {total_configs}")
    logger.info(f"Total Experiments: {total_global_exps}")

    # Initialize results
    output_file = get_output_filepath()
    initialize_results_file(output_file)

    # Determine starting point
    if start_from_exp is None:
        last_completed = get_last_completed_experiment(output_file)
        start_exp = last_completed + 1
        logger.info(f"Auto-resume: Starting from experiment {start_exp}")
    else:
        start_exp = start_from_exp
        logger.info(f"Manual-resume: Starting from experiment {start_exp}")

    if start_exp > total_global_exps:
        logger.info("All experiments already completed!")
        return True

    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Process experiments
    current_csv_file = None
    df = None
    
    # Cache preprocessed data per CSV
    cached_data = {
        'csv_file': None,
        'df_clean': None,
        'hist_feats': None,
        'fcst_feats': None,
        'scaler_hist': None,
        'scaler_fcst': None,
        'scaler_target': None,
        'no_hist_power': None,
        'X_hist': None,
        'X_fcst': None,
        'y': None,
        'hours': None,
        'dates': None
    }
    
    for global_exp_num in range(start_exp, total_global_exps + 1):
        csv_index = (global_exp_num - 1) // total_configs
        config_index = (global_exp_num - 1) % total_configs
        
        csv_file = csv_files[csv_index]
        config = configs[config_index]
        project_id = extract_project_id(csv_file)
        
        # Save ALL models (not just first CSV)
        save_models = False
        
        original_exp_name = config['experiment_name']
        # Format: projectID_model_experimentname
        prefixed_exp_name = f"{project_id}_{original_exp_name}"
        
        print("\n" + "=" * 110)
        print(f"[EXPERIMENT {global_exp_num}/{total_global_exps}]")
        print(f"[PROJECT_ID: {project_id}] [CSV {csv_index + 1}/{total_csvs}] {csv_file}")
        print(f"[CONFIG {config_index + 1}/{total_configs}] {prefixed_exp_name}")
        print("[MODEL SAVING: ENABLED]")
        print("=" * 110)

        result_dict = {
            'global_exp_num': global_exp_num,
            'project_id': project_id,
            'csv_file': csv_file,
            'csv_index': csv_index + 1,
            'config_index': config_index + 1,
            'experiment_name': prefixed_exp_name,
            'model': config['model'],
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'nrmse': np.nan,
            'smape': np.nan,
            'daytime_mae': np.nan,
            'daytime_rmse': np.nan,
            'daytime_r2': np.nan,
            'daytime_nrmse': np.nan,
            'daytime_smape': np.nan,
            'daytime_samples': 0,
            'forecast_mean_daytime_rmse': np.nan,
            'forecast_median_daytime_rmse': np.nan,
            'forecast_samples_collected': 0,
            'model_saved': False,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            # Load and preprocess CSV only when switching files OR first config
            if csv_file != cached_data['csv_file']:
                logger.info(f"\n[DATA PREPROCESSING] New CSV detected - preprocessing data...")
                
                # Load CSV
                if csv_file != current_csv_file:
                    full_path = os.path.join(CSV_DIR, csv_file)
                    df = pd.read_csv(full_path)
                    df['Datetime'] = safe_parse_datetime(df['DateTime'])
                    df['Datetime'] = df['Datetime'].dt.tz_localize(None)
                    current_csv_file = csv_file
                    logger.info(f"[INFO] Loaded CSV: {csv_file}")
                    logger.info(f"[INFO] Date range: {df['Datetime'].min()} → {df['Datetime'].max()}")

                # Preprocess (same config for both models)
                df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = \
                    preprocess_features(df.copy(), config)

                X_hist, X_fcst, y, hours, dates = create_sliding_windows(
                    df_clean, config['past_hours'], config['future_hours'],
                    hist_feats, fcst_feats, no_hist_power
                )
                
                # Cache for next model on same CSV
                cached_data = {
                    'csv_file': csv_file,
                    'df_clean': df_clean,
                    'hist_feats': hist_feats,
                    'fcst_feats': fcst_feats,
                    'scaler_hist': scaler_hist,
                    'scaler_fcst': scaler_fcst,
                    'scaler_target': scaler_target,
                    'no_hist_power': no_hist_power,
                    'X_hist': X_hist,
                    'X_fcst': X_fcst,
                    'y': y,
                    'hours': hours,
                    'dates': dates
                }
                
                logger.info(f"[DATA PREPROCESSING] Complete - cached for subsequent models")
            else:
                logger.info(f"\n[DATA CACHE] Using cached data from previous model")
                # Reuse cached data
                df_clean = cached_data['df_clean']
                hist_feats = cached_data['hist_feats']
                fcst_feats = cached_data['fcst_feats']
                scaler_hist = cached_data['scaler_hist']
                scaler_fcst = cached_data['scaler_fcst']
                scaler_target = cached_data['scaler_target']
                no_hist_power = cached_data['no_hist_power']
                X_hist = cached_data['X_hist']
                X_fcst = cached_data['X_fcst']
                y = cached_data['y']
                hours = cached_data['hours']
                dates = cached_data['dates']

            # Split data
            total_samples = len(X_hist)
            tr = int(total_samples * 0.8)
            va = int(total_samples * 0.9)

            train_idx = np.arange(0, tr)
            test_idx = np.arange(va, total_samples)

            # Enable forecast sampling only for first CSV to save time
            enable_forecast_sampling = (csv_index == 0)

            # Train ML model
            model, metrics = train_ml_model(
                config,
                X_hist[train_idx],
                None if X_fcst is None else X_fcst[train_idx],
                y[train_idx],
                X_hist[test_idx],
                None if X_fcst is None else X_fcst[test_idx],
                y[test_idx],
                [dates[i] for i in test_idx],
                scaler_target=scaler_target,
                save_model=save_models,
                project_id=project_id,
                collect_forecast_samples=enable_forecast_sampling,
                df_clean=df_clean,
                hist_feats=hist_feats,
                fcst_feats=fcst_feats,
                n_forecast_samples=5
            )

            # Update results
            result_dict['mae'] = metrics['mae']
            result_dict['rmse'] = metrics['rmse']
            result_dict['r2'] = metrics['r2']
            result_dict['nrmse'] = metrics.get('nrmse', np.nan)
            result_dict['smape'] = metrics.get('smape', np.nan)
            result_dict['daytime_mae'] = metrics.get('daytime_mae', np.nan)
            result_dict['daytime_rmse'] = metrics.get('daytime_rmse', np.nan)
            result_dict['daytime_r2'] = metrics.get('daytime_r2', np.nan)
            result_dict['daytime_nrmse'] = metrics.get('daytime_nrmse', np.nan)
            result_dict['daytime_smape'] = metrics.get('daytime_smape', np.nan)
            result_dict['daytime_samples'] = metrics.get('daytime_samples', 0)
            result_dict['forecast_mean_daytime_rmse'] = metrics.get('forecast_mean_daytime_rmse', np.nan)
            result_dict['forecast_median_daytime_rmse'] = metrics.get('forecast_median_daytime_rmse', np.nan)
            result_dict['forecast_samples_collected'] = metrics.get('forecast_samples_collected', 0)
            result_dict['model_saved'] = save_models

            logger.info(f"SUCCESS: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            logger.info(f"DAYTIME: MAE={metrics.get('daytime_mae', 0):.4f}, RMSE={metrics.get('daytime_rmse', 0):.4f}, R²={metrics.get('daytime_r2', 0):.4f}")
            if metrics.get('forecast_samples_collected', 0) > 0:
                logger.info(f"FORECAST SAMPLING: Mean RMSE={metrics.get('forecast_mean_daytime_rmse', 0):.4f}, Samples={metrics.get('forecast_samples_collected', 0)}")

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            traceback.print_exc()

        # Save result
        save_result(output_file, result_dict)
        
        progress_pct = (global_exp_num / total_global_exps) * 100
        logger.info(f"Progress: {global_exp_num}/{total_global_exps} ({progress_pct:.1f}%)")

    print("\n" + "=" * 110)
    print("[COMPLETE] All XGB + Linear experiments finished!")
    print(f"[OUTPUT] Results: {output_file}")
    print(f"[OUTPUT] Models: {MODELS_DIR}")
    print(f"[INFO] Total models saved: {total_global_exps}")
    print("=" * 110)

    return True


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML experiments (XGB + Linear only)')
    parser.add_argument('--start-from', type=int, default=None,
                       help='Start from specific experiment number (default: auto-resume)')
    
    args = parser.parse_args()
    
    run_ml_experiments_multi_file(start_from_exp=args.start_from)