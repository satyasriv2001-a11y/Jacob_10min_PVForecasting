#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run 160 deep learning experiments and record metrics to CSV.
This version includes **extensive progress logging in English**.
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
import time
import traceback
from datetime import datetime
import warnings
from tqdm import tqdm
from train.train_ml import train_ml_model
warnings.filterwarnings('ignore')

# Ensure working directory is correct
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

print(f"[INFO] Working directory set to: {os.getcwd()}")

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model

ML_MODELS = ['RF', 'XGB', 'LGBM', 'Linear']
DL_MODELS = ['LSTM', 'GRU', 'Transformer', 'TCN']


def generate_all_configs():
    """Generate list of all experiment configurations (160 total)."""
    print("[INFO] Generating all experiment configurations...")

    configs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "Project1140.csv")

    models = ['LSTM', 'GRU', 'Transformer', 'TCN', 'RF', 'XGB', 'LGBM', 'Linear']
    complexities = ['low', 'high']

    # PV-based experiments (128 runs)
    lookbacks = [24, 72]
    feature_combos_pv = [
        {'name': 'PV', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+HW', 'use_pv': True, 'use_hist_weather': True, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+NWP', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'PV+NWP+', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]
    te_options = [True, False]

    for model in models:
        for complexity in complexities:
            for lookback in lookbacks:
                for feat_combo in feature_combos_pv:
                    for use_te in te_options:
                        configs.append(create_config(
                            data_path, model, complexity, lookback,
                            feat_combo, use_te, is_nwp_only=False
                        ))

    # NWP-only experiments (32 runs)
    feature_combos_nwp = [
        {'name': 'NWP', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'NWP+', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]

    for model in models:
        for complexity in complexities:
            for feat_combo in feature_combos_nwp:
                for use_te in te_options:
                    configs.append(create_config(
                        data_path, model, complexity, 0,
                        feat_combo, use_te, is_nwp_only=True
                    ))

    print(f"[INFO] Total configurations generated: {len(configs)}\n")
    return configs


def create_config(data_path, model, complexity, lookback, feat_combo, use_te, is_nwp_only):
    config = {
        'data_path': data_path,
        'model': model,
        'model_complexity': complexity,
        'use_pv': feat_combo['use_pv'],
        'use_hist_weather': feat_combo['use_hist_weather'],
        'use_forecast': feat_combo['use_forecast'],
        'use_ideal_nwp': feat_combo['use_ideal_nwp'],
        'use_time_encoding': use_te,
        'weather_category': 'all_weather',
        'future_hours': 24,
        'start_date': '2022-01-01',
        'end_date': '2024-09-28',
        'save_options': {}
    }

    if is_nwp_only:
        config['past_hours'] = 0
        config['no_hist_power'] = True
        feat_name = feat_combo['name']
    else:
        config['past_hours'] = lookback
        config['no_hist_power'] = False
        feat_name = f"{feat_combo['name']}_{lookback}h"

    # ORIGINAL complexity-based model parameter defs
    if complexity == 'low':
        config['train_params'] = {'epochs': 20, 'batch_size': 64, 'learning_rate': 0.001}
        config['model_params'] = {
            'd_model': 32,
            'hidden_dim': 16,
            'num_heads': 2,
            'num_layers': 1,
            'dropout': 0.1,
            'tcn_channels': [16, 32],
            'kernel_size': 3
        }
    else:  # high
        config['train_params'] = {'epochs': 35, 'batch_size': 64, 'learning_rate': 0.001}
        config['model_params'] = {
            'd_model': 64,
            'hidden_dim': 32,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'tcn_channels': [32, 64],
            'kernel_size': 3
        }

    te_suffix = 'TE' if use_te else 'noTE'
    config['experiment_name'] = f"{model}_{complexity}_{feat_name}_{te_suffix}"
    config['save_dir'] = f"results/{config['experiment_name']}"

    return config



def run_all_experiments():
    """Main routine that runs all deep learning experiments."""
    print("=" * 100)
    print("[START] Running 160 solar forecasting experiments")
    print("=" * 100)

    all_configs = generate_all_configs()

    print("\n[INFO] Loading dataset once before experiment loop...")
    df = pd.read_csv(os.path.join(script_dir, "data", "Project1140.csv"))
    print(f"[INFO] Loaded {len(df):,} rows from CSV")

    # Convert timestamp column if needed
    print("[INFO] Converting 'date' column to datetime...")
    df['Datetime'] = pd.to_datetime(df['date'])
    print(f"[INFO] Date range in raw data: {df['Datetime'].min()} → {df['Datetime'].max()}")

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"all_experiments_results_{timestamp}.csv"
    pd.DataFrame(columns=['experiment_name', 'model', 'mae', 'rmse', 'r2']).to_csv(output_file, index=False)

    for idx, config in enumerate(all_configs, 1):
        print("\n" + "=" * 100)
        print(f"[RUNNING] Experiment {idx}/{len(all_configs)}: {config['experiment_name']}")
        print("=" * 100)

                # ==============================================================
        # SKIP all models except TCN and ML (RF, XGB, LGBM, Linear)
        # ==============================================================
        allowed_models = ['TCN','RF', 'XGB', 'LGBM', 'Linear']
        if config['model'] not in allowed_models:
            print(f"[SKIP] Model {config['model']} is not TCN or ML → skipping experiment.\n")
            continue


        try:
            print("[STEP] Preprocessing data (feature selection, scaling, time filters)...")
            df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)

            print("[STEP] Creating sliding windows...")
            X_hist, X_fcst, y, hours, dates = create_sliding_windows(
                df_clean, config['past_hours'], config['future_hours'],
                hist_feats, fcst_feats, no_hist_power
            )
            print(f"[INFO] Windowed dataset shapes → X_hist: {X_hist.shape}, y: {y.shape}")
            print("\n--- Window Debug Info ---")
            print(f"Hist feats:      {hist_feats}")
            print(f"Fcst feats:      {fcst_feats}")
            print(f"past_hours*6:    {config['past_hours']*6}")
            print(f"future_hours*6:  {config['future_hours']*6}")
            print(f"X_hist.shape:    {X_hist.shape}")
            if X_fcst is not None:
                print(f"X_fcst.shape:    {X_fcst.shape}")
            else:
                print("X_fcst:          None")
            print(f"y.shape:         {y.shape}")
            print("--------------------------\n")

            print("[STEP] Splitting train/val/test sets...")
            total_samples = len(X_hist)
            # ✅ Hard-code fallback defaults
            train_ratio = config.get('train_ratio', 0.8)
            val_ratio   = config.get('val_ratio', 0.1)
            test_ratio  = config.get('test_ratio', 0.1)

            train_size = int(total_samples * train_ratio)
            val_size = int(total_samples * val_ratio)
            # test is implicit


            train_idx = np.arange(train_size)
            val_idx = np.arange(train_size, train_size + val_size)
            test_idx = np.arange(train_size + val_size, total_samples)

        


            # SAFE handling of missing X_fcst
            if X_fcst is None:
                X_fcst_train = None
                X_fcst_val   = None
                X_fcst_test  = None
            else:
                X_fcst_train = X_fcst[train_idx]
                X_fcst_val   = X_fcst[val_idx]
                X_fcst_test  = X_fcst[test_idx]


            y_train = y[train_idx]
            y_val   = y[val_idx]
            y_test  = y[test_idx]
            
            test_dates = [dates[i] for i in test_idx]



            # Extract hour arrays — one per sample, aligned with indices
            train_hours = np.array([hours[i] for i in train_idx])
            val_hours   = np.array([hours[i] for i in val_idx])
            test_hours  = np.array([hours[i] for i in test_idx])

            X_hist_train = X_hist[train_idx]
            X_hist_val   = X_hist[val_idx]
            X_hist_test = X_hist[test_idx]

            train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
            val_data   = (X_hist_val,   X_fcst_val,   y_val,   val_hours,   [])
            test_data  = (X_hist_test,  X_fcst_test,  y_test,  test_hours,  test_dates)

            

            print("[STEP] Training model...")

            if config['model'] in ML_MODELS:
                print(f"[INFO] Using ML trainer for {config['model']}")

                # SAFE handling of missing X_fcst

                model, metrics = train_ml_model(
                    config,
                    X_hist_train, X_fcst_train,
                    y_train,
                    X_hist_test, X_fcst_test,
                    y_test,
                    test_dates,
                    scaler_target
                )

            elif config['model'] in DL_MODELS:
                print(f"[INFO] Using DL trainer for {config['model']}")
                model, metrics = train_dl_model(
                    config,
                    train_data, val_data, test_data,
                    (scaler_hist, scaler_fcst, scaler_target)
                )

            else:
                print(f"[WARN] Unknown model {config['model']} — skipping.")
                continue


            print(f"[RESULT] MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

            pd.DataFrame([{
                'experiment_name': config['experiment_name'],
                'model': config['model'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2']
            }]).to_csv(output_file, mode='a', header=False, index=False)

        # except Exception as e:
        #     print(f"[ERROR] Experiment failed → {config['experiment_name']}")
        #     print(f"[ERROR] Reason: {str(e)}")
        #     pd.DataFrame([{
        #         'experiment_name': config['experiment_name'],
        #         'model': config['model'],
        #         'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        #     }]).to_csv(output_file, mode='a', header=False, index=False)
        #     continue

        except Exception as e:
            print("\n" + "="*100)
            print(f"[ERROR] Experiment failed → {config['experiment_name']}")
            print(f"[ERROR] Reason: {str(e)}")
            print("[TRACEBACK]")
            traceback.print_exc()   # ✅ full traceback printed
            print("="*100 + "\n")

            # Record failure in CSV (still useful)
            pd.DataFrame([{
                'experiment_name': config['experiment_name'],
                'model': config['model'],
                'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
            }]).to_csv(output_file, mode='a', header=False, index=False)

            # ✅ OPTION A: Pause and wait for user input before continuing
            input("Press ENTER to continue to next experiment...")

            continue

    print("\n" + "=" * 100)
    print("[COMPLETE] All experiments finished!")
    print(f"[OUTPUT] Results saved to: {output_file}")
    print("=" * 100)

    return True


if __name__ == "__main__":
    success = run_all_experiments()
    if success:
        print("\n[SUCCESS] 所有160次实验完成！")
    else:
        print("\n[FAILED] 实验失败！")
        sys.exit(1)
