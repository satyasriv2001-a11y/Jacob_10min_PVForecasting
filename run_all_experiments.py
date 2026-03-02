#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#run_all_experiments.py
"""
Run 160 deep learning experiments and record metrics to CSV.
This version includes **extensive progress logging in English**.
"""

from matplotlib.pyplot import sca
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
warnings.filterwarnings('ignore')

from rich.logging import RichHandler
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(markup=True)]
)

logger = logging.getLogger(__name__)


# Ensure working directory is correct
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

logger.info(f"[INFO] Working directory set to: {os.getcwd()}")

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model
from plot.sliding_forecast_plotter import plot_sliding_forecast_samples
from plot.debug_sliding_plots import run_debug_for_test_samples




def generate_all_configs():
    """Generate list of all experiment configurations (160 total)."""
    logger.info("[INFO] Generating all experiment configurations...")

    configs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "Project1140.csv")

    models = ['LSTM', 'GRU', 'Transformer', 'MultiChannelLSTM','TCN', 'RF', 'XGB', 'LGBM', 'Linear']
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

    logger.info(f"[INFO] Total configurations generated: {len(configs)}\n")
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
    logger.info("=" * 100)
    logger.info("[START] Running 160 solar forecasting experiments")
    logger.info("=" * 100)

    all_configs = generate_all_configs()

    logger.info("\n[INFO] Loading dataset once before experiment loop...")
    df = pd.read_csv(os.path.join(script_dir, "data", "Project1140.csv"))
    logger.info(f"[INFO] Loaded {len(df):,} rows from CSV")

    # Convert timestamp column if needed
    logger.info("[INFO] Converting 'date' column to datetime...")
    df['Datetime'] = pd.to_datetime(df['date'])
    logger.info(f"[INFO] Date range in raw data: {df['Datetime'].min()} → {df['Datetime'].max()}")

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"all_experiments_results_{timestamp}.csv"
    pd.DataFrame(columns=['experiment_name', 'model', 'mae', 'rmse', 'r2']).to_csv(output_file, index=False)

    lstm_configs = [cfg for cfg in all_configs if cfg["model"] == "LSTM"]
    tcn_configs = [cfg for cfg in all_configs if cfg["model"] == "TCN"]
    # print(len(tcn_configs))
    multi_configs = [cfg for cfg in all_configs if cfg["model"] == "MultiChannelLSTM"]
    # print(len(multi_configs))
    transformer_configs = [cfg for cfg in all_configs if cfg["model"] == "Transformer"]
    gru_configs = [cfg for cfg in all_configs if cfg["model"] == "GRU"]



    for idx, config in enumerate(all_configs, 1):
        if config["model"] in ["TCN"]:
            logger.info("\n" + "=" * 100)
            logger.info(f"[RUNNING] Experiment {idx}/{len(all_configs)}: {config['experiment_name']}")
            logger.info("=" * 100)

            try:
                logger.info("[STEP] Preprocessing data (feature selection, scaling, time filters)...")
                df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)

                logger.info("[STEP] Creating sliding windows...")
                X_hist, X_fcst, y, hours, dates = create_sliding_windows(
                    df_clean, config['past_hours'], config['future_hours'],
                    hist_feats, fcst_feats, no_hist_power
                )
                logger.info(f"[INFO] Windowed dataset shapes → X_hist: {X_hist.shape}, y: {y.shape}")
                logger.info("\n--- Window Debug Info ---")
                logger.info(f"Hist feats:      {hist_feats}")
                logger.info(f"Fcst feats:      {fcst_feats}")
                logger.info(f"past_hours*6:    {config['past_hours']*6}")
                logger.info(f"future_hours*6:  {config['future_hours']*6}")
                logger.info(f"X_hist.shape:    {X_hist.shape}")
                if X_fcst is not None:
                    logger.info(f"X_fcst.shape:    {X_fcst.shape}")
                else:
                    logger.info("X_fcst:          None")
                logger.info(f"y.shape:         {y.shape}")
                logger.info("--------------------------\n")

                logger.info("[STEP] Splitting train/val/test sets...")
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

                # Extract hour arrays — one per sample, aligned with indices
                train_hours = np.array([hours[i] for i in train_idx])
                val_hours   = np.array([hours[i] for i in val_idx])
                test_hours  = np.array([hours[i] for i in test_idx])


                train_data = (
                    X_hist[train_idx],
                    None if X_fcst is None else X_fcst[train_idx],
                    y[train_idx],
                    train_hours,
                    []
                )

                val_data = (
                    X_hist[val_idx],
                    None if X_fcst is None else X_fcst[val_idx],
                    y[val_idx],
                    val_hours,
                    []
                )

                test_data = (
                    X_hist[test_idx],
                    None if X_fcst is None else X_fcst[test_idx],
                    y[test_idx],
                    test_hours,
                    dates     # <-- dates only needed for test set outputs
                )


                logger.info("[STEP] Training model...")
                model, metrics = train_dl_model(config, train_data, val_data, test_data, (scaler_hist, scaler_fcst, scaler_target))

                logger.info(f"[RESULT] MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

                logger.info("[STEP] Generating sliding forecast plots...")

                # plot_sliding_forecast_samples(
                #     model=model,
                #     config=config,
                #     metrics=metrics,
                #     test_data=test_data,
                #     save_root="results/sliding_plots",
                #     scaler_target=scaler_target
                # )

                # logger.info("[STEP] Generating debug plots (combined + 24 individual per sample)...")


                # debug_stats = run_debug_for_test_samples(
                #     model=model,
                #     config=config,
                #     df_clean=df_clean,
                #     hist_feats=hist_feats,
                #     fcst_feats=fcst_feats,
                #     scaler_target=scaler_target,
                #     test_data=test_data
                #     # save_root defaults to /Users/.../results/debug_plots
                # )


                pd.DataFrame([{
                    'experiment_name': config['experiment_name'],
                    'model': config['model'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2']
                }]).to_csv(output_file, mode='a', header=False, index=False)

            # except Exception as e:
            #     logger.info(f"[ERROR] Experiment failed → {config['experiment_name']}")
            #     logger.info(f"[ERROR] Reason: {str(e)}")
            #     pd.DataFrame([{
            #         'experiment_name': config['experiment_name'],
            #         'model': config['model'],
            #         'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
            #     }]).to_csv(output_file, mode='a', header=False, index=False)
            #     continue

            except Exception as e:
                logger.info("\n" + "="*100)
                logger.info(f"[ERROR] Experiment failed → {config['experiment_name']}")
                logger.info(f"[ERROR] Reason: {str(e)}")
                logger.info("[TRACEBACK]")
                traceback.print_exc()   # ✅ full traceback logger.infoed
                logger.info("="*100 + "\n")

                # Record failure in CSV (still useful)
                pd.DataFrame([{
                    'experiment_name': config['experiment_name'],
                    'model': config['model'],
                    'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
                }]).to_csv(output_file, mode='a', header=False, index=False)

                # ✅ OPTION A: Pause and wait for user input before continuing
                input("Press ENTER to continue to next experiment...")

                continue
        else:
            continue
    logger.info("\n" + "=" * 100)
    logger.info("[COMPLETE] All experiments finished!")
    logger.info(f"[OUTPUT] Results saved to: {output_file}")
    logger.info("=" * 100)

    return True


if __name__ == "__main__":
    success = run_all_experiments()
    if success:
        logger.info("\n[SUCCESS] 所有160次实验完成！")
    else:
        logger.info("\n[FAILED] 实验失败！")
        sys.exit(1)
