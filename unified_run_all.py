#!/usr/bin/env python3
"""
Unified Experiment Runner - MEMORY OPTIMIZED
Groups experiments by data config to avoid redundant preprocessing,
but processes one group at a time to avoid OOM.
"""

import os, sys, time, traceback, gc
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import warnings
from rich.logging import RichHandler
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(markup=True)]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model

ML_MODELS = ['RF', 'XGB', 'LGBM', 'Linear']
DL_MODELS = ['LSTM', 'GRU', 'Transformer', 'TCN', 'MultiChannelLSTM']



TE_COLS = [
    'month_cos','month_sin','hour_cos','hour_sin',
    'daypos_cos','daypos_sin','hourpos_sin','hourpos_cos'
]


def get_data_key(cfg):
    """Generate a unique key for data configuration."""
    return (
        cfg['past_hours'],
        cfg['use_pv'],
        cfg['use_hist_weather'],
        cfg['use_forecast'],
        cfg['use_ideal_nwp'],
        cfg.get('no_hist_power', False)
    )


def build_unified_config(model, complexity, lookback, feat, use_te, is_nwp_only):

    cfg = {
        'model': model,
        'model_complexity': complexity,
        'data_path': os.path.join(script_dir, 'data', 'Project1140.csv'),

        'use_pv': feat['use_pv'],
        'use_hist_weather': feat['use_hist_weather'],
        'use_forecast': feat['use_forecast'],
        'use_ideal_nwp': feat['use_ideal_nwp'],

        'use_time_encoding': use_te,
        'weather_category': 'all_weather',

        'future_hours': 24,
        'start_date': '2022-01-01',
        'end_date': '2024-09-28',

        'save_options': {}
    }

    if is_nwp_only:
        cfg['past_hours'] = 0
        cfg['no_hist_power'] = True
        feat_name = feat['name']
    else:
        cfg['past_hours'] = lookback
        cfg['no_hist_power'] = False
        feat_name = f"{feat['name']}_{lookback}h"

    if complexity == 'low':
        cfg['train_params'] = {'epochs': 20, 'batch_size': 64, 'learning_rate': 0.001}
        cfg['model_params'] = {
            'd_model': 32,
            'hidden_dim': 16,
            'num_heads': 2,
            'num_layers': 1,
            'dropout': 0.1,
            'tcn_channels': [16, 32],
            'kernel_size': 3,
            'te_dim': 8
        }
    else:
        cfg['train_params'] = {'epochs': 35, 'batch_size': 64, 'learning_rate': 0.001}
        cfg['model_params'] = {
            'd_model': 64,
            'hidden_dim': 32,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'tcn_channels': [32, 64],
            'kernel_size': 3,
            'te_dim': 8
        }

    te_suffix = 'TE' if use_te else 'noTE'
    cfg['experiment_name'] = f"{model}_{complexity}_{feat_name}_{te_suffix}"
    cfg['save_dir'] = f"results/{cfg['experiment_name']}"

    return cfg



def generate_unified_configs():
    logger.info("Generating all experiment configurations...")

    models = ['LSTM', 'GRU', 'Transformer', 'TCN', 'MultiChannelLSTM',
              'RF', 'XGB', 'LGBM', 'Linear']

    complexities = ['low', 'high']
    lookbacks = [24, 72]
    te_options = [True, False]

    feature_combos_pv = [
        {'name': 'PV',      'use_pv': True, 'use_hist_weather': False, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+HW',   'use_pv': True, 'use_hist_weather': True,  'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+NWP',  'use_pv': True, 'use_hist_weather': False, 'use_forecast': True,  'use_ideal_nwp': False},
        {'name': 'PV+NWP+', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True,  'use_ideal_nwp': True},
    ]

    feature_combos_nwp = [
        {'name': 'NWP',  'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'NWP+', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]

    cfgs = []

    for model in models:
        for comp in complexities:
            for look in lookbacks:
                for feat in feature_combos_pv:
                    for use_te in te_options:
                        cfgs.append(build_unified_config(
                            model, comp, look, feat, use_te, is_nwp_only=False
                        ))

    for model in models:
        for comp in complexities:
            for feat in feature_combos_nwp:
                for use_te in te_options:
                    cfgs.append(build_unified_config(
                        model, comp, 0, feat, use_te, is_nwp_only=True
                    ))

    logger.info(f"Total configs: {len(cfgs)} (expected 160 for DL and ML combined)")
    return cfgs



def compute_TE_sequences(df_clean, past_steps, fut_steps, total):
    """Compute time encoding sequences."""
    TE_all = []
    start = past_steps
    for i in range(start, start + total):
        hist_s, hist_e = i - past_steps, i
        fut_s, fut_e = i, i + fut_steps
        te_hist = df_clean.iloc[hist_s:hist_e][TE_COLS].values
        te_fut = df_clean.iloc[fut_s:fut_e][TE_COLS].values
        TE_all.append(np.concatenate([te_hist, te_fut], axis=0).reshape(-1))
    return np.array(TE_all)


def group_configs_by_data(cfgs):
    """Group configs by their data requirements."""
    groups = defaultdict(list)
    for cfg in cfgs:
        key = get_data_key(cfg)
        groups[key].append(cfg)
    return groups


def run_unified():
    logger.info("=" * 80)
    logger.info("Running MEMORY-OPTIMIZED unified experiments")
    logger.info("=" * 80)
    
    # Generate and group configs
    cfgs = generate_unified_configs()
    grouped = group_configs_by_data(cfgs)

    for key in grouped:
        grouped[key] = sorted(
            grouped[key],
            key=lambda c: (1 if c['model'] in DL_MODELS else 0)
        )
    
    logger.info(f"Grouped {len(cfgs)} experiments into {len(grouped)} unique data configurations")
    
    # Load raw data once
    logger.info("Loading dataset...")
    df_raw = pd.read_csv(os.path.join(script_dir, 'data', 'Project1140.csv'))
    df_raw['Datetime'] = pd.to_datetime(df_raw['date'])
    logger.info(f"Loaded {len(df_raw):,} rows")

    # Setup output
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = f"unified_results_{ts}.csv"
    pd.DataFrame(columns=['experiment_name', 'mae', 'rmse', 'r2']).to_csv(out_csv, index=False)

    exp_count = 0
    total_exps = len(cfgs)
    
    # Process one data group at a time
    for group_idx, (data_key, group_cfgs) in enumerate(grouped.items(), 1):
        logger.info("=" * 80)
        logger.info(f"DATA GROUP [{group_idx}/{len(grouped)}]: {len(group_cfgs)} experiments")
        logger.info(f"  past_hours={data_key[0]}, pv={data_key[1]}, hist_weather={data_key[2]}, "
                   f"forecast={data_key[3]}, ideal_nwp={data_key[4]}")
        logger.info("=" * 80)
        
        # Get a representative config for data processing
        rep_cfg = group_cfgs[0].copy()
        
        try:
            # Preprocess data ONCE for this group
            logger.info("Preprocessing data for this group...")
            df_clean, hist_feats, fcst_feats, sh, sf, st, no_hist_power = preprocess_features(
                df_raw.copy(), rep_cfg
            )
            
            logger.info("Creating sliding windows...")
            Xh, Xf, y, hours, dates = create_sliding_windows(
                df_clean, rep_cfg['past_hours'], rep_cfg['future_hours'],
                hist_feats, fcst_feats, no_hist_power
            )
            logger.info(f"Shapes: Xh={Xh.shape}, y={y.shape}")
            
            # Compute TE sequences
            N = len(Xh)
            past_steps = rep_cfg['past_hours'] * 6
            fut_steps = rep_cfg['future_hours'] * 6
            TE_all = compute_TE_sequences(df_clean, past_steps, fut_steps, N)
            
            # Split indices
            tr, va = int(N * 0.8), int(N * 0.9)
            idx_tr, idx_va, idx_te = np.arange(0, tr), np.arange(tr, va), np.arange(va, N)
            
            # Pre-split data
            Xh_tr, Xh_va, Xh_te = Xh[idx_tr], Xh[idx_va], Xh[idx_te]
            Xf_tr = Xf[idx_tr] if Xf is not None else None
            Xf_va = Xf[idx_va] if Xf is not None else None
            Xf_te = Xf[idx_te] if Xf is not None else None
            y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
            
            hrs_tr = np.array([hours[k] for k in idx_tr])
            hrs_va = np.array([hours[k] for k in idx_va])
            hrs_te = np.array([hours[k] for k in idx_te])
            dates_te = [dates[k] for k in idx_te]
            
            TE_tr, TE_va, TE_te = TE_all[idx_tr], TE_all[idx_va], TE_all[idx_te]
            
            # Free memory from intermediate objects
            del df_clean, Xh, Xf, y, hours, dates, TE_all
            gc.collect()
            
            # Run all experiments in this group
            for cfg in group_cfgs:
                exp_count += 1
                logger.info("-" * 60)
                logger.info(f"[{exp_count}/{total_exps}] {cfg['experiment_name']}")
                
                try:
                    # Update config with actual TE setting
                    # Update config with actual TE setting (safe)
                    train_cfg = cfg.copy()
                    train_cfg['use_time_encoding'] = cfg.get('_actual_use_te', cfg.get('use_time_encoding', False))

                    # Prepare data tuples
                    train_data = (Xh_tr, Xf_tr, y_tr, hrs_tr, [], TE_tr)
                    val_data = (Xh_va, Xf_va, y_va, hrs_va, [], TE_va)
                    test_data = (Xh_te, Xf_te, y_te, hrs_te, dates_te, TE_te)

                    # Train
                    # ===============================
                    # ML Models (RF, XGB, LGBM, Linear)
                    # ===============================
                    if cfg['model'] in ML_MODELS:

                        model, metrics = train_ml_model(
                            cfg,
                            Xh_tr, Xf_tr, y_tr,
                            Xh_te, Xf_te, y_te,
                            dates_te,
                            st  # scaler_target
                        )

                    # ===============================
                    # DL Models
                    # ===============================
                    elif cfg['model'] in DL_MODELS:

                        model, metrics = train_dl_model(
                            cfg,
                            train_data, val_data, test_data,
                            (sh, sf, st)
                        )

                    else:
                        raise ValueError(f"Unknown model type {cfg['model']}")

                    logger.info(f"[RESULT] MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

                    pd.DataFrame([{
                        'experiment_name': cfg['experiment_name'],
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'r2': metrics['r2']
                    }]).to_csv(out_csv, mode='a', header=False, index=False)
                    
                    # Free model memory
                    del model, metrics
                    gc.collect()

                except Exception as e:
                    logger.error(f"Experiment failed: {cfg['experiment_name']}")
                    logger.error(f"Reason: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    pd.DataFrame([{
                        'experiment_name': cfg['experiment_name'],
                        'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
                    }]).to_csv(out_csv, mode='a', header=False, index=False)
            
            # Free group data before next group
            del Xh_tr, Xh_va, Xh_te, Xf_tr, Xf_va, Xf_te
            del y_tr, y_va, y_te, TE_tr, TE_va, TE_te
            gc.collect()
                    
        except Exception as e:
            logger.error(f"Failed to process data group {data_key}: {e}")
            logger.error(traceback.format_exc())
            
            # Mark all experiments in this group as failed
            for cfg in group_cfgs:
                exp_count += 1
                pd.DataFrame([{
                    'experiment_name': cfg['experiment_name'],
                    'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
                }]).to_csv(out_csv, mode='a', header=False, index=False)

    logger.info("=" * 80)
    logger.info("All experiments complete!")
    logger.info(f"Results saved to: {out_csv}")
    logger.info("=" * 80)


if __name__ == '__main__':
    run_unified()