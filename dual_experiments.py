"""
dual_experiments.py — MultiChannelLSTM-only experiment sweep ACROSS MULTIPLE CSV FILES
MEMORY-SAFE VERSION with automatic cache clearing and GPU memory management
"""

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import traceback
import re
import warnings
import gc
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model

# ==== CONFIGURATION ====
CSV_DIR = "/content/drive/MyDrive/Solar PV electricity/data"
RESULTS_DIR = "/content/drive/MyDrive/TimeBasedResults"
RESULTS_FILENAME = "multi_channel_results9.csv"

# MEMORY MANAGEMENT SETTINGS
CLEAR_CACHE_EVERY_N_EXPERIMENTS = 40  # Clear cache every 40 experiments (1 CSV worth)
CLEAR_CACHE_EVERY_PERCENT = 1.0       # Alternative: clear every 1%

TE_COLS = [
    "month_cos", "month_sin",
    "hour_cos", "hour_sin",
    "daypos_cos", "daypos_sin",
    "hourpos_sin", "hourpos_cos"
]

# ---------------------------------------------------------
# MEMORY MANAGEMENT UTILITIES
# ---------------------------------------------------------
def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print("[MEMORY] GPU cache cleared")
    except Exception as e:
        print(f"[WARNING] Could not clear GPU memory: {e}")


def get_memory_usage():
    """Get current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
            return allocated, reserved
        return 0, 0
    except:
        return 0, 0


def should_clear_cache(exp_num, total_exps, last_clear_exp):
    """Determine if cache should be cleared."""
    # Clear every N experiments
    if exp_num - last_clear_exp >= CLEAR_CACHE_EVERY_N_EXPERIMENTS:
        return True
    
    # Clear every X percent
    progress = (exp_num / total_exps) * 100
    last_progress = (last_clear_exp / total_exps) * 100
    if progress - last_progress >= CLEAR_CACHE_EVERY_PERCENT:
        return True
    
    return False


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_project_id(csv_filename):
    """Extract project ID from CSV filename."""
    try:
        name = csv_filename.replace('.csv', '').replace('Project', '')
        match = re.match(r'^(\d+)', name)
        if match:
            return int(match.group(1))
        else:
            return int(name)
    except:
        print(f"[WARNING] Could not extract project_id from '{csv_filename}', using hash")
        return abs(hash(csv_filename)) % 100000


def safe_parse_datetime(col):
    """Parse a date column that may contain various formats."""
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


def get_output_filepath():
    """Generate output file path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, RESULTS_FILENAME)


def get_last_completed_experiment(output_file):
    """Read the existing CSV and determine the last completed experiment number."""
    if not os.path.exists(output_file):
        return 0
    try:
        df = pd.read_csv(output_file)
        if len(df) == 0:
            return 0
        completed = df[df['mae'].notna()]['global_exp_num'].max()
        return int(completed) if not pd.isna(completed) else 0
    except Exception as e:
        print(f"[WARNING] Could not read existing results: {e}")
        return 0


def initialize_results_file(output_file):
    """Create results file with headers if it doesn't exist."""
    if not os.path.exists(output_file):
        pd.DataFrame(columns=[
            'global_exp_num', 'project_id', 'csv_file', 'csv_index', 'config_index',
            'experiment_name', 'model', 'mae', 'rmse', 'r2', 'timestamp'
        ]).to_csv(output_file, index=False)
        print(f"[INFO] Created new results file: {output_file}")
    else:
        print(f"[INFO] Using existing results file: {output_file}")


def save_result(output_file, result_dict):
    """Append a single result row to the CSV file."""
    pd.DataFrame([result_dict]).to_csv(output_file, mode='a', header=False, index=False)


# ---------------------------------------------------------
# CONFIG GENERATION
# ---------------------------------------------------------
def generate_configs_multichannel():
    """Generate MultiChannelLSTM configs only."""
    print("[INFO] Generating MultiChannelLSTM configs ...")
    configs = []
    complexities = ["low", "high"]
    lookbacks = [24, 72]
    te_options = [False, True]

    feature_combos_pv = [
        {'name': 'PV', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+HW', 'use_pv': True, 'use_hist_weather': True, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+NWP', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'PV+NWP+', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]

    feature_combos_nwp = [
        {'name': 'NWP', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'NWP+', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]

    for complexity in complexities:
        for lookback in lookbacks:
            for feat in feature_combos_pv:
                for use_te in te_options:
                    configs.append(create_cfg(complexity, lookback, feat, use_te, False))

    for complexity in complexities:
        for feat in feature_combos_nwp:
            for use_te in te_options:
                configs.append(create_cfg(complexity, 0, feat, use_te, True))

    print(f"[INFO] Total configs per CSV = {len(configs)}")
    return configs


def create_cfg(complexity, lookback, feat, use_te, nwp_only):
    """Create a single config dictionary."""
    cfg = {
        "model": "MultiChannelLSTM",
        "model_complexity": complexity,
        "use_pv": feat["use_pv"],
        "use_hist_weather": feat["use_hist_weather"],
        "use_forecast": feat["use_forecast"],
        "use_ideal_nwp": feat["use_ideal_nwp"],
        "use_time_encoding": use_te,
        "weather_category": "all_weather",
        "future_hours": 24,
        "start_date": "2022-01-01",
        "end_date": "2024-09-28",
    }

    if nwp_only:
        cfg["past_hours"] = 0
        cfg["no_hist_power"] = True
        fname = feat["name"]
    else:
        cfg["past_hours"] = lookback
        cfg["no_hist_power"] = False
        fname = f"{feat['name']}_{lookback}h"

    if complexity == "low":
        cfg["train_params"] = {"epochs": 20, "batch_size": 64, "learning_rate": 0.001}
        cfg["model_params"] = {"hidden_dim": 16, "num_layers": 1, "dropout": 0.1, "te_dim": 8}
    else:
        cfg["train_params"] = {"epochs": 35, "batch_size": 64, "learning_rate": 0.001}
        cfg["model_params"] = {"hidden_dim": 32, "num_layers": 2, "dropout": 0.1, "te_dim": 8}

    te_suffix = "TE" if use_te else "noTE"
    cfg["experiment_name"] = f"MultiCH_{complexity}_{fname}_{te_suffix}"

    return cfg


def compute_all_TE(df_clean, hist_steps, fut_steps, total):
    """Compute time encoding arrays for all samples."""
    missing = [c for c in TE_COLS if c not in df_clean.columns]
    if missing:
        return None

    TE_list = []
    start = hist_steps

    for i in range(total):
        base = start + i
        hist_start = base - hist_steps
        fut_end = base + fut_steps
        te_hist = df_clean.iloc[hist_start:base][TE_COLS].values
        te_fut  = df_clean.iloc[base:fut_end][TE_COLS].values
        te = np.concatenate([te_hist, te_fut], axis=0).reshape(-1)
        TE_list.append(te)

    return np.array(TE_list, dtype=np.float32)


def select_features_from_arrays(X_hist_full, X_fcst_full, hist_feat_names, fcst_feat_names, cfg):
    """Select specific feature columns from full arrays based on config."""
    TIME_ENCODING_FEATURES = ['month_cos', 'month_sin', 'hour_cos', 'hour_sin', 
                              'daypos_cos', 'daypos_sin', 'hourpos_sin', 'hourpos_cos']
    
    hist_indices = []
    selected_hist_names = []
    
    if cfg.get("use_pv", False) and not cfg.get("no_hist_power", False):
        for i, name in enumerate(hist_feat_names):
            if name in TIME_ENCODING_FEATURES:
                continue
            name_lower = str(name).lower().replace('_', ' ')
            power_keywords = ['capacity', 'factor', 'electricity', 'generated', 'generation']
            if any(kw in name_lower for kw in power_keywords):
                hist_indices.append(i)
                selected_hist_names.append(name)
    
    if cfg.get("use_hist_weather", False):
        weather_keywords = ['temp', 'humidity', 'wind', 'pressure', 'ghi', 'dni', 'dhi', 
                           'radiation', 'irradiance', 'solar', 'weather', 'precip', 'cloud',
                           'snow', 'rain', 'apparent', 'dew', 'vapour', 'vapor', 'tilted',
                           'shortwave', 'diffuse', 'direct', 'terrestrial']
        for i, name in enumerate(hist_feat_names):
            if name in TIME_ENCODING_FEATURES:
                continue
            name_lower = str(name).lower().replace('_', ' ')
            if any(w in name_lower for w in weather_keywords):
                if not any(kw in name_lower for kw in ['capacity', 'factor', 'electricity', 'generated', 'generation']):
                    if i not in hist_indices:
                        hist_indices.append(i)
                        selected_hist_names.append(name)
    
    if (cfg.get("use_pv", False) or cfg.get("use_hist_weather", False)) and len(hist_indices) == 0:
        for i, name in enumerate(hist_feat_names):
            if name not in TIME_ENCODING_FEATURES:
                hist_indices.append(i)
                selected_hist_names.append(name)
    
    fcst_indices = []
    selected_fcst_names = []
    
    if cfg.get("use_forecast", False) and fcst_feat_names is not None:
        for i, name in enumerate(fcst_feat_names):
            if name in TIME_ENCODING_FEATURES:
                continue
            name_lower = str(name).lower().replace('_', ' ')
            if not cfg.get("use_ideal_nwp", False):
                if 'ideal' not in name_lower:
                    fcst_indices.append(i)
                    selected_fcst_names.append(name)
            else:
                fcst_indices.append(i)
                selected_fcst_names.append(name)
    
    if len(hist_indices) > 0 and X_hist_full is not None:
        X_hist = X_hist_full[:, :, hist_indices]
    else:
        X_hist = np.zeros((X_hist_full.shape[0], X_hist_full.shape[1], 0)) if X_hist_full is not None else None
    
    if len(fcst_indices) > 0 and X_fcst_full is not None:
        X_fcst = X_fcst_full[:, :, fcst_indices]
    else:
        X_fcst = None if X_fcst_full is None else np.zeros((X_fcst_full.shape[0], X_fcst_full.shape[1], 0))
    
    hist_dim = X_hist.shape[2] if X_hist is not None else 0
    fcst_dim = X_fcst.shape[2] if X_fcst is not None else 0
    
    if hist_dim == 0 and fcst_dim == 0:
        raise ValueError(f"Feature selection resulted in 0 dimensions!")
    
    return X_hist, X_fcst, selected_hist_names, selected_fcst_names


# ---------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------
def run_multichannel_experiments_multi_file(start_from_exp=None):
    """Run MultiChannelLSTM experiments across all CSV files with memory management."""
    print("=" * 120)
    print("[START] MultiChannelLSTM Multi-File Experiment Sweep (MEMORY-SAFE)")
    print(f"[MEMORY] Cache will be cleared every {CLEAR_CACHE_EVERY_N_EXPERIMENTS} experiments")
    print("=" * 120)

    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
    csv_files.sort()

    configs = generate_configs_multichannel()
    total_csvs = len(csv_files)
    total_configs = len(configs)
    total_global_exps = total_csvs * total_configs

    print(f"[INFO] Total Experiments: {total_global_exps}")

    output_file = get_output_filepath()
    initialize_results_file(output_file)

    if start_from_exp is None:
        last_completed = get_last_completed_experiment(output_file)
        start_exp = last_completed + 1
        print(f"\n[AUTO-RESUME] Starting from: {start_exp}")
    else:
        start_exp = start_from_exp

    if start_exp > total_global_exps:
        print(f"[INFO] All experiments already completed!")
        return True

    df_raw = None
    current_csv_index = -1
    preprocessing_cache = {}
    last_cache_clear_exp = 0
    
    for global_exp_num in range(start_exp, total_global_exps + 1):
        
        csv_index = (global_exp_num - 1) // total_configs
        config_index = (global_exp_num - 1) % total_configs
        
        csv_file = csv_files[csv_index]
        cfg = configs[config_index].copy()
        
        project_id = extract_project_id(csv_file)
        prefixed_exp_name = f"{project_id}_{cfg['experiment_name']}"
        
        print(f"\n[EXPERIMENT {global_exp_num}/{total_global_exps}] {prefixed_exp_name}")

        result_dict = {
            'global_exp_num': global_exp_num,
            'project_id': project_id,
            'csv_file': csv_file,
            'csv_index': csv_index + 1,
            'config_index': config_index + 1,
            'experiment_name': prefixed_exp_name,
            'model': cfg['model'],
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            # Check if we should clear cache periodically
            if should_clear_cache(global_exp_num, total_global_exps, last_cache_clear_exp):
                print(f"[MEMORY] Periodic cache clear triggered")
                preprocessing_cache.clear()
                clear_gpu_memory()
                last_cache_clear_exp = global_exp_num
                alloc, reserved = get_memory_usage()
                print(f"[MEMORY] GPU: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            if csv_index != current_csv_index:
                full_path = os.path.join(CSV_DIR, csv_file)
                df_raw = pd.read_csv(full_path)
                
                date_col = None
                for col in df_raw.columns:
                    if col.lower() in ['date', 'datetime', 'time', 'timestamp']:
                        date_col = col
                        break
                
                if date_col is None:
                    raise ValueError(f"Could not find datetime column in {csv_file}")
                
                df_raw['Datetime'] = safe_parse_datetime(df_raw[date_col])
                df_raw['Datetime'] = df_raw['Datetime'].dt.tz_localize(None)
                
                current_csv_index = csv_index
                preprocessing_cache = {}
                clear_gpu_memory()

            cfg['data_path'] = os.path.join(CSV_DIR, csv_file)
            cfg['save_dir'] = f"results/{prefixed_exp_name}"

            lookback = cfg["past_hours"]
            
            if lookback in preprocessing_cache:
                cached_data = preprocessing_cache[lookback]
                df_clean = cached_data['df_clean']
                X_hist_full = cached_data['X_hist_full']
                X_fcst_full = cached_data['X_fcst_full']
                y = cached_data['y']
                hours = cached_data['hours']
                dates = cached_data['dates']
                sh = cached_data['sh']
                sf = cached_data['sf']
                st = cached_data['st']
                hist_feat_names = cached_data['hist_feat_names']
                fcst_feat_names = cached_data['fcst_feat_names']
            else:
                preprocess_cfg = cfg.copy()
                preprocess_cfg["use_pv"] = True
                preprocess_cfg["use_hist_weather"] = True
                preprocess_cfg["use_forecast"] = True
                preprocess_cfg["use_ideal_nwp"] = True
                preprocess_cfg["use_time_encoding"] = True
                
                df_clean, hist_feats, fcst_feats, sh, sf, st, no_hist = preprocess_features(
                    df_raw.copy(), preprocess_cfg
                )

                X_hist_full, X_fcst_full, y, hours, dates = create_sliding_windows(
                    df_clean, preprocess_cfg["past_hours"], preprocess_cfg["future_hours"],
                    hist_feats, fcst_feats, no_hist
                )
                
                preprocessing_cache[lookback] = {
                    'df_clean': df_clean,
                    'X_hist_full': X_hist_full,
                    'X_fcst_full': X_fcst_full,
                    'y': y,
                    'hours': hours,
                    'dates': dates,
                    'sh': sh,
                    'sf': sf,
                    'st': st,
                    'hist_feat_names': hist_feats,
                    'fcst_feat_names': fcst_feats
                }
                hist_feat_names = hist_feats
                fcst_feat_names = fcst_feats

            X_hist, X_fcst, _, _ = select_features_from_arrays(
                X_hist_full, X_fcst_full, hist_feat_names, fcst_feat_names, cfg
            )

            total = len(X_hist)
            hist_steps = cfg["past_hours"] * 6
            fut_steps = cfg["future_hours"] * 6

            if cfg["use_time_encoding"]:
                missing_te_cols = [c for c in TE_COLS if c not in df_clean.columns]
                if missing_te_cols:
                    cfg["use_time_encoding"] = False
                    cfg["model_params"]["use_time_encoding"] = False
                    TE_all = None
                else:
                    TE_all = compute_all_TE(df_clean, hist_steps, fut_steps, total)
            else:
                TE_all = None

            tr = int(total * 0.8)
            va = int(total * 0.9)

            idx_tr = np.arange(0, tr)
            idx_va = np.arange(tr, va)
            idx_te = np.arange(va, total)

            Xh_tr, Xh_va, Xh_te = X_hist[idx_tr], X_hist[idx_va], X_hist[idx_te]
            Xf_tr = X_fcst[idx_tr] if X_fcst is not None else None
            Xf_va = X_fcst[idx_va] if X_fcst is not None else None
            Xf_te = X_fcst[idx_te] if X_fcst is not None else None

            y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
            hrs_tr = np.array([hours[i] for i in idx_tr])
            hrs_va = np.array([hours[i] for i in idx_va])
            hrs_te = np.array([hours[i] for i in idx_te])
            dates_te = [dates[i] for i in idx_te]

            TE_tr = TE_all[idx_tr] if TE_all is not None else None
            TE_va = TE_all[idx_va] if TE_all is not None else None
            TE_te = TE_all[idx_te] if TE_all is not None else None

            train_data = (Xh_tr, Xf_tr, y_tr, hrs_tr, [], TE_tr)
            val_data   = (Xh_va, Xf_va, y_va, hrs_va, [], TE_va)
            test_data  = (Xh_te, Xf_te, y_te, hrs_te, dates_te, TE_te)

            model, metrics = train_dl_model(cfg, train_data, val_data, test_data, (sh, sf, st))

            result_dict['mae'] = metrics['mae']
            result_dict['rmse'] = metrics['rmse']
            result_dict['r2'] = metrics['r2']

            print(f"[SUCCESS] MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
            
            # Clean up after each experiment
            del model
            clear_gpu_memory()

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            traceback.print_exc()

        save_result(output_file, result_dict)
        
        if global_exp_num % 10 == 0:
            progress = (global_exp_num / total_global_exps) * 100
            print(f"[PROGRESS] {global_exp_num}/{total_global_exps} ({progress:.1f}%)")

    print("\n[COMPLETE] All experiments finished!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-from', type=int, default=None)
    parser.add_argument('--clear-every', type=int, default=40,
                       help='Clear cache every N experiments (default: 40)')
    
    args = parser.parse_args()
    
    if args.clear_every:
        CLEAR_CACHE_EVERY_N_EXPERIMENTS = args.clear_every
    
    run_multichannel_experiments_multi_file(start_from_exp=args.start_from)