#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单独运行LSTM模型对比PV+NWP vs PV+NWP+
配置: High Complexity, 24h Lookback, with Time Encoding
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# 确保工作目录正确
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model
import torch

print("="*100)
print(" " * 30 + "LSTM: PV+NWP vs PV+NWP+ Comparison")
print("="*100)
print("\nConfiguration:")
print("  Model: LSTM")
print("  Complexity: High")
print("  Lookback: 24 hours")
print("  Time Encoding: Yes")
print("="*100)

# 基础配置
base_config = {
    'model': 'LSTM',
    'model_complexity': 'high',
    'past_hours': 24,
    'past_days': 1,
    'future_hours': 24,
    'use_time_encoding': True,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'train_params': {
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.001,
        'patience': 5,
        'min_delta': 0.001,
        'weight_decay': 1e-4
    },
    'model_params': {
        'd_model': 32,
        'hidden_dim': 16,
        'num_heads': 2,
        'num_layers': 2,
        'dropout': 0.1,
        'tcn_channels': [16, 32],
        'kernel_size': 3
    }
}

# 检查GPU
print(f"\nGPU Status: CUDA Available = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# 加载数据
print("\nLoading data...")
data_path = "data/Project1140.csv"
df = pd.read_csv(data_path)
df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
print(f"Data loaded: {len(df)} rows")

# 存储结果
results = {}

# ============================================================================
# 实验1: PV+NWP (预测天气)
# ============================================================================
print("\n" + "="*100)
print("EXPERIMENT 1: PV + NWP (Predicted Weather)")
print("="*100)

config_nwp = base_config.copy()
config_nwp.update({
    'experiment_name': 'LSTM_high_PV+NWP_24h_TE',
    'save_dir': 'results/LSTM_high_PV+NWP_24h_TE',
    'use_pv': True,
    'use_hist_weather': False,
    'use_forecast': True,
    'use_ideal_nwp': False,
    'weather_category': 'all_weather',
    'no_hist_power': False
})

start_time = time.time()

# 数据预处理
print("\nData preprocessing...")
df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config_nwp)

# 创建滑动窗口
print("Creating sliding windows...")
X_hist, X_fcst, y, hours, dates = create_sliding_windows(
    df_clean, config_nwp['past_hours'], config_nwp['future_hours'], 
    hist_feats, fcst_feats, no_hist_power
)

print(f"Data shape: X_hist={X_hist.shape}, X_fcst={X_fcst.shape}, y={y.shape}")

# 按时间顺序分割数据
total_samples = len(X_hist)
indices = np.arange(total_samples)

train_size = int(total_samples * config_nwp['train_ratio'])
val_size = int(total_samples * config_nwp['val_ratio'])

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

# 分割数据
X_hist_train, y_train = X_hist[train_idx], y[train_idx]
X_hist_val, y_val = X_hist[val_idx], y[val_idx]
X_hist_test, y_test = X_hist[test_idx], y[test_idx]

if X_fcst is not None:
    X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
else:
    X_fcst_train, X_fcst_val, X_fcst_test = None, None, None

# 分割hours和dates
train_hours = np.array([hours[i] for i in train_idx])
val_hours = np.array([hours[i] for i in val_idx])
test_hours = np.array([hours[i] for i in test_idx])
test_dates = [dates[i] for i in test_idx]

train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
test_data = (X_hist_test, X_fcst_test, y_test, test_hours, test_dates)
scalers = (scaler_hist, scaler_fcst, scaler_target)

# 训练模型
print("\nTraining model...")
model_nwp, metrics_nwp = train_dl_model(config_nwp, train_data, val_data, test_data, scalers)

training_time_nwp = time.time() - start_time

print(f"\n[PV+NWP Results]")
print(f"  RMSE: {metrics_nwp['rmse']:.4f}")
print(f"  MAE: {metrics_nwp['mae']:.4f}")
print(f"  R2: {metrics_nwp['r2']:.4f}")
print(f"  Training time: {training_time_nwp:.1f}s")
print(f"  Best epoch: {metrics_nwp.get('best_epoch', 'N/A')}")

# 保存结果
results['PV+NWP'] = {
    'metrics': metrics_nwp,
    'predictions': metrics_nwp.get('predictions', None),
    'y_true': metrics_nwp.get('y_true', None),
    'dates': test_dates,
    'training_time': training_time_nwp
}

# ============================================================================
# 实验2: PV+NWP+ (实际天气)
# ============================================================================
print("\n" + "="*100)
print("EXPERIMENT 2: PV + NWP+ (Actual Weather)")
print("="*100)

config_nwp_plus = base_config.copy()
config_nwp_plus.update({
    'experiment_name': 'LSTM_high_PV+NWP+_24h_TE',
    'save_dir': 'results/LSTM_high_PV+NWP+_24h_TE',
    'use_pv': True,
    'use_hist_weather': False,
    'use_forecast': True,
    'use_ideal_nwp': True,  # 使用实际天气
    'weather_category': 'all_weather',
    'no_hist_power': False
})

start_time = time.time()

# 数据预处理
print("\nData preprocessing...")
df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config_nwp_plus)

# 创建滑动窗口
print("Creating sliding windows...")
X_hist, X_fcst, y, hours, dates = create_sliding_windows(
    df_clean, config_nwp_plus['past_hours'], config_nwp_plus['future_hours'], 
    hist_feats, fcst_feats, no_hist_power
)

print(f"Data shape: X_hist={X_hist.shape}, X_fcst={X_fcst.shape}, y={y.shape}")

# 按时间顺序分割数据（使用相同的split）
X_hist_train, y_train = X_hist[train_idx], y[train_idx]
X_hist_val, y_val = X_hist[val_idx], y[val_idx]
X_hist_test, y_test = X_hist[test_idx], y[test_idx]

if X_fcst is not None:
    X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
else:
    X_fcst_train, X_fcst_val, X_fcst_test = None, None, None

test_dates_plus = [dates[i] for i in test_idx]

train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
test_data = (X_hist_test, X_fcst_test, y_test, test_hours, test_dates_plus)
scalers = (scaler_hist, scaler_fcst, scaler_target)

# 训练模型
print("\nTraining model...")
model_nwp_plus, metrics_nwp_plus = train_dl_model(config_nwp_plus, train_data, val_data, test_data, scalers)

training_time_nwp_plus = time.time() - start_time

print(f"\n[PV+NWP+ Results]")
print(f"  RMSE: {metrics_nwp_plus['rmse']:.4f}")
print(f"  MAE: {metrics_nwp_plus['mae']:.4f}")
print(f"  R2: {metrics_nwp_plus['r2']:.4f}")
print(f"  Training time: {training_time_nwp_plus:.1f}s")
print(f"  Best epoch: {metrics_nwp_plus.get('best_epoch', 'N/A')}")

# 保存结果
results['PV+NWP+'] = {
    'metrics': metrics_nwp_plus,
    'predictions': metrics_nwp_plus.get('predictions', None),
    'y_true': metrics_nwp_plus.get('y_true', None),
    'dates': test_dates_plus,
    'training_time': training_time_nwp_plus
}

# ============================================================================
# 对比分析和保存结果
# ============================================================================
print("\n" + "="*100)
print("COMPARISON SUMMARY")
print("="*100)

print(f"\nMetrics Comparison:")
print(f"{'Metric':<15} {'PV+NWP':<15} {'PV+NWP+':<15} {'Difference':<15}")
print("-"*60)
print(f"{'RMSE':<15} {metrics_nwp['rmse']:<15.4f} {metrics_nwp_plus['rmse']:<15.4f} {metrics_nwp_plus['rmse'] - metrics_nwp['rmse']:>+14.4f}")
print(f"{'MAE':<15} {metrics_nwp['mae']:<15.4f} {metrics_nwp_plus['mae']:<15.4f} {metrics_nwp_plus['mae'] - metrics_nwp['mae']:>+14.4f}")
print(f"{'R2':<15} {metrics_nwp['r2']:<15.4f} {metrics_nwp_plus['r2']:<15.4f} {metrics_nwp_plus['r2'] - metrics_nwp['r2']:>+14.4f}")
print(f"{'Training Time':<15} {training_time_nwp:<15.1f} {training_time_nwp_plus:<15.1f} {training_time_nwp_plus - training_time_nwp:>+14.1f}")

if metrics_nwp['rmse'] < metrics_nwp_plus['rmse']:
    print(f"\n>>> PV+NWP (Predicted Weather) performs BETTER by {(metrics_nwp_plus['rmse'] - metrics_nwp['rmse']) / metrics_nwp_plus['rmse'] * 100:.2f}%")
else:
    print(f"\n>>> PV+NWP+ (Actual Weather) performs BETTER by {(metrics_nwp['rmse'] - metrics_nwp_plus['rmse']) / metrics_nwp['rmse'] * 100:.2f}%")

# ============================================================================
# 保存预测结果到CSV
# ============================================================================
print("\n" + "="*100)
print("SAVING RESULTS TO CSV")
print("="*100)

# 获取预测结果
preds_nwp = results['PV+NWP']['predictions']
preds_nwp_plus = results['PV+NWP+']['predictions']
y_true_nwp = results['PV+NWP']['y_true']
y_true_nwp_plus = results['PV+NWP+']['y_true']
dates_nwp = results['PV+NWP']['dates']

# 验证数据一致性
print(f"Data shapes:")
print(f"  PV+NWP predictions: {preds_nwp.shape}")
print(f"  PV+NWP+ predictions: {preds_nwp_plus.shape}")
print(f"  Ground truth (NWP): {y_true_nwp.shape}")
print(f"  Ground truth (NWP+): {y_true_nwp_plus.shape}")
print(f"  Dates: {len(dates_nwp)}")

# 展平预测结果 (取每个样本的第一个预测值，即未来第1小时的预测)
# 如果要保存所有24小时的预测，需要调整
pred_nwp_flat = preds_nwp[:, 0] if len(preds_nwp.shape) > 1 else preds_nwp
pred_nwp_plus_flat = preds_nwp_plus[:, 0] if len(preds_nwp_plus.shape) > 1 else preds_nwp_plus
truth_nwp_flat = y_true_nwp[:, 0] if len(y_true_nwp.shape) > 1 else y_true_nwp
truth_nwp_plus_flat = y_true_nwp_plus[:, 0] if len(y_true_nwp_plus.shape) > 1 else y_true_nwp_plus

# 创建DataFrame
results_df = pd.DataFrame({
    'date': dates_nwp,
    'ground_truth_nwp': truth_nwp_flat,
    'prediction_nwp': pred_nwp_flat,
    'ground_truth_nwp_plus': truth_nwp_plus_flat,
    'prediction_nwp_plus': pred_nwp_plus_flat,
    'error_nwp': np.abs(truth_nwp_flat - pred_nwp_flat),
    'error_nwp_plus': np.abs(truth_nwp_plus_flat - pred_nwp_plus_flat)
})

# 保存到CSV
output_file = 'LSTM_NWP_vs_NWP_plus_comparison.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")
print(f"Total samples: {len(results_df)}")

# 显示前10行
print(f"\nFirst 10 rows:")
print(results_df.head(10).to_string(index=False))

# 统计摘要
print(f"\n" + "="*100)
print("STATISTICAL SUMMARY")
print("="*100)
print(f"\nPV+NWP (Predicted Weather):")
print(f"  Mean Absolute Error: {results_df['error_nwp'].mean():.4f}")
print(f"  Std Dev of Error: {results_df['error_nwp'].std():.4f}")
print(f"  Max Error: {results_df['error_nwp'].max():.4f}")

print(f"\nPV+NWP+ (Actual Weather):")
print(f"  Mean Absolute Error: {results_df['error_nwp_plus'].mean():.4f}")
print(f"  Std Dev of Error: {results_df['error_nwp_plus'].std():.4f}")
print(f"  Max Error: {results_df['error_nwp_plus'].max():.4f}")

print("\n" + "="*100)
print("EXPERIMENT COMPLETED")
print("="*100)

