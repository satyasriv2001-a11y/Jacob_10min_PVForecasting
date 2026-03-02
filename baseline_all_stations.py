#!/usr/bin/env python3
"""
Baseline experiment: Linear + XGBoost (all stations)
Config: high complexity, PV+NWP, 24h lookback, time encoding (TE)
Data: loops over all CSV files in data/, trains per station and aggregates results.

Run from project root (or set PROJECT_DIR). On Colab: mount Drive first and set PROJECT_DIR.
"""

import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Step 0: Environment (Colab: mount Drive + pip; local: set PROJECT_DIR)
# ---------------------------------------------------------------------------
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_DIR = "/content/drive/MyDrive/SolarPrediction"
except ImportError:
    PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
print(f"项目路径: {PROJECT_DIR}")
print(f"文件列表: {os.listdir('.')[:10]}...")

# pip install (uncomment if needed, or run once in Colab)
# import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost", "lightgbm", "rich"])

# Optional: check GPU (for XGBoost)
try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Step 1: List all CSVs in data/
# ---------------------------------------------------------------------------
data_dir = os.path.join(PROJECT_DIR, "data")
csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
if not csv_files:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "Project*.csv")))
csv_files = [f for f in csv_files if os.path.isfile(f)]

print(f"\n找到 {len(csv_files)} 个 CSV 文件:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")
if not csv_files:
    print("❌ data/ 下没有 CSV，请先上传数据到 data/ 目录")
    sys.exit(1)

if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
    print("❌ data 目录不存在")
    sys.exit(1)
print(f"✅ data 目录存在: {data_dir}，将运行 {len(csv_files)} 个 CSV")

# ---------------------------------------------------------------------------
# Step 2: Config
# ---------------------------------------------------------------------------
config = {
    "model_complexity": "high",
    "use_pv": True,
    "use_hist_weather": False,
    "use_forecast": True,
    "use_ideal_nwp": False,
    "use_time_encoding": True,
    "past_hours": 24,
    "future_hours": 24,
    "weather_category": "all_weather",
    "start_date": "2022-01-01",
    "end_date": "2024-09-28",
}

print("\n实验配置:")
print(f"  特征组合: PV + NWP（历史发电 + 天气预报）")
print(f"  回看窗口: {config['past_hours']}h → 预测窗口: {config['future_hours']}h ({config['future_hours']*6} 步)")
print(f"  模型复杂度: {config['model_complexity']}, 时间编码: 开启")

# ---------------------------------------------------------------------------
# Step 3: Run experiment per station (preprocess → split → train Linear & XGBoost)
# ---------------------------------------------------------------------------
from data.data_utils import preprocess_features, create_sliding_windows
from train.train_ml import train_ml_model

config_linear = config.copy()
config_linear["model"] = "Linear"
config_xgb = config.copy()
config_xgb["model"] = "XGB"

all_results = []

for i, data_path in enumerate(csv_files):
    station_name = os.path.splitext(os.path.basename(data_path))[0]
    print(f"\n[{i+1}/{len(csv_files)}] {station_name} ...")
    t0 = time.time()
    df_raw = pd.read_csv(data_path)
    df_raw["Datetime"] = pd.to_datetime(df_raw["date"])
    try:
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = (
            preprocess_features(df_raw.copy(), config)
        )
        Xh, Xf, y, hours, dates = create_sliding_windows(
            df_clean, config["past_hours"], config["future_hours"],
            hist_feats, fcst_feats, no_hist_power,
        )
    except Exception as e:
        print(f"  ⚠️ 跳过 {station_name}: {e}")
        continue
    total = len(Xh)
    train_end = int(total * 0.85)
    Xh_train, Xh_test = Xh[:train_end], Xh[train_end:]
    Xf_train, Xf_test = (Xf[:train_end], Xf[train_end:]) if Xf is not None else (None, None)
    y_train, y_test = y[:train_end], y[train_end:]
    dates_test = dates[train_end:]
    try:
        linear_model, linear_metrics = train_ml_model(
            config_linear, Xh_train, Xf_train, y_train,
            Xh_test, Xf_test, y_test, dates_test,
            scaler_target=scaler_target,
        )
        xgb_model, xgb_metrics = train_ml_model(
            config_xgb, Xh_train, Xf_train, y_train,
            Xh_test, Xf_test, y_test, dates_test,
            scaler_target=scaler_target,
        )
    except Exception as e:
        print(f"  ⚠️ 训练失败 {station_name}: {e}")
        continue
    all_results.append({
        "station": station_name,
        "linear_metrics": linear_metrics,
        "xgb_metrics": xgb_metrics,
        "n_train": train_end,
        "n_test": total - train_end,
    })
    print(f"  完成: MAE Linear={linear_metrics['mae']:.4f}, XGB={xgb_metrics['mae']:.4f} ({time.time()-t0:.1f}s)")

print(f"\n✅ 共完成 {len(all_results)} 个站点")

if not all_results:
    print("没有成功完成的站点，退出")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 4: Summary table (all stations)
# ---------------------------------------------------------------------------
rows = []
for r in all_results:
    rows.append({
        "station": r["station"],
        "n_train": r["n_train"],
        "n_test": r["n_test"],
        "MAE_linear": r["linear_metrics"]["mae"],
        "MAE_xgb": r["xgb_metrics"]["mae"],
        "RMSE_linear": r["linear_metrics"]["rmse"],
        "RMSE_xgb": r["xgb_metrics"]["rmse"],
        "R2_linear": r["linear_metrics"]["r2"],
        "R2_xgb": r["xgb_metrics"]["r2"],
        "sMAPE_linear": r["linear_metrics"]["smape"],
        "sMAPE_xgb": r["xgb_metrics"]["smape"],
    })
df_results = pd.DataFrame(rows)
print("\n" + df_results.to_string())
print("\n--- 所有站点平均 ---")
print(df_results[["MAE_linear", "MAE_xgb", "RMSE_linear", "RMSE_xgb", "R2_linear", "R2_xgb"]].mean())

# ---------------------------------------------------------------------------
# Step 5: Who wins (station counts)
# ---------------------------------------------------------------------------
win_linear = (df_results["MAE_linear"] < df_results["MAE_xgb"]).sum()
win_xgb = (df_results["MAE_xgb"] < df_results["MAE_linear"]).sum()
print(f"\nMAE:  Linear 更好 {win_linear} 站,  XGBoost 更好 {win_xgb} 站")
win_linear_r2 = (df_results["R2_linear"] > df_results["R2_xgb"]).sum()
win_xgb_r2 = (df_results["R2_xgb"] > df_results["R2_linear"]).sum()
print(f"R²:   Linear 更好 {win_linear_r2} 站,  XGBoost 更好 {win_xgb_r2} 站")
print(f"平均 MAE:  Linear={df_results['MAE_linear'].mean():.4f},  XGBoost={df_results['MAE_xgb'].mean():.4f}")
print(f"平均 R²:   Linear={df_results['R2_linear'].mean():.4f},  XGBoost={df_results['R2_xgb'].mean():.4f}")

# ---------------------------------------------------------------------------
# Step 6: Plot – MAE by station + first station one sample
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
stations = df_results["station"].tolist()
x = np.arange(len(stations))
w = 0.35
axes[0].bar(x - w / 2, df_results["MAE_linear"], w, label="Linear", color="steelblue")
axes[0].bar(x + w / 2, df_results["MAE_xgb"], w, label="XGBoost", color="coral")
axes[0].set_xticks(x)
axes[0].set_xticklabels(stations, rotation=45, ha="right")
axes[0].set_ylabel("MAE")
axes[0].set_title("MAE 按站点: Linear vs XGBoost")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

if all_results:
    r0 = all_results[0]
    y_true = r0["linear_metrics"]["y_true"]
    pred_linear = r0["linear_metrics"]["predictions"]
    pred_xgb = r0["xgb_metrics"]["predictions"]
    t = np.arange(144) * 10 / 60
    idx = 0
    axes[1].plot(t, y_true[idx], "k-", lw=1.5, label="实际值")
    axes[1].plot(t, pred_linear[idx], "b--", lw=1.2, alpha=0.8, label="Linear")
    axes[1].plot(t, pred_xgb[idx], "r--", lw=1.2, alpha=0.8, label="XGBoost")
    axes[1].set_xlabel("预测时间 (小时)")
    axes[1].set_ylabel("容量因子 (%)")
    axes[1].set_title(f"首个站点 {r0['station']} 样本 0")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

plt.suptitle("Solar PV 10-min 预测: 所有站点 Linear vs XGBoost", fontsize=12, fontweight="bold")
plt.tight_layout()
save_path = os.path.join(PROJECT_DIR, "baseline_all_stations_comparison.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n图片已保存: {save_path}")

# ---------------------------------------------------------------------------
# Step 7: First station detailed comparison (optional)
# ---------------------------------------------------------------------------
if all_results:
    r = all_results[0]
    lm, xm = r["linear_metrics"], r["xgb_metrics"]
    print(f"\n站点: {r['station']}")
    print(f"{'指标':<12} {'Linear':<12} {'XGBoost':<12} {'谁赢了':<10}")
    print("-" * 46)
    for key, name in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²"), ("smape", "sMAPE(%)")]:
        lv, xv = lm[key], xm[key]
        winner = "XGBoost" if (xv > lv if key == "r2" else xv < lv) else "Linear"
        print(f"{name:<12} {lv:<12.4f} {xv:<12.4f} {winner}")

# ---------------------------------------------------------------------------
# Step 8: First station – 4 test samples curves
# ---------------------------------------------------------------------------
if all_results:
    r0 = all_results[0]
    y_true_arr = r0["linear_metrics"]["y_true"]
    pred_linear = r0["linear_metrics"]["predictions"]
    pred_xgb = r0["xgb_metrics"]["predictions"]
    n_test = len(y_true_arr)
    n_plots = min(4, n_test)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    t = np.arange(144) * 10 / 60
    for i in range(n_plots):
        idx = i * (n_test // n_plots) if n_test >= n_plots else i
        axes[i].plot(t, y_true_arr[idx], "k-", lw=1.5, label="实际值")
        axes[i].plot(t, pred_linear[idx], "b--", lw=1.2, alpha=0.8, label="Linear")
        axes[i].plot(t, pred_xgb[idx], "r--", lw=1.2, alpha=0.8, label="XGBoost")
        axes[i].set_ylabel("容量因子 (%)")
        axes[i].set_title(f"{r0['station']} 测试样本 {idx+1}")
        axes[i].legend(loc="upper right", fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(bottom=0)
    axes[-1].set_xlabel("预测时间 (小时)")
    plt.suptitle(
        f"Solar PV 24h 预测: {r0['station']} (配置: high, PV+NWP, 24h lookback, TE)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "baseline_first_station_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()

print("\n🎉 实验完成")
