"""
指定日期 + 指定起始时间的预测片段图
目标：2024-05-01, 2024-05-08, 2024-05-16，均从 06:00 开始的 24h 预测
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

BASE = '/Users/terry/Library/CloudStorage/GoogleDrive-zl2268@cornell.edu/.shortcut-targets-by-id/1tYXd7zscd3BkVOz6B4y72IxqdsQEKgzW/SolarPrediction'

# ── 加载缓存 ──────────────────────────────────────────────────────────────────
print("加载 project1033_cache.pkl...")
cache         = joblib.load(os.path.join(BASE, 'project1033_cache.pkl'))
Xh            = cache['Xh']
Xf            = cache['Xf']
y             = cache['y']
dates         = cache['dates']
scaler_target = cache['scaler_target']
train_end     = cache['train_end']
FUTURE_STEPS  = y.shape[1]
STEP_MIN      = 10
print(f"  总样本 {len(Xh):,}  train_end={train_end:,}  test={len(Xh)-train_end:,}")

# ── test split ────────────────────────────────────────────────────────────────
def flatten(Xh, Xf):
    return np.concatenate([Xh.reshape(Xh.shape[0], -1),
                           Xf.reshape(Xf.shape[0], -1)], axis=1)

X_test     = flatten(Xh[train_end:], Xf[train_end:])
y_test     = y[train_end:]
dates_test = dates[train_end:]

def inverse(arr):
    return scaler_target.inverse_transform(arr.reshape(-1, 1)).reshape(-1, FUTURE_STEPS)

y_true = inverse(y_test)

# ── 加载线性模型 ──────────────────────────────────────────────────────────────
LR_PATH = os.path.join(BASE, 'experiments', 'lza1033', 'models', 'linear_model.pkl')
lr = joblib.load(LR_PATH)['model']
lr_preds = np.clip(inverse(lr.predict(X_test)), 0, 100)

# ── 目标日期和起始时间 ────────────────────────────────────────────────────────
TARGET_STARTS = [
    pd.Timestamp('2024-05-01 06:00:00'),
    pd.Timestamp('2024-05-08 06:00:00'),
    pd.Timestamp('2024-05-16 06:00:00'),
]
# dates_test 存的是窗口结束时刻，对应 start + 143*10min = start + 23h50min
TARGET_ENDS = [s + pd.Timedelta(minutes=STEP_MIN * (FUTURE_STEPS - 1))
               for s in TARGET_STARTS]

# 在 dates_test 中查找匹配样本
found_indices = []
for target_end in TARGET_ENDS:
    match = None
    for i, d in enumerate(dates_test):
        try:
            if pd.Timestamp(d) == target_end:
                match = i
                break
        except Exception:
            pass
    if match is not None:
        found_indices.append(match)
        print(f"  找到: end={target_end}  test_idx={match}")
    else:
        print(f"  ⚠️  未找到: end={target_end}")

if not found_indices:
    raise RuntimeError("没有找到任何目标样本，请检查日期是否在测试集内")

# ── 画图 ──────────────────────────────────────────────────────────────────────
n_plots = len(found_indices)
fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
if n_plots == 1:
    axes = [axes]

for ax, idx in zip(axes, found_indices):
    end_dt   = pd.Timestamp(dates_test[idx])
    start_dt = end_dt - pd.Timedelta(minutes=STEP_MIN * (FUTURE_STEPS - 1))
    start_h  = start_dt.hour + start_dt.minute / 60.0   # = 6.0

    # x 轴用实际时钟小时（start_h 到 start_h+24，可能跨越 24）
    t_axis = start_h + np.arange(FUTURE_STEPS) * STEP_MIN / 60.0  # 6.0 ~ 29.83

    # 白天区域：10:00-18:00，在 t_axis 坐标系里就是 10 和 18
    ax.axvspan(10, 18, alpha=0.10, color='yellow', label='Daytime (10:00-18:00)', zorder=1)

    ax.plot(t_axis, y_true[idx],   'k-',  linewidth=2.2, label='Actual',            zorder=3)
    ax.plot(t_axis, lr_preds[idx], 'r--', linewidth=1.8, label='Linear Regression',  zorder=2, alpha=0.9)

    mae_i = mean_absolute_error(y_true[idx], lr_preds[idx])
    ax.set_title(f'{start_dt.strftime("%Y-%m-%d")}  (start 06:00)\nMAE = {mae_i:.2f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Hour of day', fontsize=9)
    ax.set_ylabel('Capacity Factor (%)', fontsize=9)

    # x 轴：6, 12, 18, 24(=0), 30(=6)
    tick_vals   = [start_h + h for h in range(0, 25, 6)]        # [6,12,18,24,30]
    tick_labels = [f'{int(v % 24):02d}:00' for v in tick_vals]  # 06,12,18,00,06
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlim(start_h, start_h + 24)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

plt.suptitle('Linear Regression — 06:00 Forecast Start\nProject1033',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(BASE, 'linear_specific_days.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'\n✅ 图片已保存: {save_path}')
