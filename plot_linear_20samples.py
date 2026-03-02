"""
线性回归模型：真实值 vs 预测值对比图（20个样本）
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

BASE = '/Users/terry/Library/CloudStorage/GoogleDrive-zl2268@cornell.edu/.shortcut-targets-by-id/1tYXd7zscd3BkVOz6B4y72IxqdsQEKgzW/SolarPrediction'

# ── 加载缓存 ──────────────────────────────────────────────────────────────────
CACHE_PATH = os.path.join(BASE, 'project1033_cache.pkl')
print("加载 project1033_cache.pkl...")
cache = joblib.load(CACHE_PATH)

Xh            = cache['Xh']
Xf            = cache['Xf']
y             = cache['y']
dates         = cache['dates']
scaler_target = cache['scaler_target']
train_end     = cache['train_end']
print(f"  总样本 {len(Xh):,}  train_end={train_end:,}  test={len(Xh)-train_end:,}")

# ── 拆分 test ─────────────────────────────────────────────────────────────────
def flatten(Xh, Xf):
    h = Xh.reshape(Xh.shape[0], -1)
    f = Xf.reshape(Xf.shape[0], -1)
    return np.concatenate([h, f], axis=1)

X_test     = flatten(Xh[train_end:], Xf[train_end:])
y_test     = y[train_end:]
dates_test = dates[train_end:]
FUTURE_STEPS = y.shape[1]  # 144

# ── 反归一化 ──────────────────────────────────────────────────────────────────
def inverse(arr):
    return scaler_target.inverse_transform(
        arr.reshape(-1, 1)
    ).reshape(-1, FUTURE_STEPS)

y_true = inverse(y_test)

# ── 加载线性模型 + 推理 ───────────────────────────────────────────────────────
LR_MODEL_PATH = os.path.join(BASE, 'experiments', 'lza1033', 'models', 'linear_model.pkl')
print("加载 Linear Regression 模型...")
lr_pkl = joblib.load(LR_MODEL_PATH)
lr = lr_pkl['model']
print("  推理中...")
lr_preds = np.clip(inverse(lr.predict(X_test)), 0, 100)
print("  完成")

# ── 整体指标 ──────────────────────────────────────────────────────────────────
mae_all = mean_absolute_error(y_true.flatten(), lr_preds.flatten())
r2_all  = r2_score(y_true.flatten(), lr_preds.flatten())
print(f"  Linear — MAE={mae_all:.4f}  R²={r2_all:.4f}")

# ── 时间轴工具 ────────────────────────────────────────────────────────────────
import pandas as pd

STEP_MIN = 10   # 每步10分钟
n_test   = len(y_true)

# 找出预测窗口从 00:00 开始（起始小时=0）的样本，再均匀取20个
midnight_indices = []
for i in range(n_test):
    try:
        end_dt   = pd.Timestamp(dates_test[i])
        start_dt = end_dt - pd.Timedelta(minutes=STEP_MIN * (FUTURE_STEPS - 1))
        if start_dt.hour == 0 and start_dt.minute == 0:
            midnight_indices.append(i)
    except Exception:
        pass

print(f"  找到从00:00开始的窗口: {len(midnight_indices)} 个")

# 均匀取20个
if len(midnight_indices) >= 20:
    picks = np.linspace(0, len(midnight_indices) - 1, 20, dtype=int)
    sample_indices = [midnight_indices[p] for p in picks]
else:
    # 不够20个时，降级为均匀取所有测试样本
    print("  警告：00:00起始样本不足20个，改用所有测试样本均匀抽取")
    sample_indices = [int(n_test * r) for r in np.linspace(0.02, 0.98, 20)]

t_axis = np.arange(FUTURE_STEPS) * STEP_MIN / 60.0   # 0~23.83 h（相对时间）

# ── 画图：4行×5列 ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 5, figsize=(26, 17))
axes = axes.flatten()

for plot_i, idx in enumerate(sample_indices):
    ax = axes[plot_i]

    try:
        end_dt   = pd.Timestamp(dates_test[idx])
        start_dt = end_dt - pd.Timedelta(minutes=STEP_MIN * (FUTURE_STEPS - 1))
        date_lbl = start_dt.strftime('%Y-%m-%d')
    except Exception:
        date_lbl = str(dates_test[idx]) if idx < len(dates_test) else f'Sample {idx}'

    ax.plot(t_axis, y_true[idx],   'k-',  linewidth=2.0, label='Actual',           zorder=3)
    ax.plot(t_axis, lr_preds[idx], 'r--', linewidth=1.8, label='Linear Regression', zorder=2, alpha=0.85)
    ax.axvspan(10, 18, alpha=0.07, color='yellow', label='Daytime (10:00-18:00)')

    mae_i = np.mean(np.abs(y_true[idx] - lr_preds[idx]))
    ax.set_title(f'{date_lbl}  |  MAE={mae_i:.2f}', fontsize=8.5)
    ax.set_xlabel('Hour of day', fontsize=7)
    ax.set_ylabel('Capacity Factor (%)', fontsize=7)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xticklabels([f'{h}:00' for h in range(0, 25, 6)], fontsize=6)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    if plot_i == 0:
        ax.legend(loc='upper right', fontsize=7)

plt.suptitle(
    f'Linear Regression — Actual vs Predicted (20 Test Samples)\n'
    f'Overall MAE={mae_all:.4f}   R²={r2_all:.4f}',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
save_path = os.path.join(BASE, 'linear_20samples.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'\n✅ 图片已保存: {save_path}')
