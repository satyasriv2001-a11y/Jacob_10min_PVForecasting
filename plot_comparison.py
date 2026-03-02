"""
三模型对比图：Actual vs Linear vs XGBoost
优先从 project1033_cache.pkl 加载（已有则直接用，否则重新生成）
"""
import os, sys, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

CACHE_PATH = os.path.join(BASE, 'project1033_cache.pkl')

# ── 加载或生成 Project1033 专属 cache ────────────────────────────────────
if os.path.exists(CACHE_PATH):
    print("加载 project1033_cache.pkl...")
    cache = joblib.load(CACHE_PATH)
else:
    print("未找到 project1033_cache.pkl，重新生成（约 4 分钟）...")
    from data.data_utils import preprocess_features, create_sliding_windows
    config = {
        'model': 'XGB', 'model_complexity': 'low',
        'use_pv': True, 'use_hist_weather': False, 'use_forecast': True,
        'use_ideal_nwp': False, 'use_time_encoding': True,
        'weather_category': 'all_weather',
        'past_hours': 24, 'future_hours': 24,
        'start_date': '2022-01-01', 'end_date': '2024-09-28',
    }
    df_raw = __import__('pandas').read_csv(os.path.join(BASE, 'data', 'Project1033.csv'))
    df_raw['Datetime'] = __import__('pandas').to_datetime(df_raw['date'])
    df_raw = df_raw[df_raw['Datetime'].dt.year >= 2022].reset_index(drop=True)
    night = (df_raw['Datetime'].dt.hour < 5) & (df_raw['global_tilted_irradiance_pred'].fillna(0) > 5)
    df_raw = df_raw[~night].reset_index(drop=True)

    df_clean, hist_feats, fcst_feats, _, _, scaler_target, no_hist_power = \
        preprocess_features(df_raw.copy(), config)
    Xh, Xf, y, hours, dates = create_sliding_windows(
        df_clean, config['past_hours'], config['future_hours'],
        hist_feats, fcst_feats, no_hist_power)
    Xh = Xh.astype('float32'); Xf = Xf.astype('float32'); y = y.astype('float32')

    # XGBoost 用自己的 scaler（与训练时一致）
    xgb_pkl_tmp = joblib.load(os.path.join(BASE, 'xgb_low_results', 'xgb_model_full.pkl'))
    scaler_target = xgb_pkl_tmp['scaler']

    train_end = int(len(Xh) * 0.85)
    cache = {'Xh': Xh, 'Xf': Xf, 'y': y, 'dates': dates,
             'scaler_target': scaler_target, 'train_end': train_end}
    joblib.dump(cache, CACHE_PATH)
    print(f"  已保存到 {CACHE_PATH}")

Xh            = cache['Xh']
Xf            = cache['Xf']
y             = cache['y']
dates         = cache['dates']
scaler_target = cache['scaler_target']
train_end     = cache['train_end']
print(f"  总样本 {len(Xh):,}  train_end={train_end:,}  test={len(Xh)-train_end:,}")

# ── 拆分 train/test ───────────────────────────────────────────────────────
def flatten(Xh, Xf):
    h = Xh.reshape(Xh.shape[0], -1)
    f = Xf.reshape(Xf.shape[0], -1)
    return np.concatenate([h, f], axis=1)

X_train = flatten(Xh[:train_end], Xf[:train_end])
X_test  = flatten(Xh[train_end:], Xf[train_end:])
y_train = y[:train_end].reshape(len(Xh[:train_end]), -1)
y_test  = y[train_end:].reshape(len(Xh[train_end:]), -1)
dates_test = dates[train_end:]
FUTURE_STEPS = y.shape[1]  # 144

# ── 反归一化工具 ──────────────────────────────────────────────────────────
def inverse(preds_flat):
    return scaler_target.inverse_transform(
        preds_flat.reshape(-1, 1)
    ).reshape(-1, FUTURE_STEPS)

y_true = inverse(y_test)

# ── 加载 XGBoost 模型 + 推理 ──────────────────────────────────────────────
print("\n加载 XGBoost 模型...")
xgb_pkl = joblib.load(os.path.join(BASE, 'xgb_low_results', 'xgb_model_full.pkl'))
xgb_model = xgb_pkl['model']   # MultiOutputRegressor
print("  XGBoost 推理中...")
xgb_preds = np.clip(inverse(xgb_model.predict(X_test)), 0, 100)
print("  完成")

# ── 加载或训练线性回归 ────────────────────────────────────────────────────
LR_MODEL_PATH = os.path.join(BASE, 'experiments', 'lza1033', 'models', 'linear_model.pkl')
if os.path.exists(LR_MODEL_PATH):
    print("\n加载 Linear Regression 模型...")
    lr_pkl = joblib.load(LR_MODEL_PATH)
    lr = lr_pkl['model']
    print("  完成")
else:
    print("\n训练 Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump({'model': lr, 'scaler': scaler_target, 'train_end': train_end}, LR_MODEL_PATH)
    print(f"  完成，已保存到 {LR_MODEL_PATH}")
lr_preds = np.clip(inverse(lr.predict(X_test)), 0, 100)

# ── 指标 ─────────────────────────────────────────────────────────────────
xgb_mae = mean_absolute_error(y_true.flatten(), xgb_preds.flatten())
xgb_r2  = r2_score(y_true.flatten(), xgb_preds.flatten())
lin_mae = mean_absolute_error(y_true.flatten(), lr_preds.flatten())
lin_r2  = r2_score(y_true.flatten(), lr_preds.flatten())
print(f"\n  XGBoost — MAE={xgb_mae:.4f}  R²={xgb_r2:.4f}")
print(f"  Linear  — MAE={lin_mae:.4f}  R²={lin_r2:.4f}")

# ── 画图 ──────────────────────────────────────────────────────────────────
t_axis = np.arange(FUTURE_STEPS) * 10 / 60
n_test = len(y_true)
sample_indices = [int(n_test * r) for r in [0.05, 0.2, 0.4, 0.6, 0.75, 0.9]]

fig, axes = plt.subplots(3, 2, figsize=(18, 13))
axes = axes.flatten()

for plot_i, idx in enumerate(sample_indices):
    ax = axes[plot_i]
    ax.plot(t_axis, y_true[idx],    'k-',  linewidth=2.2, label='实际值',       zorder=4)
    ax.plot(t_axis, xgb_preds[idx], 'b--', linewidth=1.8, label='XGBoost 预测', zorder=3, alpha=0.9)
    ax.plot(t_axis, lr_preds[idx],  'r:',  linewidth=1.8, label='Linear 预测',  zorder=2, alpha=0.85)
    ax.axvspan(10, 18, alpha=0.07, color='yellow', label='白天(10am-6pm)')

    mae_xgb = np.mean(np.abs(y_true[idx] - xgb_preds[idx]))
    mae_lin = np.mean(np.abs(y_true[idx] - lr_preds[idx]))
    date_lbl = str(dates_test[idx]) if idx < len(dates_test) else f'样本{idx}'
    ax.set_title(f'{date_lbl}  |  XGB={mae_xgb:.2f}  Lin={mae_lin:.2f}', fontsize=10)
    ax.set_xlabel('预测时间 (h)')
    ax.set_ylabel('容量因子 (%)')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    if plot_i == 0:
        ax.legend(loc='upper right', fontsize=9)

plt.suptitle(
    f'三模型对比 — Project1033\n'
    f'XGBoost (low): MAE={xgb_mae:.4f}  R²={xgb_r2:.4f}    '
    f'Linear: MAE={lin_mae:.4f}  R²={lin_r2:.4f}',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
save_path = os.path.join(BASE, 'xgb_vs_linear_3curves.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'\n✅ 图片已保存: {save_path}')
