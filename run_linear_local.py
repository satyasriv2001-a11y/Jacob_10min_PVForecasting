"""
本地运行线性回归基线（不依赖Colab/GPU）
场站: Project1033，2022-01-01 ~ 2024-09-28
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无头模式，不弹窗
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from data.data_utils import preprocess_features, create_sliding_windows

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
config = {
    'model': 'Linear',
    'model_complexity': 'high',
    'use_pv': True,
    'use_hist_weather': False,
    'use_forecast': True,
    'use_ideal_nwp': False,
    'use_time_encoding': True,
    'weather_category': 'all_weather',
    'past_hours': 24,
    'future_hours': 24,
    'start_date': '2022-01-01',
    'end_date': '2024-09-28',
}
FUTURE_STEPS = config['future_hours'] * 6   # 144
PAST_STEPS   = config['past_hours']   * 6   # 144

# ─────────────────────────────────────────────
# Step 1: 加载数据 + 过滤
# ─────────────────────────────────────────────
print('='*55)
print('Step 1: 加载数据')
print('='*55)
data_path = os.path.join(PROJECT_DIR, 'data', 'Project1033.csv')
t0 = time.time()
df_raw = pd.read_csv(data_path)
df_raw['Datetime'] = pd.to_datetime(df_raw['date'])

# Filter 1: 2022+
before = len(df_raw)
df_raw = df_raw[df_raw['Datetime'].dt.year >= 2022].reset_index(drop=True)
print(f'Filter 1 (2022+): {before:,} → {len(df_raw):,} 行')

# Filter 2: 夜间irradiance异常
night_anom = (df_raw['Datetime'].dt.hour < 5) & \
             (df_raw['global_tilted_irradiance_pred'].fillna(0) > 5)
df_raw = df_raw[~night_anom].reset_index(drop=True)
print(f'Filter 2 (夜间异常): 去除 {night_anom.sum()} 行')
print(f'最终数据: {len(df_raw):,} 行，加载耗时 {time.time()-t0:.1f}s')

# ─────────────────────────────────────────────
# Step 2: 特征工程 + 滑动窗口
# ─────────────────────────────────────────────
print('\n' + '='*55)
print('Step 2: 特征工程 + 滑动窗口')
print('='*55)
t0 = time.time()

df_clean, hist_feats, fcst_feats, _, _, scaler_target, no_hist_power = \
    preprocess_features(df_raw.copy(), config)
del df_raw; gc.collect()

Xh, Xf, y, hours, dates = create_sliding_windows(
    df_clean, config['past_hours'], config['future_hours'],
    hist_feats, fcst_feats, no_hist_power
)
del df_clean; gc.collect()

Xh = Xh.astype(np.float32)
Xf = Xf.astype(np.float32) if Xf is not None else None
y  = y.astype(np.float32)

print(f'Xh: {Xh.shape}, Xf: {Xf.shape}, y: {y.shape}')
print(f'特征工程耗时: {time.time()-t0:.1f}s')

# ─────────────────────────────────────────────
# Step 3: 训练/测试集划分（按时序，85/15）
# ─────────────────────────────────────────────
total     = len(Xh)
train_end = int(total * 0.85)

Xh_train, Xh_test = Xh[:train_end], Xh[train_end:]
Xf_train = Xf[:train_end] if Xf is not None else None
Xf_test  = Xf[train_end:] if Xf is not None else None
y_train, y_test   = y[:train_end], y[train_end:]
dates_test = dates[train_end:]

print(f'\n总样本: {total:,}  训练: {train_end:,} (85%)  测试: {total-train_end:,} (15%)')

# ─────────────────────────────────────────────
# Step 4: 训练线性回归
# ─────────────────────────────────────────────
print('\n' + '='*55)
print('Step 4: 训练 Linear Regression')
print('='*55)

def flatten(Xh, Xf):
    h = Xh.reshape(Xh.shape[0], -1)
    if Xf is not None:
        f = Xf.reshape(Xf.shape[0], -1)
        return np.concatenate([h, f], axis=1)
    return h

X_train = flatten(Xh_train, Xf_train)
X_test  = flatten(Xh_test,  Xf_test)
y_train_flat = y_train.reshape(y_train.shape[0], -1)
y_test_flat  = y_test.reshape(y_test.shape[0], -1)

print(f'输入维度: {X_train.shape[1]}  输出维度: {FUTURE_STEPS}')
print(f'训练样本: {len(X_train):,}')

t_start = time.time()
model = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train_flat)
train_time = time.time() - t_start
print(f'✅ 训练完成，耗时: {train_time:.1f}s')

# ─────────────────────────────────────────────
# Step 5: 评估
# ─────────────────────────────────────────────
print('\n' + '='*55)
print('Step 5: 评估指标')
print('='*55)

preds_flat = model.predict(X_test)

# 反归一化
y_true = scaler_target.inverse_transform(y_test_flat).reshape(-1, FUTURE_STEPS)
y_pred = scaler_target.inverse_transform(preds_flat.reshape(-1, 1)).reshape(-1, FUTURE_STEPS)
y_pred = np.clip(y_pred, 0, 100)

def calc_metrics(yt, yp, label='整体'):
    mae   = np.mean(np.abs(yt - yp))
    rmse  = np.sqrt(np.mean((yt - yp)**2))
    r2    = r2_score(yt.flatten(), yp.flatten())
    denom = (np.abs(yt) + np.abs(yp)) / 2
    smape = np.mean(np.where(denom > 0, np.abs(yt - yp) / denom, 0)) * 100
    print(f'  [{label}]  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  sMAPE={smape:.2f}%')
    return dict(mae=mae, rmse=rmse, r2=r2, smape=smape)

m_all = calc_metrics(y_true, y_pred, '全天 24h')

# 白天指标（10am-6pm = step 60~107）
day_slice = slice(60, 108)
m_day = calc_metrics(y_true[:, day_slice], y_pred[:, day_slice], '白天10-18h')

# ─────────────────────────────────────────────
# Step 6: 可视化
# ─────────────────────────────────────────────
print('\n生成图表...')
t_axis = np.arange(FUTURE_STEPS) * 10 / 60

# 图1: 时序对比（6个样本）
n_test = len(y_true)
sample_indices = [int(n_test * r) for r in [0.05, 0.2, 0.4, 0.6, 0.75, 0.9]]
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for plot_i, idx in enumerate(sample_indices):
    ax = axes.flatten()[plot_i]
    ax.plot(t_axis, y_true[idx], 'k-',  lw=2,   label='实际值')
    ax.plot(t_axis, y_pred[idx], 'r--', lw=1.5, label='预测值', alpha=0.85)
    ax.fill_between(t_axis, y_true[idx], y_pred[idx], alpha=0.15, color='red')
    ax.axvspan(10, 18, alpha=0.07, color='yellow')
    mae_i = np.mean(np.abs(y_true[idx] - y_pred[idx]))
    ax.set_title(f'{dates_test[idx]}  MAE={mae_i:.3f}', fontsize=10)
    ax.set_xlim(0, 24); ax.set_ylim(bottom=0)
    ax.set_xlabel('时间 (h)'); ax.set_ylabel('CF (%)')
    ax.grid(True, alpha=0.3)
    if plot_i == 0:
        ax.legend(fontsize=9)
plt.suptitle(f'Project1033 Linear Regression 24h预测\nMAE={m_all["mae"]:.4f}  R²={m_all["r2"]:.4f}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
p1 = os.path.join(PROJECT_DIR, 'linear_timeseries_1033.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f'  保存: {p1}')

# 图2: 散点图
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
np.random.seed(42)
idx_s = np.random.choice(y_true.size, min(8000, y_true.size), replace=False)
yt_f, yp_f = y_true.flatten()[idx_s], y_pred.flatten()[idx_s]
step_f = np.tile(np.arange(FUTURE_STEPS), n_test).flatten()[idx_s]
is_day = (step_f >= 60) & (step_f < 108)

for ax, mask, title in zip(axes,
    [np.ones(len(yt_f), bool), is_day],
    ['全天', '白天(10-18h)']):
    r2_v = r2_score(yt_f[mask], yp_f[mask])
    ax.scatter(yt_f[~mask & ~is_day], yp_f[~mask & ~is_day], s=3, alpha=0.3, c='steelblue', label='夜间')
    ax.scatter(yt_f[mask & is_day],   yp_f[mask & is_day],   s=3, alpha=0.4, c='orange',    label='白天')
    lim = max(yt_f.max(), yp_f.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'r-', lw=1.5, label='y=x')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('实际值'); ax.set_ylabel('预测值')
    ax.set_title(f'{title}  R²={r2_v:.4f}')
    ax.legend(markerscale=3, fontsize=9); ax.grid(True, alpha=0.3)
plt.suptitle('Project1033 — 散点图', fontsize=13, fontweight='bold')
plt.tight_layout()
p2 = os.path.join(PROJECT_DIR, 'linear_scatter_1033.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f'  保存: {p2}')

# 图3: 误差分布
errors = (y_pred - y_true).flatten()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(errors, bins=100, color='steelblue', alpha=0.7, edgecolor='white')
axes[0].axvline(0, color='red', lw=2, ls='--', label='误差=0')
axes[0].axvline(errors.mean(), color='orange', lw=2, ls='--', label=f'均值={errors.mean():.3f}')
axes[0].set_title('误差分布直方图'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

mae_per_step = np.mean(np.abs(y_pred - y_true), axis=0)
axes[1].fill_between(t_axis, mae_per_step, alpha=0.6, color='steelblue')
axes[1].plot(t_axis, mae_per_step, color='navy', lw=1.5)
axes[1].axvspan(10, 18, alpha=0.12, color='yellow', label='白天')
axes[1].set_xlabel('时间 (h)'); axes[1].set_ylabel('MAE')
axes[1].set_title(f'每时刻MAE  峰值={mae_per_step.max():.3f}@{t_axis[mae_per_step.argmax()]:.1f}h')
axes[1].set_xlim(0, 24); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle('Project1033 — 误差分析', fontsize=13, fontweight='bold')
plt.tight_layout()
p3 = os.path.join(PROJECT_DIR, 'linear_error_1033.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f'  保存: {p3}')

# ─────────────────────────────────────────────
# 最终汇总
# ─────────────────────────────────────────────
print('\n' + '='*55)
print('Project1033 Linear Regression 结果汇总')
print('='*55)
print(f'  场站        : Project1033（正常场站，无clipping）')
print(f'  训练时间    : {train_time:.1f}s')
print(f'  训练样本    : {train_end:,}')
print(f'  测试样本    : {total-train_end:,}')
print(f'  [全天]  MAE={m_all["mae"]:.4f}  RMSE={m_all["rmse"]:.4f}  R²={m_all["r2"]:.4f}  sMAPE={m_all["smape"]:.2f}%')
print(f'  [白天]  MAE={m_day["mae"]:.4f}  RMSE={m_day["rmse"]:.4f}  R²={m_day["r2"]:.4f}  sMAPE={m_day["smape"]:.2f}%')
print(f'\n图表已保存:')
print(f'  {p1}')
print(f'  {p2}')
print(f'  {p3}')
