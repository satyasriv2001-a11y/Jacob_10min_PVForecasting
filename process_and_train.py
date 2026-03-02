"""
将 .tab 文件转换为 CSV，然后训练线性回归模型
用法：python process_and_train.py Project283 13245157 Project293 13245224
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

BASE     = '/Users/terry/Library/CloudStorage/GoogleDrive-zl2268@cornell.edu/.shortcut-targets-by-id/1tYXd7zscd3BkVOz6B4y72IxqdsQEKgzW/SolarPrediction'
DATA_DIR = os.path.join(BASE, 'data')
EXP_DIR  = os.path.join(BASE, 'experiments')
sys.path.insert(0, BASE)

from data.data_utils import preprocess_features, create_sliding_windows

# ── 目标场站 ──────────────────────────────────────────────────────────────────
TARGETS = [
    ('Project283', '13245157'),
    ('Project293', '13245224'),
]

config = {
    'model': 'Linear', 'model_complexity': 'high',
    'use_pv': True, 'use_hist_weather': False, 'use_forecast': True,
    'use_ideal_nwp': False, 'use_time_encoding': True,
    'weather_category': 'all_weather',
    'past_hours': 24, 'future_hours': 24,
    'start_date': '2022-01-01', 'end_date': '2024-09-28',
}
FUTURE_STEPS = 144
STEP_MIN     = 10

def flatten(Xh, Xf):
    h = Xh.reshape(Xh.shape[0], -1)
    if Xf is not None:
        return np.concatenate([h, Xf.reshape(Xf.shape[0], -1)], axis=1)
    return h

def calc_metrics(yt, yp):
    return dict(
        mae  = float(np.mean(np.abs(yt - yp))),
        rmse = float(np.sqrt(np.mean((yt - yp)**2))),
        r2   = float(r2_score(yt.flatten(), yp.flatten()))
    )

summary_rows = []

for proj, fid in TARGETS:
    print(f'\n{"="*60}')
    print(f'  {proj}  (fileId={fid})')
    print(f'{"="*60}')
    t_proj = time.time()

    # ── Step 1: .tab → CSV ────────────────────────────────────────────────────
    tab_path = f'/tmp/new_{fid}.tab'
    csv_path = os.path.join(DATA_DIR, f'{proj}.csv')

    if not os.path.exists(csv_path):
        print(f'  转换 {tab_path} → {csv_path}')
        df_tab = pd.read_csv(tab_path, sep='\t', low_memory=False)
        # 去重（DST重复时间戳）
        if 'date' in df_tab.columns:
            df_tab['date'] = pd.to_datetime(df_tab['date'], errors='coerce')
            df_tab = df_tab.drop_duplicates(subset=['date'], keep='first')
        df_tab.to_csv(csv_path, index=False)
        print(f'  → 保存完成，行数: {len(df_tab):,}')
    else:
        print(f'  CSV已存在，跳过转换')

    # ── Step 2: 创建实验文件夹 ─────────────────────────────────────────────────
    proj_dir   = os.path.join(EXP_DIR, proj)
    model_dir  = os.path.join(proj_dir, 'models')
    result_dir = os.path.join(proj_dir, 'results')
    data_dir   = os.path.join(proj_dir, 'data')
    for d in [model_dir, result_dir, data_dir]:
        os.makedirs(d, exist_ok=True)

    # 复制CSV到实验文件夹
    import shutil
    shutil.copy2(csv_path, os.path.join(data_dir, f'{proj}.csv'))

    # ── Step 3: 加载数据 ───────────────────────────────────────────────────────
    df_raw = pd.read_csv(csv_path)
    df_raw['Datetime'] = pd.to_datetime(df_raw['date'], utc=True).dt.tz_convert(None)
    df_raw = df_raw[df_raw['Datetime'].dt.year >= 2022].reset_index(drop=True)
    night_anom = (df_raw['Datetime'].dt.hour < 5) & \
                 (df_raw['global_tilted_irradiance_pred'].fillna(0) > 5)
    df_raw = df_raw[~night_anom].reset_index(drop=True)
    print(f'  数据行数: {len(df_raw):,}')

    # ── Step 4: 特征工程 + 滑动窗口 ───────────────────────────────────────────
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
    print(f'  Xh={Xh.shape}  Xf={Xf.shape}  y={y.shape}')

    # ── Step 5: 训练/测试划分 + 训练 ──────────────────────────────────────────
    total     = len(Xh)
    train_end = int(total * 0.85)
    X_train   = flatten(Xh[:train_end], Xf[:train_end] if Xf is not None else None)
    X_test    = flatten(Xh[train_end:], Xf[train_end:] if Xf is not None else None)
    y_train   = y[:train_end].reshape(train_end, -1)
    y_test    = y[train_end:]
    dates_test = dates[train_end:]
    print(f'  训练={train_end:,}  测试={total-train_end:,}')

    t0    = time.time()
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    train_sec = time.time() - t0
    print(f'  训练完成  {train_sec:.1f}s')

    # ── Step 6: 预测 + 评估 ───────────────────────────────────────────────────
    preds_raw = model.predict(X_test)
    y_true = scaler_target.inverse_transform(
        y_test.reshape(-1, 1)).reshape(-1, FUTURE_STEPS)
    y_pred = np.clip(
        scaler_target.inverse_transform(
            preds_raw.reshape(-1, 1)).reshape(-1, FUTURE_STEPS), 0, 100)

    m_all = calc_metrics(y_true, y_pred)
    m_day = calc_metrics(y_true[:, 60:108], y_pred[:, 60:108])
    print(f'  [全天] MAE={m_all["mae"]:.4f}  RMSE={m_all["rmse"]:.4f}  R²={m_all["r2"]:.4f}')
    print(f'  [白天] MAE={m_day["mae"]:.4f}  RMSE={m_day["rmse"]:.4f}  R²={m_day["r2"]:.4f}')

    # ── Step 7: 保存模型 + 结果 ───────────────────────────────────────────────
    joblib.dump({'model': model, 'scaler_target': scaler_target},
                os.path.join(model_dir, 'linear_model.pkl'))

    offset = pd.Timedelta(minutes=STEP_MIN * (FUTURE_STEPS - 1))
    forecast_starts = [
        (pd.Timestamp(dates_test[i]) - offset).strftime('%Y-%m-%d %H:%M')
        for i in range(len(y_pred))
    ]
    pred_df = pd.DataFrame(
        {'forecast_start': forecast_starts,
         **{f'predicted_{k}': np.round(y_pred[:, k], 4) for k in range(FUTURE_STEPS)}}
    )
    pred_df.to_csv(os.path.join(result_dir, 'predictions_144steps.csv'), index=False)

    pd.DataFrame([{
        'project': proj, 'n_train': train_end, 'n_test': total-train_end,
        'train_sec': round(train_sec, 1),
        'mae_all': round(m_all['mae'], 4), 'rmse_all': round(m_all['rmse'], 4),
        'r2_all': round(m_all['r2'], 4),
        'mae_day': round(m_day['mae'], 4), 'rmse_day': round(m_day['rmse'], 4),
        'r2_day': round(m_day['r2'], 4),
    }]).to_csv(os.path.join(result_dir, 'metrics.csv'), index=False)

    summary_rows.append({'project': proj, **m_all,
                         'mae_day': m_day['mae'], 'r2_day': m_day['r2']})
    print(f'  完成，总耗时 {time.time()-t_proj:.1f}s')

    del Xh, Xf, y, X_train, X_test, y_true, y_pred; gc.collect()

print(f'\n{"="*60}\n全部完成')
for r in summary_rows:
    print(f"  {r['project']}  MAE={r['mae']:.4f}  R²={r['r2']:.4f}")
