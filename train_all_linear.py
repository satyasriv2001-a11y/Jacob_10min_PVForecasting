"""
批量训练 13 个场站的线性回归模型（复制 Project1033 流程）
输出结构（每个场站）:
  experiments/ProjectXXXX/
    data/ProjectXXXX.csv
    models/linear_model.pkl
    results/predictions_144steps.csv   ← forecast_start + predicted_0~143 (145列)
    results/metrics.csv                ← MAE/RMSE/R² 汇总
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE     = '/Users/terry/Library/CloudStorage/GoogleDrive-zl2268@cornell.edu/.shortcut-targets-by-id/1tYXd7zscd3BkVOz6B4y72IxqdsQEKgzW/SolarPrediction'
DATA_DIR = os.path.join(BASE, 'data')
EXP_DIR  = os.path.join(BASE, 'experiments')
sys.path.insert(0, BASE)

from data.data_utils import preprocess_features, create_sliding_windows

# ── 13 个场站（去除 Project1140）───────────────────────────────────────────────
PROJECTS = [
    'Project1033', 'Project1070', 'Project1077', 'Project1081', 'Project1084',
    'Project1137', 'Project171',  'Project172',  'Project186',  'Project212',
    'Project213',  'Project240',  'Project249',
]

# ── 训练配置（同 1033）────────────────────────────────────────────────────────
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
FUTURE_STEPS = config['future_hours'] * 6  # 144
STEP_MIN     = 10

def flatten(Xh, Xf):
    h = Xh.reshape(Xh.shape[0], -1)
    if Xf is not None:
        return np.concatenate([h, Xf.reshape(Xf.shape[0], -1)], axis=1)
    return h

def calc_metrics(yt, yp):
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp)**2)))
    r2   = float(r2_score(yt.flatten(), yp.flatten()))
    return dict(mae=mae, rmse=rmse, r2=r2)

# ── 汇总记录 ──────────────────────────────────────────────────────────────────
summary_rows = []

# ═════════════════════════════════════════════════════════════════════════════
for proj in PROJECTS:
    print(f'\n{"="*60}')
    print(f'  {proj}')
    print(f'{"="*60}')
    t_proj = time.time()

    # ── 创建文件夹 ────────────────────────────────────────────────────────────
    proj_dir     = os.path.join(EXP_DIR, proj)
    data_out_dir = os.path.join(proj_dir, 'data')
    model_dir    = os.path.join(proj_dir, 'models')
    result_dir   = os.path.join(proj_dir, 'results')
    for d in [data_out_dir, model_dir, result_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    src_csv = os.path.join(DATA_DIR, f'{proj}.csv')
    dst_csv = os.path.join(data_out_dir, f'{proj}.csv')
    if not os.path.exists(dst_csv):
        import shutil; shutil.copy2(src_csv, dst_csv)

    df_raw = pd.read_csv(src_csv)
    df_raw['Datetime'] = pd.to_datetime(df_raw['date'])
    df_raw = df_raw[df_raw['Datetime'].dt.year >= 2022].reset_index(drop=True)

    # 过滤夜间irradiance异常
    night_anom = (df_raw['Datetime'].dt.hour < 5) & \
                 (df_raw['global_tilted_irradiance_pred'].fillna(0) > 5)
    df_raw = df_raw[~night_anom].reset_index(drop=True)
    print(f'  数据行数: {len(df_raw):,}')

    # ── 特征工程 + 滑动窗口 ───────────────────────────────────────────────────
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

    # ── 训练/测试划分（85/15）────────────────────────────────────────────────
    total     = len(Xh)
    train_end = int(total * 0.85)
    X_train   = flatten(Xh[:train_end], Xf[:train_end] if Xf is not None else None)
    X_test    = flatten(Xh[train_end:], Xf[train_end:] if Xf is not None else None)
    y_train   = y[:train_end].reshape(train_end, -1)
    y_test    = y[train_end:]
    dates_test = dates[train_end:]
    print(f'  训练={train_end:,}  测试={total-train_end:,}')

    # ── 训练线性回归 ──────────────────────────────────────────────────────────
    t0    = time.time()
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    train_sec = time.time() - t0
    print(f'  训练完成  {train_sec:.1f}s')

    # ── 预测 + 反归一化 ───────────────────────────────────────────────────────
    preds_raw = model.predict(X_test)
    y_true = scaler_target.inverse_transform(
        y_test.reshape(-1, 1)).reshape(-1, FUTURE_STEPS)
    y_pred = np.clip(
        scaler_target.inverse_transform(
            preds_raw.reshape(-1, 1)).reshape(-1, FUTURE_STEPS),
        0, 100)

    # ── 评估指标 ──────────────────────────────────────────────────────────────
    m_all = calc_metrics(y_true, y_pred)
    m_day = calc_metrics(y_true[:, 60:108], y_pred[:, 60:108])
    print(f'  [全天] MAE={m_all["mae"]:.4f}  RMSE={m_all["rmse"]:.4f}  R²={m_all["r2"]:.4f}')
    print(f'  [白天] MAE={m_day["mae"]:.4f}  RMSE={m_day["rmse"]:.4f}  R²={m_day["r2"]:.4f}')

    # ── 保存模型 ──────────────────────────────────────────────────────────────
    model_path = os.path.join(model_dir, 'linear_model.pkl')
    joblib.dump({'model': model, 'scaler_target': scaler_target}, model_path)

    # ── 保存 predictions_144steps.csv ────────────────────────────────────────
    offset = pd.Timedelta(minutes=STEP_MIN * (FUTURE_STEPS - 1))
    forecast_starts = [
        (pd.Timestamp(dates_test[i]) - offset).strftime('%Y-%m-%d %H:%M')
        for i in range(len(y_pred))
    ]
    pred_df = pd.DataFrame(
        {'forecast_start': forecast_starts,
         **{f'predicted_{k}': np.round(y_pred[:, k], 4) for k in range(FUTURE_STEPS)}}
    )
    pred_path = os.path.join(result_dir, 'predictions_144steps.csv')
    pred_df.to_csv(pred_path, index=False)

    # ── 保存 metrics.csv ──────────────────────────────────────────────────────
    metrics_df = pd.DataFrame([{
        'project': proj,
        'n_train': train_end,
        'n_test': total - train_end,
        'train_sec': round(train_sec, 1),
        'mae_all':  round(m_all['mae'],  4),
        'rmse_all': round(m_all['rmse'], 4),
        'r2_all':   round(m_all['r2'],   4),
        'mae_day':  round(m_day['mae'],  4),
        'rmse_day': round(m_day['rmse'], 4),
        'r2_day':   round(m_day['r2'],   4),
    }])
    metrics_df.to_csv(os.path.join(result_dir, 'metrics.csv'), index=False)

    summary_rows.append(metrics_df.iloc[0].to_dict())
    print(f'  [{proj}] 完成，总耗时 {time.time()-t_proj:.1f}s')
    print(f'  → {pred_path}  ({len(pred_df):,} rows × {len(pred_df.columns)} cols)')

    del Xh, Xf, y, X_train, X_test, y_train, y_test, y_true, y_pred
    gc.collect()

# ═════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print('全部完成！汇总：')
print(f'{"="*60}')
summary = pd.DataFrame(summary_rows)
print(summary[['project','n_test','mae_all','rmse_all','r2_all','mae_day','r2_day']].to_string(index=False))

# 保存全局汇总
summary.to_csv(os.path.join(EXP_DIR, 'all_projects_metrics.csv'), index=False)
print(f'\n全局汇总已保存: {os.path.join(EXP_DIR, "all_projects_metrics.csv")}')
