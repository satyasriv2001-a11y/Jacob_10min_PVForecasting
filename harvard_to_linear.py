"""
Harvard Dataverse 场站 → 线性回归 端到端管道
=============================================
用法：
    # 方式1：直接指定场站（推荐）
    在 TARGETS 填入 [('ProjectXXX', '文件ID'), ...]
    python harvard_to_linear.py

    # 方式2：自动筛选 N 个场站
    TARGETS = []，设置 AUTO_SCREEN_N > 0
    python harvard_to_linear.py

步骤：
    1. [可选] 从 Harvard Dataverse 自动筛选场站
    2. 下载 .tab 文件到 /tmp/
    3. 小时级 → 10 分钟级插值
    4. 训练线性回归（24h lookback → 144步 × 10min ahead）
    5. 评估：MAE, RMSE, R²（全天 + 白天）
    6. 画图：随机 10 天 actual vs predicted
    7. 更新 experiments/all_projects_metrics.csv

筛选标准（AUTO_SCREEN 模式）：
    - CF 峰值小时 == 13（排除时间戳偏移场站）
    - clipping 率（CF>65%）< 10%
    - 行数 >= 35000
    - 2024 年仍有发电
    - 测试集最后 15% 不全为 0
    - 白天 CF 变异系数 CV > 0.5
"""

import os, sys, time, gc, shutil, warnings
import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

BASE     = '/Users/terry/Library/CloudStorage/GoogleDrive-zl2268@cornell.edu/.shortcut-targets-by-id/1tYXd7zscd3BkVOz6B4y72IxqdsQEKgzW/SolarPrediction'
DATA_DIR = os.path.join(BASE, 'data')
EXP_DIR  = os.path.join(BASE, 'experiments')
sys.path.insert(0, BASE)

from data.data_utils import preprocess_features, create_sliding_windows

# ══════════════════════════════════════════════════════════════════════════════
#  配置区：每次新增场站只需修改这里
# ══════════════════════════════════════════════════════════════════════════════

# 直接指定 (项目名, fileId)；留空则启动自动筛选
TARGETS = [
    # ('ProjectXXXX', '文件ID'),
]

# 自动筛选：TARGETS 为空时，从 Harvard 筛选 N 个新场站
AUTO_SCREEN_N = 5

# 已存在的场站（自动筛选时跳过）
EXISTING_PROJECTS = {
    'Project1033', 'Project1070', 'Project1077', 'Project1081', 'Project1084',
    'Project1137', 'Project171',  'Project172',  'Project186',  'Project212',
    'Project213',  'Project240',  'Project249',  'Project1140',
    'Project263',  'Project283',  'Project293',  'Project318',  'Project322',
    'Project357',  'Project358',  'Project369',
}

# 模型配置（一般不需要修改）
MODEL_CONFIG = {
    'model': 'Linear', 'model_complexity': 'high',
    'use_pv': True, 'use_hist_weather': False, 'use_forecast': True,
    'use_ideal_nwp': False, 'use_time_encoding': True,
    'weather_category': 'all_weather',
    'past_hours': 24, 'future_hours': 24,
    'start_date': '2022-01-01', 'end_date': '2024-09-28',
}
FUTURE_STEPS = 144
STEP_MIN     = 10


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1：自动筛选（可选）
# ══════════════════════════════════════════════════════════════════════════════

def screen_station(tab_path):
    """返回 (ok: bool, msg: str)"""
    try:
        df = pd.read_csv(tab_path, sep='\t', low_memory=False,
                         usecols=['Hour (Eastern Time, Daylight-Adjusted)',
                                  'Capacity Factor', 'date'])
        df['CF']   = pd.to_numeric(df['Capacity Factor'], errors='coerce')
        df['Hour'] = pd.to_numeric(df['Hour (Eastern Time, Daylight-Adjusted)'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_convert(None)
        df = df.dropna(subset=['CF', 'Hour', 'date']).sort_values('date').reset_index(drop=True)

        if len(df) < 2:
            return False, "数据太少"

        # 行数
        if len(df) < 35000:
            return False, f"行数不足({len(df):,}，需≥35000)"

        # CF 峰值小时
        peak_hour = df.groupby('Hour')['CF'].mean().idxmax()
        if peak_hour != 13.0:
            return False, f"CF峰值={peak_hour}h≠13"

        # clipping 率
        clip_rate = (df['CF'] > 65).mean()
        if clip_rate >= 0.10:
            return False, f"clipping={clip_rate*100:.1f}%≥10%"

        # 2024 年有发电
        df2024 = df[df['date'].dt.year == 2024]
        if len(df2024) == 0 or df2024['CF'].max() == 0:
            return False, "2024年无发电"

        # 测试集不全为 0
        df2022 = df[df['date'].dt.year >= 2022]
        test_start = int(len(df2022) * 0.85)
        if df2022.iloc[test_start:]['CF'].max() == 0:
            return False, "测试集CF全为0"

        # CF 变异系数
        daytime = df[df['Hour'].between(9, 17)]
        cv = daytime['CF'].std() / (daytime['CF'].mean() + 1e-6)
        if cv < 0.5:
            return False, f"白天CF变化太小(CV={cv:.2f}<0.5)"

        return True, (f"OK  peak={peak_hour}h  clip={clip_rate*100:.1f}%  "
                      f"rows={len(df):,}  CV={cv:.2f}")
    except Exception as e:
        return False, f"错误: {e}"


def auto_screen(n_target):
    """从 Harvard Dataverse 自动筛选 n_target 个新场站，返回 [(name, fid), ...]"""
    print(f"\n{'='*60}")
    print(f"  自动筛选模式：从 Harvard Dataverse 寻找 {n_target} 个新场站")
    print(f"{'='*60}")

    url = ("https://dataverse.harvard.edu/api/datasets/:persistentId/"
           "versions/1.0/files?persistentId=doi:10.7910/DVN/3VKAGM&limit=200")
    resp = requests.get(url, timeout=60)
    all_files = [(item['label'].replace('.tab', ''), item['dataFile']['id'])
                 for item in resp.json()['data']]
    print(f"  共 {len(all_files)} 个文件")

    candidates = [(n, fid) for n, fid in all_files if n not in EXISTING_PROJECTS]
    print(f"  候选（排除已有）：{len(candidates)} 个\n")

    found = []
    for idx, (name, fid) in enumerate(candidates, 1):
        if len(found) >= n_target:
            break

        tab_path = f'/tmp/new_{fid}.tab'
        print(f"  [{idx:3d}] {name} (fileId={fid}) ... ", end='', flush=True)

        try:
            r = requests.get(f"https://dataverse.harvard.edu/api/access/datafile/{fid}",
                             timeout=120)
            with open(tab_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"下载失败: {e}")
            continue

        ok, msg = screen_station(tab_path)
        print(msg)
        if ok:
            found.append((name, str(fid)))
        else:
            try:
                os.remove(tab_path)
            except Exception:
                pass

        time.sleep(0.3)

    print(f"\n  筛选完成：找到 {len(found)} 个合格场站")
    for name, fid in found:
        print(f"    {name}  fileId={fid}")
    return found


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2：下载 .tab
# ══════════════════════════════════════════════════════════════════════════════

def download_tab(fid):
    """下载 .tab 到 /tmp/new_{fid}.tab，已存在则跳过。返回路径。"""
    tab_path = f'/tmp/new_{fid}.tab'
    if os.path.exists(tab_path):
        print(f"  .tab 已存在，跳过下载：{tab_path}")
        return tab_path

    print(f"  下载 fileId={fid} ...", end='', flush=True)
    r = requests.get(f"https://dataverse.harvard.edu/api/access/datafile/{fid}",
                     timeout=180)
    with open(tab_path, 'wb') as f:
        f.write(r.content)
    size_mb = os.path.getsize(tab_path) / 1024 / 1024
    print(f" {size_mb:.1f} MB")
    return tab_path


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3：插值（小时 → 10 分钟）
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_to_10min(tab_path, proj):
    """读取小时级 .tab，插值为 10 分钟级，返回 DataFrame。"""
    df = pd.read_csv(tab_path, sep='\t', low_memory=False)
    print(f"  原始: {len(df):,} 行 × {len(df.columns)} 列")

    # 解析时间，去除 UTC 时区
    df['Datetime'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_convert(None)
    df = df.dropna(subset=['Datetime'])
    df = df.set_index('Datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    print(f"  有效: {len(df):,} 行  ({df.index[0]} ~ {df.index[-1]})")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    df_num = df[num_cols].resample('10min').asfreq().interpolate(method='time')
    df_str = df[str_cols].resample('10min').ffill()

    df_10 = pd.concat([df_num, df_str], axis=1)

    if 'Capacity Factor' in df_10.columns:
        df_10['Capacity Factor'] = df_10['Capacity Factor'].clip(0, 100)

    df_10['date'] = df_10.index.strftime('%Y-%m-%d %H:%M:%S')
    df_10 = df_10.reset_index(drop=True)
    print(f"  插值后: {len(df_10):,} 行")
    return df_10


def save_csv(df, proj):
    """保存到 data/ 和 experiments/{proj}/data/。"""
    dst_main = os.path.join(DATA_DIR, f'{proj}.csv')
    dst_exp  = os.path.join(EXP_DIR, proj, 'data', f'{proj}.csv')
    os.makedirs(os.path.dirname(dst_exp), exist_ok=True)
    df.to_csv(dst_main, index=False)
    shutil.copy2(dst_main, dst_exp)
    size_mb = os.path.getsize(dst_main) / 1024 / 1024
    print(f"  保存: {dst_main}  ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 4 & 5：训练 + 评估
# ══════════════════════════════════════════════════════════════════════════════

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


def train_linear(proj):
    """训练线性回归，保存模型和结果，返回 metrics dict。"""
    csv_path = os.path.join(DATA_DIR, f'{proj}.csv')

    # 目录
    proj_dir   = os.path.join(EXP_DIR, proj)
    model_dir  = os.path.join(proj_dir, 'models')
    result_dir = os.path.join(proj_dir, 'results')
    chart_dir  = os.path.join(result_dir, 'charts')
    for d in [model_dir, result_dir, chart_dir]:
        os.makedirs(d, exist_ok=True)

    # 加载
    df_raw = pd.read_csv(csv_path)
    df_raw['Datetime'] = pd.to_datetime(df_raw['date'], utc=True).dt.tz_convert(None)
    df_raw = df_raw[df_raw['Datetime'].dt.year >= 2022].reset_index(drop=True)
    night_anom = (df_raw['Datetime'].dt.hour < 5) & \
                 (df_raw['global_tilted_irradiance_pred'].fillna(0) > 5)
    df_raw = df_raw[~night_anom].reset_index(drop=True)
    print(f"  数据行数: {len(df_raw):,}")

    # 特征工程 + 滑动窗口
    df_clean, hist_feats, fcst_feats, _, _, scaler_target, no_hist_power = \
        preprocess_features(df_raw.copy(), MODEL_CONFIG)
    del df_raw; gc.collect()

    Xh, Xf, y, hours, dates = create_sliding_windows(
        df_clean, MODEL_CONFIG['past_hours'], MODEL_CONFIG['future_hours'],
        hist_feats, fcst_feats, no_hist_power
    )
    del df_clean; gc.collect()

    Xh = Xh.astype(np.float32)
    Xf = Xf.astype(np.float32) if Xf is not None else None
    y  = y.astype(np.float32)
    print(f"  Xh={Xh.shape}  Xf={Xf.shape if Xf is not None else None}  y={y.shape}")

    # 划分
    total     = len(Xh)
    train_end = int(total * 0.85)
    X_train = flatten(Xh[:train_end], Xf[:train_end] if Xf is not None else None)
    X_test  = flatten(Xh[train_end:], Xf[train_end:] if Xf is not None else None)
    y_train = y[:train_end].reshape(train_end, -1)
    y_test  = y[train_end:]
    dates_test = dates[train_end:]
    print(f"  训练={train_end:,}  测试={total-train_end:,}")

    # 训练
    t0 = time.time()
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    train_sec = time.time() - t0
    print(f"  训练完成  {train_sec:.1f}s")

    # 预测 + 反归一化
    preds_raw = model.predict(X_test)
    y_true = scaler_target.inverse_transform(
        y_test.reshape(-1, 1)).reshape(-1, FUTURE_STEPS)
    y_pred = np.clip(
        scaler_target.inverse_transform(
            preds_raw.reshape(-1, 1)).reshape(-1, FUTURE_STEPS), 0, 100)

    m_all = calc_metrics(y_true, y_pred)
    m_day = calc_metrics(y_true[:, 60:108], y_pred[:, 60:108])
    print(f"  [全天] MAE={m_all['mae']:.4f}  RMSE={m_all['rmse']:.4f}  R²={m_all['r2']:.4f}")
    print(f"  [白天] MAE={m_day['mae']:.4f}  RMSE={m_day['rmse']:.4f}  R²={m_day['r2']:.4f}")

    # 保存模型
    joblib.dump({'model': model, 'scaler_target': scaler_target},
                os.path.join(model_dir, 'linear_model.pkl'))

    # 保存预测结果
    offset = pd.Timedelta(minutes=STEP_MIN * (FUTURE_STEPS - 1))
    forecast_starts = [
        (pd.Timestamp(dates_test[i]) - offset).strftime('%Y-%m-%d %H:%M')
        for i in range(len(y_pred))
    ]
    pred_df = pd.DataFrame({
        'forecast_start': forecast_starts,
        **{f'predicted_{k}': np.round(y_pred[:, k], 4) for k in range(FUTURE_STEPS)}
    })
    pred_df.to_csv(os.path.join(result_dir, 'predictions_144steps.csv'), index=False)

    # 保存场站指标
    pd.DataFrame([{
        'project': proj, 'n_train': train_end, 'n_test': total - train_end,
        'train_sec': round(train_sec, 1),
        'mae_all': round(m_all['mae'], 4), 'rmse_all': round(m_all['rmse'], 4),
        'r2_all': round(m_all['r2'], 4),
        'mae_day': round(m_day['mae'], 4), 'rmse_day': round(m_day['rmse'], 4),
        'r2_day': round(m_day['r2'], 4),
    }]).to_csv(os.path.join(result_dir, 'metrics.csv'), index=False)

    del Xh, Xf, y, X_train, X_test; gc.collect()
    return dict(proj=proj, n_test=total-train_end, train_sec=round(train_sec, 1),
                **{k: round(v, 4) for k, v in m_all.items()},
                mae_day=round(m_day['mae'], 4), r2_day=round(m_day['r2'], 4),
                y_true=y_true, y_pred=y_pred, forecast_starts=forecast_starts)


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 6：画图（随机 10 天）
# ══════════════════════════════════════════════════════════════════════════════

def plot_10days(proj, y_true, y_pred, forecast_starts):
    """随机抽取 10 天，画 actual vs predicted。"""
    # 只保留从零点（00:00）开始的预测
    midnight = [i for i, s in enumerate(forecast_starts) if s.endswith('00:00')]
    if len(midnight) == 0:
        print("  ⚠️  无零点预测，跳过画图")
        return

    rng = np.random.RandomState(42)
    chosen = sorted(rng.choice(midnight, size=min(10, len(midnight)), replace=False))

    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 7), sharey=True)
    t_axis = np.arange(FUTURE_STEPS) * STEP_MIN / 60  # hours

    for ax_idx, idx in enumerate(chosen):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        ax.plot(t_axis, y_true[idx], label='Actual',    color='steelblue', linewidth=1.5)
        ax.plot(t_axis, y_pred[idx], label='Predicted', color='tomato',    linewidth=1.5, linestyle='--')
        ax.set_title(forecast_starts[idx][:10], fontsize=9)
        ax.set_xlabel('Hour')
        ax.set_ylim(0, 105)
        if ax_idx % ncols == 0:
            ax.set_ylabel('CF (%)')
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f'{proj} — Random 10 Days (Linear Regression)', fontsize=13)
    plt.tight_layout()

    chart_dir = os.path.join(EXP_DIR, proj, 'results', 'charts')
    save_path = os.path.join(chart_dir, 'linear_10days.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  图表: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 7：更新全局指标
# ══════════════════════════════════════════════════════════════════════════════

def update_global_metrics(new_rows):
    """将新场站指标追加/更新到 all_projects_metrics.csv。"""
    global_path = os.path.join(EXP_DIR, 'all_projects_metrics.csv')

    keep_cols = ['project', 'n_train', 'n_test', 'train_sec',
                 'mae_all', 'rmse_all', 'r2_all', 'mae_day', 'rmse_day', 'r2_day']

    if os.path.exists(global_path):
        df_exist = pd.read_csv(global_path)
        # 删除同名旧记录
        new_projs = {r['proj'] for r in new_rows}
        df_exist = df_exist[~df_exist['project'].isin(new_projs)]
    else:
        df_exist = pd.DataFrame(columns=keep_cols)

    new_df = pd.DataFrame([
        {c: r.get(c if c != 'project' else 'proj', r.get(c, None)) for c in keep_cols}
        for r in new_rows
    ])
    new_df.rename(columns={'proj': 'project'}, inplace=True, errors='ignore')

    df_all = pd.concat([df_exist, new_df], ignore_index=True)
    df_all = df_all.sort_values('r2_all', ascending=False)
    df_all.to_csv(global_path, index=False)
    print(f"\n全局指标已更新: {global_path}")
    print(df_all[['project', 'n_test', 'mae_all', 'r2_all', 'r2_day']].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    targets = list(TARGETS)

    # Stage 1：自动筛选
    if not targets and AUTO_SCREEN_N > 0:
        targets = auto_screen(AUTO_SCREEN_N)
        if not targets:
            print("未找到合格场站，退出。")
            sys.exit(0)
    elif not targets:
        print("TARGETS 为空且 AUTO_SCREEN_N=0，请配置后重新运行。")
        sys.exit(0)

    summary = []
    t_total = time.time()

    for proj, fid in targets:
        print(f'\n{"="*60}')
        print(f'  {proj}  (fileId={fid})')
        print(f'{"="*60}')
        t_proj = time.time()

        try:
            # Stage 2：下载
            tab_path = download_tab(fid)

            # Stage 3：插值
            csv_path = os.path.join(DATA_DIR, f'{proj}.csv')
            if not os.path.exists(csv_path):
                df_10 = interpolate_to_10min(tab_path, proj)
                save_csv(df_10, proj)
            else:
                print(f"  CSV 已存在，跳过插值：{csv_path}")

            # Stage 4 & 5：训练 + 评估
            result = train_linear(proj)

            # Stage 6：画图
            plot_10days(proj, result.pop('y_true'), result.pop('y_pred'),
                        result.pop('forecast_starts'))

            result['status'] = 'OK'
            print(f"  完成，耗时 {time.time()-t_proj:.1f}s")

        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {'proj': proj, 'status': f'FAILED: {e}'}

        summary.append(result)

    # Stage 7：更新全局指标
    ok_rows = [r for r in summary if r.get('status') == 'OK']
    if ok_rows:
        update_global_metrics(ok_rows)

    print(f'\n{"="*60}')
    print(f'全部完成  总耗时 {time.time()-t_total:.1f}s')
    print(f'{"="*60}')
    for r in summary:
        status = r.get('status', '?')
        if status == 'OK':
            print(f"  {r['proj']:<16} R²={r['r2']:.4f}  MAE={r['mae']:.4f}")
        else:
            print(f"  {r.get('proj','?'):<16} {status}")
