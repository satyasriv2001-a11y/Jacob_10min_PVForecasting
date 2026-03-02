"""
将 Harvard Dataverse 小时级 .tab 文件插值为 10 分钟级 CSV
======================================================
用法：
    python interpolate_hourly_to_10min.py

输入：
    TARGETS 列表中指定的 (项目名, fileId) 对
    .tab 文件路径：/tmp/new_{fileId}.tab（由 find_8_more_stations.py 下载）

输出：
    data/{ProjectXXXX}.csv              ← 主数据目录
    experiments/{ProjectXXXX}/data/{ProjectXXXX}.csv  ← 实验数据目录

插值方法：
    - 数值列：pandas time-based 线性插值（method='time'）
    - 字符串列：前向填充（ffill）
    - Capacity Factor：clip 到 [0, 100]
    - 时间戳：精确到 10 分钟，去除 UTC 时区信息

验证：
    - 每天应有 144 行（00:00 ~ 23:50）
    - 总行数约 249,547（2020-01-01 ~ 2024-09-28）
"""

import os, sys, shutil
import numpy as np
import pandas as pd

BASE    = '/Users/terry/Library/CloudStorage/GoogleDrive-zl2268@cornell.edu/.shortcut-targets-by-id/1tYXd7zscd3BkVOz6B4y72IxqdsQEKgzW/SolarPrediction'
DATA_DIR = os.path.join(BASE, 'data')
EXP_DIR  = os.path.join(BASE, 'experiments')

# ── 在这里添加需要处理的场站 ──────────────────────────────────────────────────
TARGETS = [
    # ('ProjectXXXX', '文件ID'),
    # ('Project283',  '13245157'),  # 已处理，仅作示例
]


def interpolate_tab_to_10min(tab_path: str, proj: str) -> pd.DataFrame:
    """
    读取小时级 .tab 文件，插值为 10 分钟级 DataFrame。

    参数
    ----
    tab_path : str   .tab 文件路径
    proj     : str   项目名（用于日志）

    返回
    ----
    df_10min : pd.DataFrame  10 分钟间隔的完整数据，含 'date' 列
    """
    # ── 1. 读取原始小时数据 ────────────────────────────────────────────────
    df = pd.read_csv(tab_path, sep='\t', low_memory=False)
    print(f'  原始读取: {len(df):,} 行 × {len(df.columns)} 列')

    # ── 2. 解析日期，去除 UTC 时区，去重 ────────────────────────────────
    df['Datetime'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_convert(None)
    before = len(df)
    df = df.dropna(subset=['Datetime'])
    df = df.set_index('Datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    print(f'  有效行数: {len(df):,}（丢弃 {before - len(df)} 个无效/重复时间戳）')
    print(f'  时间范围: {df.index[0]} ~ {df.index[-1]}')

    # 验证原始间隔
    diffs = df.index.to_series().diff().dropna()
    median_min = diffs.median().total_seconds() / 60
    print(f'  原始时间间隔: {median_min:.0f} 分钟')
    if abs(median_min - 60) > 5:
        print(f'  ⚠️  警告：预期 60 分钟间隔，实际为 {median_min:.0f} 分钟')

    # ── 3. 分列处理 ───────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 数值列：先 asfreq 填入 NaN，再 time 插值
    df_num = df[num_cols].resample('10min').asfreq()
    df_num = df_num.interpolate(method='time')

    # 字符串列：前向填充
    df_str = df[str_cols].resample('10min').ffill()

    df_10min = pd.concat([df_num, df_str], axis=1)

    # ── 4. 后处理 ─────────────────────────────────────────────────────────
    # Capacity Factor 物理约束
    if 'Capacity Factor' in df_10min.columns:
        df_10min['Capacity Factor'] = df_10min['Capacity Factor'].clip(0, 100)

    # 还原 date 列（naive datetime 字符串）
    df_10min['date'] = df_10min.index.strftime('%Y-%m-%d %H:%M:%S')
    df_10min = df_10min.reset_index(drop=True)

    # ── 5. 验证 ───────────────────────────────────────────────────────────
    sample_day = '2022-06-15'
    n_day = len(df_10min[df_10min['date'].str.startswith(sample_day)])
    print(f'  插值后行数: {len(df_10min):,}（{sample_day} 当天 {n_day} 行，期望 144）')
    if 'Capacity Factor' in df_10min.columns:
        cf = df_10min['Capacity Factor']
        print(f'  Capacity Factor: min={cf.min():.3f}  max={cf.max():.3f}  mean={cf.mean():.3f}')

    return df_10min


def save_csv(df: pd.DataFrame, proj: str):
    """保存到 data/ 和 experiments/{proj}/data/ 两个位置。"""
    dst_main = os.path.join(DATA_DIR, f'{proj}.csv')
    dst_exp  = os.path.join(EXP_DIR, proj, 'data', f'{proj}.csv')

    os.makedirs(os.path.dirname(dst_exp), exist_ok=True)

    df.to_csv(dst_main, index=False)
    shutil.copy2(dst_main, dst_exp)

    size_mb = os.path.getsize(dst_main) / 1024 / 1024
    print(f'  已保存: {dst_main}  ({size_mb:.1f} MB)')
    print(f'  已同步: {dst_exp}')


# ── 主流程 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not TARGETS:
        print('TARGETS 列表为空，请先在脚本中填写要处理的场站。')
        sys.exit(0)

    summary = []
    for proj, fid in TARGETS:
        print(f'\n{"="*55}')
        print(f'  {proj}  (fileId={fid})')
        print(f'{"="*55}')

        tab_path = f'/tmp/new_{fid}.tab'
        if not os.path.exists(tab_path):
            print(f'  ❌ 文件不存在: {tab_path}')
            print(f'     请先运行 find_8_more_stations.py 下载，或手动下载到 /tmp/')
            summary.append({'project': proj, 'status': '文件缺失'})
            continue

        try:
            df_10min = interpolate_tab_to_10min(tab_path, proj)
            save_csv(df_10min, proj)
            summary.append({'project': proj, 'status': '完成', 'rows': len(df_10min)})
        except Exception as e:
            print(f'  ❌ 处理失败: {e}')
            summary.append({'project': proj, 'status': f'失败: {e}'})

    print(f'\n{"="*55}\n汇总\n{"="*55}')
    for s in summary:
        rows_str = f"  {s.get('rows', 0):,} 行" if 'rows' in s else ''
        print(f"  {s['project']:<16} {s['status']}{rows_str}")
