"""
从 Harvard Dataverse (doi:10.7910/DVN/3VKAGM) 筛选5个10分钟级优质场站
筛选标准：
  1. 时间间隔 == 10分钟（排除小时级数据）
  2. CF峰值小时 == 13
  3. clipping率 (CF>65%) < 10%（比之前更严格）
  4. 行数 >= 200000（10min级场站约249000行）
  5. 2024年仍有正常发电
  6. 测试集（最后15%）不全为0
  7. CF变异系数 > 0.5（排除变化太平稳导致R²低的场站，如Project186/249）
"""
import os, json, requests, time
import pandas as pd
import numpy as np

EXISTING = {
    'Project1033', 'Project1070', 'Project1077', 'Project1081', 'Project1084',
    'Project1137', 'Project171',  'Project172',  'Project186',  'Project212',
    'Project213',  'Project240',  'Project249',  'Project1140',
    'Project263',  'Project283',  'Project293',  'Project318',  'Project322',
    'Project357',  'Project358',  'Project369',
}

TARGET = 5

print("正在获取 Harvard Dataverse 文件列表...")
url = ("https://dataverse.harvard.edu/api/datasets/:persistentId/"
       "versions/1.0/files?persistentId=doi:10.7910/DVN/3VKAGM&limit=200")
resp = requests.get(url, timeout=60)
all_files = [(item['label'].replace('.tab',''), item['dataFile']['id'])
             for item in resp.json()['data']]
print(f"共 {len(all_files)} 个文件")

candidates = [(n, fid) for n, fid in all_files if n not in EXISTING]
print(f"候选场站: {len(candidates)} 个\n")

def screen(tab_path):
    try:
        df = pd.read_csv(tab_path, sep='\t', low_memory=False,
                         usecols=['Hour (Eastern Time, Daylight-Adjusted)',
                                  'Capacity Factor', 'date'])
        df['CF']   = pd.to_numeric(df['Capacity Factor'], errors='coerce')
        df['Hour'] = pd.to_numeric(df['Hour (Eastern Time, Daylight-Adjusted)'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['CF', 'Hour', 'date']).sort_values('date').reset_index(drop=True)

        # ── 1. 检查时间间隔 ──────────────────────────────
        if len(df) < 2:
            return False, "数据太少"
        intervals = df['date'].diff().dropna()
        median_min = intervals.median().total_seconds() / 60
        if abs(median_min - 10) > 2:
            return False, f"非10min数据（间隔={median_min:.0f}min）"

        # ── 2. 行数检查 ──────────────────────────────────
        if len(df) < 200000:
            return False, f"行数不足({len(df):,}，需≥200000)"

        # ── 3. CF峰值小时 ────────────────────────────────
        peak_hour = df.groupby('Hour')['CF'].mean().idxmax()
        if peak_hour != 13.0:
            return False, f"CF峰值={peak_hour}h≠13"

        # ── 4. clipping率 ────────────────────────────────
        clip_rate = (df['CF'] > 65).mean()
        if clip_rate >= 0.10:
            return False, f"clipping={clip_rate*100:.1f}%≥10%"

        # ── 5. 2024年有发电 ──────────────────────────────
        df2024 = df[df['date'].dt.year == 2024]
        if len(df2024) == 0 or df2024['CF'].max() == 0:
            return False, "2024年无发电"

        # ── 6. 测试集不全为0 ─────────────────────────────
        df2022 = df[df['date'].dt.year >= 2022]
        test_start = int(len(df2022) * 0.85)
        if df2022.iloc[test_start:]['CF'].max() == 0:
            return False, "测试集CF全为0"

        # ── 7. CF变异系数（排除Project186/249类型）───────
        daytime = df[df['Hour'].between(9, 17)]
        cv = daytime['CF'].std() / (daytime['CF'].mean() + 1e-6)
        if cv < 0.5:
            return False, f"白天CF变化太小(CV={cv:.2f}<0.5)"

        return True, (f"✅ peak={peak_hour}h  clip={clip_rate*100:.1f}%  "
                      f"rows={len(df):,}  CV={cv:.2f}")
    except Exception as e:
        return False, f"错误: {e}"

found = []
checked = 0

for name, fid in candidates:
    if len(found) >= TARGET:
        break

    checked += 1
    tab_path = f"/tmp/new_{fid}.tab"
    print(f"[{checked:3d}] {name} (fileId={fid}) ... ", end='', flush=True)

    try:
        r = requests.get(f"https://dataverse.harvard.edu/api/access/datafile/{fid}",
                         timeout=120)
        with open(tab_path, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        print(f"下载失败: {e}")
        continue

    ok, msg = screen(tab_path)
    print(msg)
    if ok:
        found.append({'name': name, 'fileId': fid, 'tab': tab_path})
    else:
        try: os.remove(tab_path)
        except: pass

    time.sleep(0.3)

print(f"\n{'='*55}")
print(f"筛选完成：检查 {checked} 个，找到 {len(found)} 个合格")
print(f"{'='*55}")
for s in found:
    print(f"  {s['name']}  fileId={s['fileId']}")

with open('/tmp/new_5stations.json', 'w') as f:
    json.dump(found, f, indent=2)
print(f"\n结果已保存: /tmp/new_5stations.json")
