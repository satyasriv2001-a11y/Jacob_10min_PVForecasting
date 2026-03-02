"""
从 Harvard Dataverse (doi:10.7910/DVN/3VKAGM) 筛选8个新的正常场站
筛选标准：
  1. CF峰值小时 == 13
  2. clipping率 (CF>65%) < 15%
  3. 行数 >= 35000
  4. 2024年仍有正常发电（测试集不全为0）
  5. 不在已有13个场站列表中
"""
import os, json, requests, time
import pandas as pd
import numpy as np

# ── 已有场站（跳过）────────────────────────────────────────────────────────────
EXISTING = {
    'Project1033', 'Project1070', 'Project1077', 'Project1081', 'Project1084',
    'Project1137', 'Project171',  'Project172',  'Project186',  'Project212',
    'Project213',  'Project240',  'Project249',  'Project1140',
}

TARGET = 8   # 目标新场站数

# ── Step 1: 获取全部100个场站的 fileId ────────────────────────────────────────
print("正在获取 Harvard Dataverse 文件列表...")
url = ("https://dataverse.harvard.edu/api/datasets/:persistentId/"
       "versions/1.0/files?persistentId=doi:10.7910/DVN/3VKAGM&limit=200")
resp = requests.get(url, timeout=60)
data = resp.json()
all_files = [(item['label'].replace('.tab',''), item['dataFile']['id'])
             for item in data['data']]
print(f"共 {len(all_files)} 个文件")

# 过滤掉已有场站
candidates = [(name, fid) for name, fid in all_files if name not in EXISTING]
print(f"候选场站: {len(candidates)} 个（去除已有 {len(EXISTING)} 个）\n")

# ── 筛选函数 ──────────────────────────────────────────────────────────────────
def screen(tab_path):
    try:
        df = pd.read_csv(tab_path, sep='\t', low_memory=False,
                         usecols=['Hour (Eastern Time, Daylight-Adjusted)',
                                  'Capacity Factor', 'date'])
        df['CF']   = pd.to_numeric(df['Capacity Factor'], errors='coerce')
        df['Hour'] = pd.to_numeric(df['Hour (Eastern Time, Daylight-Adjusted)'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['CF','Hour','date'])

        if len(df) < 35000:
            return False, f"行数不足({len(df)})"

        peak_hour = df.groupby('Hour')['CF'].mean().idxmax()
        clip_rate = (df['CF'] > 65).mean()

        if peak_hour != 13.0:
            return False, f"CF峰值={peak_hour}h≠13"
        if clip_rate >= 0.15:
            return False, f"clipping={clip_rate*100:.1f}%≥15%"

        # 额外检查：2024年有正常发电
        df2024 = df[df['date'].dt.year == 2024]
        if len(df2024) == 0 or df2024['CF'].max() == 0:
            return False, "2024年无发电记录"

        # 测试集（最后15%）不能全为0
        n = len(df[df['date'].dt.year >= 2022])
        test_start = int(n * 0.85)
        df_2022 = df[df['date'].dt.year >= 2022].iloc[test_start:]
        if df_2022['CF'].max() == 0:
            return False, "测试集CF全为0（电站已停运）"

        return True, f"peak={peak_hour}h clip={clip_rate*100:.1f}% rows={len(df):,}"
    except Exception as e:
        return False, f"读取错误: {e}"

# ── Step 2: 逐个下载筛选 ───────────────────────────────────────────────────────
found = []
checked = 0

for name, fid in candidates:
    if len(found) >= TARGET:
        break

    checked += 1
    tab_path = f"/tmp/new_{fid}.tab"
    print(f"[{checked:3d}] {name} (fileId={fid}) ... ", end='', flush=True)

    # 下载
    try:
        dl_url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}"
        r = requests.get(dl_url, timeout=120)
        with open(tab_path, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        print(f"下载失败: {e}")
        continue

    # 筛选
    ok, msg = screen(tab_path)
    if ok:
        print(f"✅ {msg}")
        found.append({'name': name, 'fileId': fid, 'tab': tab_path})
    else:
        print(f"❌ {msg}")
        try: os.remove(tab_path)
        except: pass

    time.sleep(0.3)

# ── 结果 ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"筛选完成：共检查 {checked} 个场站，找到 {len(found)} 个合格")
print(f"{'='*55}")
for s in found:
    print(f"  {s['name']}  fileId={s['fileId']}  → {s['tab']}")

# 保存筛选结果
result_path = '/tmp/new_stations.json'
with open(result_path, 'w') as f:
    json.dump(found, f, indent=2)
print(f"\n结果已保存: {result_path}")
