"""
Save linear regression test results to CSV (wide format)
21,577 rows × (2 + 144 + 144) columns
Each row = one forecast window (sliding 10-min step)
Columns:
  forecast_start  — start datetime of the 24h window
  forecast_end    — end datetime of the 24h window
  actual_0 .. actual_143    — 144 actual CF values (%)
  predicted_0 .. predicted_143 — 144 predicted CF values (%)
"""
import os
import numpy as np
import pandas as pd
import joblib

BASE    = '/Users/terry/Library/CloudStorage/GoogleDrive-zl2268@cornell.edu/.shortcut-targets-by-id/1tYXd7zscd3BkVOz6B4y72IxqdsQEKgzW/SolarPrediction'
OUT_DIR = os.path.join(BASE, 'experiments', 'lza1033', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

STEP_MIN = 10

# ── load cache ────────────────────────────────────────────────────────────────
print("Loading project1033_cache.pkl...")
cache         = joblib.load(os.path.join(BASE, 'project1033_cache.pkl'))
Xh            = cache['Xh']
Xf            = cache['Xf']
y             = cache['y']
dates         = cache['dates']
scaler_target = cache['scaler_target']
train_end     = cache['train_end']
FUTURE_STEPS  = y.shape[1]
print(f"  total={len(Xh):,}  test={len(Xh)-train_end:,}  FUTURE_STEPS={FUTURE_STEPS}")

# ── test split ────────────────────────────────────────────────────────────────
def flatten(Xh, Xf):
    return np.concatenate([Xh.reshape(Xh.shape[0], -1),
                           Xf.reshape(Xf.shape[0], -1)], axis=1)

X_test     = flatten(Xh[train_end:], Xf[train_end:])
y_test     = y[train_end:]
dates_test = dates[train_end:]

def inverse(arr):
    return scaler_target.inverse_transform(
        arr.reshape(-1, 1)
    ).reshape(-1, FUTURE_STEPS)

y_true   = inverse(y_test)

# ── load linear model + predict ───────────────────────────────────────────────
print("Loading Linear Regression model...")
lr       = joblib.load(os.path.join(BASE, 'experiments', 'lza1033', 'models', 'linear_model.pkl'))['model']
lr_preds = np.clip(inverse(lr.predict(X_test)), 0, 100)
print("  done")

# ── build wide-format DataFrame ───────────────────────────────────────────────
print("Building wide-format DataFrame...")
n_samples = len(y_true)
offset_td = pd.Timedelta(minutes=STEP_MIN * (FUTURE_STEPS - 1))

forecast_starts = []
forecast_ends   = []
for i in range(n_samples):
    end_dt   = pd.Timestamp(dates_test[i])
    start_dt = end_dt - offset_td
    forecast_starts.append(start_dt.strftime('%Y-%m-%d %H:%M'))
    forecast_ends.append(end_dt.strftime('%Y-%m-%d %H:%M'))

actual_cols    = {f'actual_{s}':    np.round(y_true[:, s],    4) for s in range(FUTURE_STEPS)}
predicted_cols = {f'predicted_{s}': np.round(lr_preds[:, s],  4) for s in range(FUTURE_STEPS)}

df = pd.DataFrame({'forecast_start': forecast_starts,
                   'forecast_end':   forecast_ends,
                   **actual_cols,
                   **predicted_cols})

print(f"  rows={len(df):,}  cols={len(df.columns)}")

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'test_results.csv')
df.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
print(f"  rows={len(df):,}  cols={len(df.columns)}")
print(f"  date range: {df['forecast_start'].iloc[0]} ~ {df['forecast_end'].iloc[-1]}")
print(f"\nFirst row preview (first 6 columns):")
print(df.iloc[0, :6].to_string())
