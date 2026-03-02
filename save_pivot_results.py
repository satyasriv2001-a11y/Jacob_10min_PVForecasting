"""
Save test results in pivot format:
  rows    = every 10-min datetime in the test period
  columns = datetime, ground_tru, forecast_+0, forecast_+1, ..., forecast_+143

  forecast_+k at datetime T  =  predicted value from the forecast window that
                                 started k steps (k*10 min) before T
                                 (i.e., sample whose forecast_start = T - k*10min,
                                  using that sample's predicted_k column)

Total rows ≈ n_test_samples + 143  (= 21,577 + 143 = 21,720)
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
n_samples  = len(y_test)

def inverse(arr):
    return scaler_target.inverse_transform(
        arr.reshape(-1, 1)
    ).reshape(-1, FUTURE_STEPS)

y_true = inverse(y_test)

# ── load linear model + predict ───────────────────────────────────────────────
print("Loading Linear Regression model...")
lr       = joblib.load(os.path.join(BASE, 'experiments', 'lza1033', 'models', 'linear_model.pkl'))['model']
lr_preds = np.clip(inverse(lr.predict(X_test)), 0, 100)
print("  done")

# ── verify stride = 10 min ────────────────────────────────────────────────────
end0 = pd.Timestamp(dates_test[0])
end1 = pd.Timestamp(dates_test[1])
stride_min = (end1 - end0).total_seconds() / 60
print(f"  stride between consecutive samples: {stride_min:.0f} min")
assert stride_min == STEP_MIN, f"Expected stride={STEP_MIN}, got {stride_min}"

# ── build datetime index ──────────────────────────────────────────────────────
# sample i has forecast_start = dates_test[i] - 143*10min
# sample 0 start = earliest datetime
start0   = end0 - pd.Timedelta(minutes=STEP_MIN * (FUTURE_STEPS - 1))
n_dts    = n_samples + FUTURE_STEPS - 1   # 21,577 + 143 = 21,720
dt_index = pd.date_range(start=start0, periods=n_dts, freq=f'{STEP_MIN}min')
print(f"  datetime range: {dt_index[0]} ~ {dt_index[-1]}")
print(f"  total unique datetimes: {n_dts:,}")

# ── build actual array (vectorized) ──────────────────────────────────────────
# For sample i, step k → datetime index t = i + k
# actual_array[t] = y_true[i, k]  (same value regardless of which i,k pair)
actual_array = np.full(n_dts, np.nan)
for k in range(FUTURE_STEPS):
    actual_array[k : k + n_samples] = y_true[:, k]

# ── build forecast matrix (n_dts × FUTURE_STEPS) ─────────────────────────────
# forecast_matrix[t, k] = prediction for datetime t, from window started k steps before t
#                        = lr_preds[i, k]  where i = t - k
print("Building forecast matrix...")
forecast_matrix = np.full((n_dts, FUTURE_STEPS), np.nan)
for k in range(FUTURE_STEPS):
    forecast_matrix[k : k + n_samples, k] = lr_preds[:, k]

# ── assemble DataFrame ────────────────────────────────────────────────────────
print("Assembling DataFrame...")
data = {'datetime': dt_index, 'ground_tru': np.round(actual_array, 4)}
for k in range(FUTURE_STEPS):
    data[f'forecast_+{k}'] = np.round(forecast_matrix[:, k], 4)

df = pd.DataFrame(data)
print(f"  shape: {df.shape}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'test_results_pivot.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"\nFirst 5 rows (first 6 columns):")
print(df.iloc[:5, :6].to_string(index=False))
