"""
Save test results in long format (4 columns):
  date        — specific 10-min datetime (forecast_start of each window)
  projectID   — "Project1033"
  ground_truth — actual CF at step 0 of that window
  prediction   — linear regression predicted CF at step 0 of that window

21,577 rows (one per test sample)
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

y_true = inverse(y_test)

# ── load model + predict ──────────────────────────────────────────────────────
print("Loading Linear Regression model...")
lr       = joblib.load(os.path.join(BASE, 'experiments', 'lza1033', 'models', 'linear_model.pkl'))['model']
lr_preds = np.clip(inverse(lr.predict(X_test)), 0, 100)

# ── build forecast_start datetimes ───────────────────────────────────────────
offset   = pd.Timedelta(minutes=STEP_MIN * (FUTURE_STEPS - 1))
forecast_starts = [
    (pd.Timestamp(dates_test[i]) - offset).strftime('%Y-%m-%d %H:%M')
    for i in range(len(y_true))
]

# ── assemble 4-column DataFrame ───────────────────────────────────────────────
df = pd.DataFrame({
    'date':         forecast_starts,
    'projectID':    'Project1033',
    'ground_truth': np.round(y_true[:, 0], 4),
    'prediction':   np.round(lr_preds[:, 0], 4),
})

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'test_results_long.csv')
df.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
print(f"  rows={len(df):,}  cols={list(df.columns)}")
print(f"  date range: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
print(f"\nFirst 5 rows:")
print(df.head().to_string(index=False))
