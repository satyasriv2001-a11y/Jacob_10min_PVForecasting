"""
Plot actual vs predicted for 10 evenly-spaced days from test_results_long.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE    = '/Users/terry/Library/CloudStorage/GoogleDrive-zl2268@cornell.edu/.shortcut-targets-by-id/1tYXd7zscd3BkVOz6B4y72IxqdsQEKgzW/SolarPrediction'
OUT_DIR = os.path.join(BASE, 'experiments', 'lza1033', 'results', 'charts')
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, 'experiments', 'lza1033', 'results', 'test_results_long.csv'))
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()

# ── pick 10 evenly-spaced days ────────────────────────────────────────────────
all_dates = df.index.normalize().unique()
indices   = np.linspace(0, len(all_dates) - 1, 10, dtype=int)
selected_days = all_dates[indices]

print("Selected days:")
for d in selected_days:
    print(f"  {d.date()}")

# ── plot 2×5 grid ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
axes = axes.flatten()

for ax, day in zip(axes, selected_days):
    day_df = df[df.index.normalize() == day]

    ax.plot(day_df.index, day_df['ground_truth'], color='steelblue',
            lw=1.5, label='Actual')
    ax.plot(day_df.index, day_df['prediction'],  color='tomato',
            lw=1.5, linestyle='--', label='Predicted')

    ax.set_title(day.strftime('%Y-%m-%d'), fontsize=10, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel('CF (%)', fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

axes[0].legend(fontsize=8, loc='upper left')

fig.suptitle('Project1033 — Linear Regression: Actual vs Predicted (10 Days)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, 'linear_10days.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {out_path}")
