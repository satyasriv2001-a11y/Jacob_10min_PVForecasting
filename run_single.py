#!/usr/bin/env python3
"""
Train all deep-learning models for ONE fixed setup:
    <model>_low_PV_24h_TE

Models included:
  - LSTM
  - GRU
  - Transformer
  - TCN
  - MultiChannelLSTM

Each model:
  - trains to convergence (train_dl_model handles early stopping)
  - saves best weights to: saved_models/<model>.pt
  - performs inference on the SAME 10 samples
  - computes MAPE
  - generates a plot (10 subplots OR single combined figure)

Plots match the user's TCN styling, X-axis length = 144 (24 hours @ 10-min).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

# ==========================================
#  Ensure working directory consistency
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model
from models.rnn_models import LSTM, GRU, MultiChannelLSTM
from models.transformer import Transformer
from models.tcn import TCNModel


# =============================================================
#  FIXED CONFIG (low, PV, TE, 24h forecast)
# =============================================================
def build_fixed_config():
    cfg = {
        "model_complexity": "low",
        "use_pv": True,
        "use_hist_weather": False,
        "use_forecast": False,
        "use_ideal_nwp": False,
        "use_time_encoding": True,
        "past_hours": 24,
        "future_hours": 24,
        "weather_category": "all_weather",
        "start_date": "2022-01-01",
        "end_date": "2024-09-28",
        "save_options": {},
        "train_params": {
            "epochs": 15,
            "batch_size": 64,
            "learning_rate": 0.001,
            "patience": 5,
            "min_delta": 1e-4
        },
        "model_params": {
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.1,
            "d_model": 32,
            "num_heads": 2,
            "tcn_channels": [16, 32],
            "kernel_size": 3,
            "te_dim": 8
        }
    }
    return cfg


# =============================================================
#  Load & preprocess dataset ONCE
# =============================================================
def load_dataset():
    df_raw = None
    path = os.path.join(script_dir, "data", "Project1140.csv")
    df_raw = __import__("pandas").read_csv(path)
    df_raw["Datetime"] = __import__("pandas").to_datetime(df_raw["date"])
    return df_raw


# =============================================================
#  Build + split windows (done ONCE)
# =============================================================
def build_dataset_bank(df_raw, cfg):
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = \
        preprocess_features(df_raw.copy(), cfg)

    Xh, Xf, y, hours, dates = create_sliding_windows(
        df_clean,
        cfg["past_hours"],
        cfg["future_hours"],
        hist_feats,
        fcst_feats,
        no_hist_power
    )

    total = len(Xh)
    tr = int(total * 0.8)
    va = int(total * 0.9)

    train = (
        Xh[:tr],
        None if Xf is None else Xf[:tr],
        y[:tr],
        np.array(hours[:tr]),
        []
    )
    val = (
        Xh[tr:va],
        None if Xf is None else Xf[tr:va],
        y[tr:va],
        np.array(hours[tr:va]),
        []
    )
    test = (
        Xh[va:],
        None if Xf is None else Xf[va:],
        y[va:],
        np.array(hours[va:]),
        dates[va:]
    )

    return train, val, test, (scaler_hist, scaler_fcst, scaler_target)


# =============================================================
#  Perform inference for 10 samples
# =============================================================
def collect_10_inference(model, device, Xh_te, Xf_te, y_te):
    model.eval()
    Xh_s = Xh_te[:10]
    Xf_s = None if Xf_te is None else Xf_te[:10]

    y_true = y_te[:10]

    preds = []
    with __import__("torch").no_grad():
        for i in range(10):
            xh = __import__("torch").tensor(Xh_s[i:i+1], dtype=__import__("torch").float32).to(device)
            if Xf_s is not None:
                xf = __import__("torch").tensor(Xf_s[i:i+1], dtype=__import__("torch").float32).to(device)
                p = model(xh, xf)
            else:
                p = model(xh)
            preds.append(p.cpu().numpy().squeeze())

    return np.array(preds), y_true


# =============================================================
#  Plot results (TCN-style)
# =============================================================
def plot_inference(model_name, preds, y_true, out_dir):
    future_steps = y_true.shape[1]
    t = np.arange(future_steps)

    # MAPE over the 10 samples
    mape = np.mean(np.abs((y_true - preds) / np.maximum(y_true, 1e-6))) * 100

    fig, axes = plt.subplots(10, 1, figsize=(11, 18), sharex=True)

    for i in range(10):
        ax = axes[i]
        ax.plot(t, y_true[i], color="black", linewidth=1.4, label="Ground Truth")
        ax.plot(t, preds[i], linestyle="--", color="green", linewidth=1.2,
                label=f"{model_name} (MAPE={mape:.2f}%)")

        ax.set_ylim(0, max(y_true[i].max(), preds[i].max()) * 1.1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name}: 10 Inference Samples", fontsize=16)
    plt.xlabel("Time (10-minute steps)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    fig_path = os.path.join(out_dir, f"{model_name}_inference.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path, mape


# =============================================================
#  MAIN SCRIPT
# =============================================================
def main():
    cfg = build_fixed_config()

    # models to train
    model_list = ["LSTM", "GRU", "Transformer", "TCN", "MultiChannelLSTM"]

    save_dir = os.path.join(script_dir, "one_setup_results")
    os.makedirs(save_dir, exist_ok=True)

    print("\n====================================================")
    print("  TRAINING ALL MODELS: low_PV_24h_TE")
    print("====================================================\n")

    df_raw = load_dataset()
    train_data, val_data, test_data, scalers = build_dataset_bank(df_raw, cfg)

    Xh_te, Xf_te, y_te, hrs_te, dates_te = test_data

    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    print(f"[DEVICE] Using {device}")

    summary = []

    for model_name in model_list:
        print("\n----------------------------------------------------")
        print(f"TRAINING MODEL: {model_name}")
        print("----------------------------------------------------")

        cfg_local = cfg.copy()
        cfg_local["model"] = model_name

        # Train
        model, metrics = train_dl_model(cfg_local, train_data, val_data, test_data, scalers)

        # Save best model
        out_path = os.path.join(save_dir, f"{model_name}.pt")
        __import__("torch").save(model.state_dict(), out_path)
        print(f"Saved model: {out_path}")

        # Inference on 10 samples
        preds, truths = collect_10_inference(model, device, Xh_te, Xf_te, y_te)

        # Visualization
        fig_path, mape = plot_inference(model_name, preds, truths, save_dir)
        print(f"[PLOT SAVED] {fig_path}")

        summary.append((model_name, metrics["mae"], metrics["rmse"], metrics["r2"], mape))

    print("\n==================== SUMMARY ====================")
    for model_name, mae, rmse, r2, mape in summary:
        print(f"{model_name:20s} | MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f} | Inference MAPE={mape:.2f}%")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
