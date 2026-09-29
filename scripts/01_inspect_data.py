"""01_inspect_data.py
Discover every CSV under data/raw/emotiv, report its structure, and plot it.
Purpose: confirm the column layout (14 EEG then 9 motion, 128 Hz) with real evidence.
Run from the project root:  .venv\\Scripts\\python scripts\\01_inspect_data.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "emotiv"
FIG = ROOT / "results" / "figures" / "inspect"
INTERIM = ROOT / "data" / "interim"
FIG.mkdir(parents=True, exist_ok=True)
INTERIM.mkdir(parents=True, exist_ok=True)

FS = 128.0          # stated in the Data in Brief article for EEG and motion sensors
N_EEG = 14          # article: first 14 columns are EEG
N_MOTION = 9        # article: next 9 columns are motion sensors
EEG_NAMES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
             "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def load_numeric(path):
    """Read a CSV with no header assumption. Return numeric frame and skipped header text."""
    raw = pd.read_csv(path, header=None, low_memory=False)
    num = raw.apply(pd.to_numeric, errors="coerce")
    is_data_row = num.notna().mean(axis=1) > 0.9
    if not is_data_row.any():
        raise ValueError(f"No numeric rows found in {path.name}")
    first = int(np.argmax(is_data_row.values))
    header_text = raw.iloc[:first].astype(str).values.tolist()
    data = num.loc[is_data_row].dropna(axis=1, how="all").reset_index(drop=True)
    return data, header_text, first


def plot_file(name, data, path_out):
    x = data.to_numpy(dtype=float)
    n = min(len(x), int(10 * FS))            # first 10 seconds
    t = np.arange(n) / FS
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    eeg = x[:n, :N_EEG] - np.nanmean(x[:, :N_EEG], axis=0)
    step = 4 * np.nanstd(eeg) if np.nanstd(eeg) > 0 else 1
    for i in range(min(N_EEG, eeg.shape[1])):
        axes[0].plot(t, eeg[:, i] + i * step, lw=0.6)
    axes[0].set_title(f"{name}: EEG columns 0 to {N_EEG - 1} (demeaned, stacked)")
    axes[0].set_yticks([])
    mot = x[:n, N_EEG:N_EEG + N_MOTION]
    mot = mot - np.nanmean(x[:, N_EEG:N_EEG + N_MOTION], axis=0)
    for j in range(mot.shape[1]):
        axes[1].plot(t, mot[:, j], lw=0.7, label=f"col {N_EEG + j}")
    axes[1].set_title("Motion columns (demeaned)")
    axes[1].set_xlabel("seconds")
    axes[1].legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(path_out, dpi=110)
    plt.close(fig)


def main():
    files = sorted(RAW.rglob("*.csv"))
    print(f"Found {len(files)} CSV files under {RAW}\n")
    if not files:
        raise SystemExit("No CSVs found. Check the unpack step.")
    rows = []
    for f in files:
        data, header_text, first = load_numeric(f)
        n_rows, n_cols = data.shape
        print("=" * 78)
        print(f"{f.name}")
        print(f"  rows: {n_rows}   columns: {n_cols}   duration at {FS:.0f} Hz: {n_rows / FS:.1f} s")
        print(f"  non-data lines skipped at top: {first}")
        if header_text:
            print(f"  header preview: {header_text[0][:8]}")
        print(f"  NaN cells: {int(data.isna().sum().sum())}")
        desc = data.describe().T[["mean", "std", "min", "max"]].round(2)
        print(desc.to_string())
        plot_file(f.stem, data, FIG / f"{f.stem}.png")
        rows.append({"file": f.name, "rows": n_rows, "cols": n_cols,
                     "seconds_at_128Hz": round(n_rows / FS, 1)})
    pd.DataFrame(rows).to_csv(INTERIM / "inspect_summary.csv", index=False)
    print("\nSaved plots to", FIG)
    print("Saved summary to", INTERIM / "inspect_summary.csv")


if __name__ == "__main__":
    main()