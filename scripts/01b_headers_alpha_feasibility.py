"""01b_headers_alpha_feasibility.py
Three checks before any pipeline code is written:
  1. Print the true column names of every CSV and flag files whose layout differs.
  2. Confirm the sampling rate and the eyes-closed alpha effect (Relax vs Relaxopen, O1/O2).
  3. Quick feasibility test: can lagged IMU columns predict EEG on HELD-OUT data?
Run from the project root:
  .venv\\Scripts\\python scripts\\01b_headers_alpha_feasibility.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "emotiv"
FIG = ROOT / "results" / "figures" / "inspect"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

FS = 128.0
TRIM_S = 3.0                 # drop edges after filtering (start has a transient in Relax)
MAX_LAG = 32                 # 32 samples = 250 ms at 128 Hz
RIDGE = 10.0                 # ridge strength on standardised features
SKIP_NAMES = ("counter", "interp", "timestamp", "marker", "time")

SOS = butter(4, [1.0, 45.0], btype="bandpass", fs=FS, output="sos")


def load(path):
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df


def split_columns(df):
    eeg = [c for c in df.columns if str(c).startswith("EEG.")]
    other = [c for c in df.columns if c not in eeg]
    motion = [c for c in other if not any(k in str(c).lower() for k in SKIP_NAMES)]
    return eeg, other, motion


def bandpass(x):
    x = x - np.nanmean(x, axis=0)
    return sosfiltfilt(SOS, x, axis=0)


def lagged(X, max_lag):
    n = X.shape[0]
    blocks = []
    for k in range(max_lag + 1):
        shifted = np.zeros_like(X)
        shifted[k:] = X[: n - k]
        blocks.append(shifted)
    return np.hstack(blocks)


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    A = Xs.T @ Xs + lam * len(Xs) * 1e-3 * np.eye(Xs.shape[1])
    W = np.linalg.solve(A, Xs.T @ (Y - ym))
    return lambda Z: ((Z - mu) / sd) @ W + ym


def heldout_r2(X, Y):
    """Two-fold contiguous split. Returns mean R2 over channels and folds."""
    n = len(X)
    half = n // 2
    scores = []
    for tr, te in ((slice(0, half), slice(half, n)), (slice(half, n), slice(0, half))):
        pred = ridge_fit(X[tr], Y[tr], RIDGE)(X[te])
        sse = ((Y[te] - pred) ** 2).sum(0)
        sst = ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
        scores.append(1.0 - sse / sst)
    return float(np.mean(scores)), np.mean(scores, axis=0)


def band_power(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapezoid(p[m], f[m]))


def main():
    files = sorted(RAW.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs under {RAW}")

    # ---------- Check 1: headers ----------
    print("=" * 78)
    print("CHECK 1: column names")
    layouts = {}
    for f in files:
        df = load(f)
        eeg, other, motion = split_columns(df)
        layouts[f.stem] = (df.shape[1], eeg, other, motion)
        print(f"\n{f.stem}: {df.shape[1]} columns, {len(eeg)} EEG")
        print(f"  non-EEG columns ({len(other)}): {other}")
        print(f"  used as IMU ({len(motion)}):     {motion}")
    counts = {v[0] for v in layouts.values()}
    if len(counts) > 1:
        odd = [k for k, v in layouts.items() if v[0] != 23]
        print(f"\n*** Column counts differ across files. Non-23 files: {odd}")

    # ---------- Check 2: sampling rate and alpha ----------
    print("\n" + "=" * 78)
    print("CHECK 2: eyes-closed alpha (Relax) vs eyes-open (Relaxopen), O1 and O2")
    ec_key = next((k for k in layouts if k.lower() == "sub1relax"), None)
    eo_key = next((k for k in layouts if k.lower() == "sub1relaxopen"), None)
    if ec_key and eo_key:
        spectra = {}
        for label, key in (("eyes closed", ec_key), ("eyes open", eo_key)):
            df = load(next(p for p in files if p.stem == key))
            occ = df[["EEG.O1", "EEG.O2"]].to_numpy(float)
            xf = bandpass(occ)[int(TRIM_S * FS): -int(TRIM_S * FS)]
            f_, p_ = welch(xf, fs=FS, nperseg=512, axis=0)
            spectra[label] = (f_, p_.mean(1))
        (f1, pc), (f2, po) = spectra["eyes closed"], spectra["eyes open"]
        m = (f1 >= 7) & (f1 <= 13)
        peak_c = f1[m][np.argmax(pc[m])]
        peak_o = f2[m][np.argmax(po[m])]
        a_c, a_o = band_power(f1, pc, 8, 12), band_power(f2, po, 8, 12)
        print(f"  alpha peak (7-13 Hz):  closed {peak_c:.2f} Hz | open {peak_o:.2f} Hz")
        print(f"  alpha power 8-12 Hz:   closed {a_c:.3g} | open {a_o:.3g} | ratio closed/open {a_c / a_o:.2f}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(f1, pc, label="eyes closed")
        ax.semilogy(f2, po, label="eyes open")
        ax.axvspan(8, 12, alpha=0.15)
        ax.set_xlim(0, 45)
        ax.set_xlabel("Hz")
        ax.set_ylabel("PSD (mean of O1, O2)")
        ax.legend()
        ax.set_title(f"Alpha check at assumed fs = {FS:.0f} Hz")
        fig.tight_layout()
        fig.savefig(FIG / "alpha_check.png", dpi=110)
        plt.close(fig)
    else:
        print("  Could not find both Relax and Relaxopen files. Names seen:", list(layouts))

    # ---------- Check 3: feasibility ----------
    print("\n" + "=" * 78)
    print("CHECK 3: held-out R2 of lagged-IMU ridge regression on EEG (1-45 Hz, 2-fold contiguous)")
    rows = []
    for f in files:
        df = load(f)
        eeg, other, motion = split_columns(df)
        E = bandpass(df[eeg].to_numpy(float))
        M = bandpass(df[motion].to_numpy(float))
        cut = int(TRIM_S * FS)
        E, M = E[cut:-cut], M[cut:-cut]
        r2, per_ch = heldout_r2(lagged(M, MAX_LAG), E)
        rows.append({"file": f.stem, "n_imu_cols": len(motion), "heldout_R2_mean": round(r2, 3),
                     "heldout_R2_best_channel": round(float(per_ch.max()), 3)})
        print(f"  {f.stem:<16} IMU cols {len(motion):>2}   mean R2 {r2:7.3f}   best channel {per_ch.max():7.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "feasibility_r2.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(res["file"], res["heldout_R2_mean"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("held-out R2 (mean over 14 channels)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "feasibility_r2.png", dpi=110)
    plt.close(fig)
    print("\nSaved:", FIG / "alpha_check.png", "|", FIG / "feasibility_r2.png", "|", TAB / "feasibility_r2.csv")


if __name__ == "__main__":
    main()