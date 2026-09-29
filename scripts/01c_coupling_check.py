"""01c_coupling_check.py
A stricter test of whether head-motion sensors and EEG are linked at all.
  1. Uses the 14 canonical EEG channels by name (Sub1EM and Sub1Walking carry an extra EEG.* column).
  2. Magnitude-squared coherence between every EEG channel and the 6 gyro/accel columns,
     compared against a time-shifted surrogate (destroys coupling, keeps signal structure).
  3. Ridge regression on 1-8 Hz signals with contiguous blocked cross-validation and a gap.
     The best lambda is reported, so the R2 is an OPTIMISTIC CEILING. If the ceiling is ~0, stop.
Run from the project root:
  .venv\\Scripts\\python scripts\\01c_coupling_check.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "emotiv"
FIG = ROOT / "results" / "figures" / "inspect"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

FS = 128.0
TRIM = int(3.0 * FS)
EEG_NAMES = ["EEG." + n for n in "AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4".split()]
IMU_NAMES = ["MOT.GyroX", "MOT.GyroY", "MOT.GyroZ", "MOT.AccX", "MOT.AccY", "MOT.AccZ"]
BANDS = {"low_1_8Hz": (1.0, 8.0), "mid_8_30Hz": (8.0, 30.0)}
MAX_LAG = 16                       # 125 ms
LAMBDAS = [1e1, 1e2, 1e3, 1e4]
N_FOLDS = 5
GAP = int(1.0 * FS)


def load(path):
    df = pd.read_csv(path)
    return df.apply(pd.to_numeric, errors="coerce")


def bp(x, lo, hi):
    sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x - x.mean(0), axis=0)


def lagged(X, max_lag):
    n = len(X)
    out = []
    for k in range(max_lag + 1):
        s = np.zeros_like(X)
        s[k:] = X[: n - k]
        out.append(s)
    return np.hstack(out)


def coh_matrix(E, M):
    """Coherence spectra averaged inside each band. Returns dict band -> (n_eeg, n_imu)."""
    res = {b: np.zeros((E.shape[1], M.shape[1])) for b in BANDS}
    for i in range(E.shape[1]):
        for j in range(M.shape[1]):
            f, c = coherence(E[:, i], M[:, j], fs=FS, nperseg=256, noverlap=128)
            for b, (lo, hi) in BANDS.items():
                res[b][i, j] = c[(f >= lo) & (f <= hi)].mean()
    return res


def coh_stat(mat):
    """Mean over the 3 most coupled EEG channels of their best IMU column.
    Motion coupling is channel-specific, so a median over all channels would hide it.
    The surrogate uses the same statistic, so the selection bias cancels in the excess."""
    return float(np.sort(mat.max(axis=1))[-3:].mean())


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (Y - ym))
    return lambda Z: ((Z - mu) / sd) @ W + ym


def blocked_r2(X, Y, lam):
    n = len(X)
    edges = np.linspace(0, n, N_FOLDS + 1).astype(int)
    sse = np.zeros(Y.shape[1])
    sst = np.zeros(Y.shape[1])
    for i in range(N_FOLDS):
        a, b = edges[i], edges[i + 1]
        train = np.ones(n, bool)
        train[max(0, a - GAP): min(n, b + GAP)] = False
        pred = ridge_fit(X[train], Y[train], lam)(X[a:b])
        sse += ((Y[a:b] - pred) ** 2).sum(0)
        sst += ((Y[a:b] - Y[a:b].mean(0)) ** 2).sum(0)
    return 1.0 - sse / sst


def main():
    files = sorted(RAW.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs under {RAW}")
    rows = []
    for f in files:
        df = load(f)
        missing = [c for c in EEG_NAMES + IMU_NAMES if c not in df.columns]
        if missing:
            raise SystemExit(f"{f.stem}: missing columns {missing}")
        extras = [c for c in df.columns if c.startswith("EEG.") and c not in EEG_NAMES]
        print("=" * 78)
        print(f.stem)
        for c in extras:
            s = df[c]
            print(f"  EXTRA EEG-prefixed column {c!r}: mean {s.mean():.1f}  std {s.std():.1f}  "
                  f"min {s.min():.1f}  max {s.max():.1f}  unique values {s.nunique()}")

        E = bp(df[EEG_NAMES].to_numpy(float), 1.0, 45.0)[TRIM:-TRIM]
        M = bp(df[IMU_NAMES].to_numpy(float), 1.0, 45.0)[TRIM:-TRIM]
        n = len(E)

        real = coh_matrix(E, M)
        null = [coh_matrix(E, np.roll(M, s, axis=0)) for s in (n // 3, 2 * n // 3)]
        row = {"file": f.stem, "n_extra_eeg_cols": len(extras)}
        for b in BANDS:
            r = coh_stat(real[b])
            z = float(np.mean([coh_stat(m[b]) for m in null]))
            row[f"{b}_coh_real"] = round(r, 3)
            row[f"{b}_coh_null"] = round(z, 3)
            row[f"{b}_coh_excess"] = round(r - z, 3)
        best_imu = IMU_NAMES[int(np.argmax(real["low_1_8Hz"].mean(0)))]
        row["best_imu_low"] = best_imu

        El = bp(df[EEG_NAMES].to_numpy(float), 1.0, 8.0)[TRIM:-TRIM]
        Ml = bp(df[IMU_NAMES].to_numpy(float), 1.0, 8.0)[TRIM:-TRIM]
        X = lagged(Ml, MAX_LAG)
        best = None
        for lam in LAMBDAS:
            r2 = blocked_r2(X, El, lam)
            if best is None or r2.mean() > best[1].mean():
                best = (lam, r2)
        lam, r2 = best
        k = int(np.argmax(r2))
        row.update({"ridge_ceiling_R2_mean": round(float(r2.mean()), 3), "best_lambda": lam,
                    "best_channel": EEG_NAMES[k].replace("EEG.", ""), "best_channel_R2": round(float(r2[k]), 3)})
        rows.append(row)
        print(f"  coherence excess  low 1-8 Hz {row['low_1_8Hz_coh_excess']:+.3f} "
              f"(real {row['low_1_8Hz_coh_real']:.3f}, null {row['low_1_8Hz_coh_null']:.3f}) | "
              f"mid 8-30 Hz {row['mid_8_30Hz_coh_excess']:+.3f}")
        print(f"  strongest IMU column (low band): {best_imu}")
        print(f"  ridge ceiling (1-8 Hz, blocked CV): mean R2 {row['ridge_ceiling_R2_mean']:+.3f} at lambda {lam:g}; "
              f"best channel {row['best_channel']} R2 {row['best_channel_R2']:+.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(TAB / "coupling_check.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(res))
    axes[0].bar(x - 0.2, res["low_1_8Hz_coh_excess"], 0.4, label="1-8 Hz")
    axes[0].bar(x + 0.2, res["mid_8_30Hz_coh_excess"], 0.4, label="8-30 Hz")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Coherence above time-shifted surrogate")
    axes[0].legend()
    axes[1].bar(x, res["ridge_ceiling_R2_mean"])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("Ridge R2 ceiling, 1-8 Hz, blocked CV (mean of 14 channels)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(res["file"], rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "coupling_check.png", dpi=110)
    plt.close(fig)
    print("\nSaved:", TAB / "coupling_check.csv", "|", FIG / "coupling_check.png")


if __name__ == "__main__":
    main()