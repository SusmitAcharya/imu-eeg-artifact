"""03_run_methods.py
Run all 4 methods (baseline, ICA, ASR, IMU-regression batch+RLS) across
14/8/4-channel subsets on all 12 recordings, scoring against the 3
hypotheses in docs/hypotheses.md. Writes results/tables/method_results.csv.

Requires scripts/02_preprocess.py to have been run first (reads data/interim/*.npz).
Run from the project root:  .venv\\Scripts\\python scripts\\03_run_methods.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from imu_eeg.config import EEG_NAMES, FS, INTERIM, TABLES
from imu_eeg.methods import asr_clean, baseline, ica_clean, imu_regression
from imu_eeg.metrics import alpha_contrast, imu_predictability_residual, relax_psd, spectral_fidelity
from imu_eeg.subsample import subset_indices

CHANNEL_COUNTS = [14, 8, 4]
METHODS = ["baseline", "ica", "asr", "imu_batch", "imu_rls"]


def load_all():
    files = sorted(INTERIM.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {INTERIM}. Run scripts/02_preprocess.py first.")
    return {f.stem: dict(np.load(f, allow_pickle=True)) for f in files}


def run_one(eeg, imu, ch_names_full, label, calib_eeg=None, calib_relax_psd=None):
    """eeg: (n, k) for one channel subset. Returns list of result dicts, one per method."""
    ch_names_bare = [c.replace("EEG.", "") for c in ch_names_full]
    n = len(eeg)
    half = n // 2
    rows = []

    def score(name, cleaned_full, true_full, imu_seg):
        """cleaned_full/true_full: same length as imu_seg, held-out portion only.
        PRIMARY metrics for the paper: alpha_contrast (H3, do-no-harm) and
        spectral_fidelity_D_PSD (H1/H2, how far the cleaned motion-file spectrum
        sits from this subject's own Relax spectrum -- lower is better/cleaner).
        residual_imu_r2 is DIAGNOSTIC ONLY: it re-fits IMU->cleaned_EEG and reports
        that fit's own held-out R2. It is unstable on these short recordings (see
        01b/01c checks) and does not distinguish "artifact removed" from "signal
        destroyed" -- do not use it to rank methods in the paper."""
        r2 = imu_predictability_residual(true_full, cleaned_full, imu_seg)
        contrast = alpha_contrast(cleaned_full, ch_names_full)
        fid = spectral_fidelity(cleaned_full, calib_relax_psd) if calib_relax_psd is not None else np.nan
        rows.append({"method": name, "residual_imu_r2_DIAGNOSTIC_ONLY": round(r2, 4),
                     "alpha_contrast": round(contrast, 4) if not np.isnan(contrast) else np.nan,
                     "spectral_fidelity_D_PSD": round(fid, 4) if not np.isnan(fid) else np.nan})

    score("baseline", baseline.clean(eeg[half:]), eeg[half:], imu[half:])

    try:
        cleaned_ica, excluded = ica_clean.clean(eeg, imu, ch_names_bare, fs=FS)
        score("ica", cleaned_ica[half:], eeg[half:], imu[half:])
        rows[-1]["ica_excluded_components"] = len(excluded)
    except Exception as e:
        rows.append({"method": "ica", "error": str(e)})

    if calib_eeg is not None:
        try:
            cleaned_asr = asr_clean.clean(eeg, fs=FS, calib_eeg=calib_eeg, cutoff=20.0)
            score("asr", cleaned_asr[half:], eeg[half:], imu[half:])
        except Exception as e:
            rows.append({"method": "asr", "error": str(e)})

    cleaned_batch, true_batch, n_train = imu_regression.clean_batch(eeg, imu, max_lag=32, lam=100.0)
    score("imu_batch", cleaned_batch, true_batch, imu[n_train:])

    rls = imu_regression.RLSCleaner(n_features=imu.shape[1] * 33, n_outputs=eeg.shape[1], lam=100.0)
    cleaned_rls, _ = rls.run(eeg, imu, max_lag=32)
    score("imu_rls", cleaned_rls[half:], eeg[half:], imu[half:])

    return rows


def main():
    data = load_all()
    all_rows = []
    for stem, rec in data.items():
        label = str(rec["label"])
        eeg_full, imu = rec["eeg"], rec["imu"]
        relax_stem = next((s for s in data if str(data[s]["label"]) == "Relax"
                            and str(data[s]["subject"]) == str(rec["subject"])), None)
        calib_eeg = data[relax_stem]["eeg"] if relax_stem else None
        _, r_psd = relax_psd(calib_eeg) if calib_eeg is not None else (None, None)

        for k in CHANNEL_COUNTS:
            idx, names_k = subset_indices(k)
            eeg_k = eeg_full[:, idx]
            calib_k = calib_eeg[:, idx] if calib_eeg is not None else None
            r_psd_k = r_psd[:, [EEG_NAMES.index(c) for c in names_k]] if r_psd is not None else None

            print(f"{stem:<16} ({label})  {k:>2} ch ...", end=" ", flush=True)
            rows = run_one(eeg_k, imu, names_k, label, calib_eeg=calib_k, calib_relax_psd=r_psd_k)
            for r in rows:
                r.update({"file": stem, "label": label, "n_channels": k})
                all_rows.append(r)
            ok = [r["method"] for r in rows if "error" not in r]
            print(f"ok: {ok}")

    df = pd.DataFrame(all_rows)
    out = TABLES / "method_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")
    print("\nspectral_fidelity_D_PSD (lower = closer to Relax = more artifact removed), by n_channels/method:")
    print(df.groupby(["n_channels", "method"])["spectral_fidelity_D_PSD"].mean().round(3))
    print("\nalpha_contrast (H3 do-no-harm; only defined at 14 channels, O1/O2 not in 8/4-ch subsets):")
    print(df[df.n_channels == 14].groupby(["label", "method"])["alpha_contrast"].mean().round(3))
    print("\nresidual_imu_r2_DIAGNOSTIC_ONLY (do not rank methods on this -- see docstring in score()):")
    print(df.groupby(["n_channels", "method"])["residual_imu_r2_DIAGNOSTIC_ONLY"].mean().round(3))


if __name__ == "__main__":
    main()