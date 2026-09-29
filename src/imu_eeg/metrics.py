"""Scoring functions for the three pre-specified hypotheses in docs/hypotheses.md.

H1 / H2 (kinematic vs ocular-speech groups): held-out residual R2 against IMU,
  scored on a contiguous fold the fitting method never saw. This is NOT the
  same in-sample check that project notes flagged as circular -- every method
  here is scored on data disjoint from whatever it was calibrated/fit on.
H3 (do-no-harm): eyes-closed vs eyes-open alpha contrast at O1/O2 must survive
  cleaning. Measured as (alpha - flanking-band) contrast, not raw alpha power,
  per the caution that the raw closed/open ratio in the 01b check was inflated
  by broadband differences, not an alpha-specific effect.
Spectral fidelity: distance to the subject's own Relax spectrum (WearBCI's
  D_PSD), so numbers stay comparable if we replicate on WearBCI later.
"""
import numpy as np
from scipy.signal import welch

from .config import EEG_NAMES, FS


def imu_predictability_residual(eeg_true, eeg_cleaned, imu, max_lag=32):
    """DIAGNOSTIC ONLY -- do not use this to rank cleaning methods in the paper.

    Fits a FRESH IMU->eeg_cleaned ridge regression and reports that fit's own
    held-out R2: how much of the CLEANED EEG is still explained by lagged IMU.
    This scores eeg_cleaned directly (not eeg_true - eeg_cleaned): for baseline,
    cleaned == true, so this reports the ORIGINAL coupling; for a subtractive
    method, it reports what coupling is LEFT after subtraction. Lower is better
    IN PRINCIPLE, but a method that destroys real EEG alongside the artifact
    also scores well here, since it leaves the residual just as decorrelated
    from IMU. It does not distinguish "artifact removed" from "signal
    destroyed." It is also numerically unstable on these ~35-41s recordings
    (198 regressors: 6 IMU channels x 33 lags, ~2000 held-out samples per fold
    -- see the 01b/01c diagnostic checks, where the same regression swung to
    R2 = -22.9 on one file). Use alpha_contrast (H3) and spectral_fidelity (H1/H2)
    as the primary metrics instead; keep this one around only as a secondary,
    clearly-labelled diagnostic column.

    eeg_true is accepted for signature symmetry with callers but unused here.
    Caller is responsible for passing only held-out samples."""
    from .methods.imu_regression import build_lagged, fit_ridge, predict_ridge
    n = len(eeg_cleaned)
    half = n // 2
    X = build_lagged(imu, max_lag)
    scores = []
    for tr, te in ((slice(0, half), slice(half, n)), (slice(half, n), slice(0, half))):
        W, mu_x, sd_x, mu_y = fit_ridge(X[tr], eeg_cleaned[tr], lam=100.0)
        pred = predict_ridge(X[te], W, mu_x, sd_x, mu_y)
        sse = ((eeg_cleaned[te] - pred) ** 2).sum()
        sst = ((eeg_cleaned[te] - eeg_cleaned[te].mean(0)) ** 2).sum()
        scores.append(1.0 - sse / sst)
    return float(np.mean(scores))


def alpha_contrast(eeg, ch_names, fs=FS, band=(8, 12), flank=(13, 20)):
    """Alpha power at O1/O2 relative to a flanking band, so a uniform gain
    change from cleaning doesn't masquerade as an alpha effect."""
    idx = [ch_names.index(c) for c in ("EEG.O1", "EEG.O2") if c in ch_names]
    if not idx:
        return np.nan
    occ = eeg[:, idx]
    f, p = welch(occ, fs=fs, nperseg=min(512, len(occ)), axis=0)
    def band_power(lo, hi):
        m = (f >= lo) & (f <= hi)
        return float(np.trapezoid(p[m], f[m], axis=0).mean())
    return band_power(*band) / (band_power(*flank) + 1e-12)


def spectral_fidelity(eeg, relax_psd, fs=FS, band=(1, 45)):
    """D_PSD: mean absolute log-spectral distance to the subject's own Relax
    spectrum (per channel, then averaged). Matches WearBCI's evaluation so
    results stay comparable if we replicate there. Lower is better."""
    f, p = welch(eeg, fs=fs, nperseg=min(512, len(eeg)), axis=0)
    m = (f >= band[0]) & (f <= band[1])
    logp = np.log(p[m] + 1e-12)
    logr = np.log(relax_psd[m] + 1e-12)
    return float(np.abs(logp - logr).mean())


def relax_psd(eeg, fs=FS):
    """Reference PSD from a subject's own Relax recording, for spectral_fidelity."""
    f, p = welch(eeg, fs=fs, nperseg=min(512, len(eeg)), axis=0)
    return f, p