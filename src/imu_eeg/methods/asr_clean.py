"""Artifact Subspace Reconstruction, via meegkit (primary) with asrpy as fallback.

asrpy 0.0.8's calibration routine (fit_eeg_distribution) calls int() on a
single-element NumPy array, which NumPy >=2.0 no longer allows to implicitly
convert (TypeError: only 0-dimensional arrays can be converted to Python
scalars). This breaks asrpy under the numpy 2.x pin this project otherwise
needs (see config/ways-of-working notes on numpy.trapezoid). meegkit.asr.ASR
implements the same Mullen et al. (2015) algorithm and is numpy-2.x-compatible,
so it is the primary path.
"""
import numpy as np

try:
    from meegkit.asr import ASR as _ASR
    _BACKEND = "meegkit"
except ImportError:
    from asrpy import ASR as _ASR
    _BACKEND = "asrpy"


def clean(eeg, fs=128.0, calib_eeg=None, cutoff=20.0):
    """eeg, calib_eeg: (n, k) arrays in microvolts, SAME channel set and order.
    calib_eeg should be the Relax (or Relaxopen) recording for this subject.
    meegkit wants (channels, samples); this function keeps the project's
    (samples, channels) convention at the call site and transposes internally.
    """
    if _BACKEND == "meegkit":
        asr = _ASR(sfreq=fs, cutoff=cutoff)
        asr.fit(calib_eeg.T)
        return asr.transform(eeg.T).T
    else:  # asrpy fallback: needs an MNE Raw object, not a bare array
        import mne
        info = mne.create_info([f"ch{i}" for i in range(eeg.shape[1])], sfreq=fs, ch_types="eeg")
        calib_raw = mne.io.RawArray(calib_eeg.T * 1e-6, info, verbose=False)
        target_raw = mne.io.RawArray(eeg.T * 1e-6, info, verbose=False)
        asr = _ASR(sfreq=fs, cutoff=cutoff)
        asr.fit(calib_raw)
        return asr.transform(target_raw).get_data().T * 1e6


BACKEND = _BACKEND