"""Zero-phase band-pass and edge trimming. Batch (offline) use only.
The streaming RLS variant needs a causal filter and gets its own function later."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import BAND, FS, TRIM_S


def bandpass(x, fs=FS, band=BAND, order=4):
    """Demean each column, then zero-phase Butterworth band-pass along axis 0."""
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x - x.mean(axis=0), axis=0)


def preprocess(eeg, imu, fs=FS, band=BAND, trim_s=TRIM_S):
    """Filter EEG and IMU identically, then trim both. Returns (eeg, imu)."""
    cut = int(round(trim_s * fs))
    if len(eeg) != len(imu):
        raise ValueError("EEG and IMU lengths differ")
    if len(eeg) <= 2 * cut + int(10 * fs):
        raise ValueError("Recording too short after trimming")
    return bandpass(eeg, fs, band)[cut:-cut], bandpass(imu, fs, band)[cut:-cut]