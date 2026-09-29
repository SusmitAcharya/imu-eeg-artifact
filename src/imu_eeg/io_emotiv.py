"""Load one Emotiv recording CSV by column NAME, never by position."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EEG_NAMES, IMU_NAMES, RAW_EMOTIV


def list_recordings(root=RAW_EMOTIV):
    """Return {file stem: path} for every CSV under root, sorted by stem."""
    files = sorted(Path(root).rglob("*.csv"), key=lambda p: p.stem)
    if not files:
        raise FileNotFoundError(f"No CSV files under {root}")
    return {p.stem: p for p in files}


def split_stem(stem):
    """'Sub1Relaxopen' -> ('Sub1', 'Relaxopen')."""
    m = re.match(r"^(Sub\d+)(.+)$", stem)
    if not m:
        raise ValueError(f"Unexpected file name: {stem}")
    return m.group(1), m.group(2)


def load_recording(path):
    """Return a dict with eeg (n, 14), imu (n, 6), names, subject, label, and dropped columns."""
    path = Path(path)
    df = pd.read_csv(path).apply(pd.to_numeric, errors="coerce")
    missing = [c for c in EEG_NAMES + IMU_NAMES if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing columns {missing}")
    eeg = df[EEG_NAMES].to_numpy(float)
    imu = df[IMU_NAMES].to_numpy(float)
    if not (np.isfinite(eeg).all() and np.isfinite(imu).all()):
        raise ValueError(f"{path.name}: NaN or inf in used columns")
    subject, label = split_stem(path.stem)
    dropped = [c for c in df.columns if c not in EEG_NAMES + IMU_NAMES]
    return {"eeg": eeg, "imu": imu, "eeg_names": list(EEG_NAMES), "imu_names": list(IMU_NAMES),
            "subject": subject, "label": label, "dropped": dropped}