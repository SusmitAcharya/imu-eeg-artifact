"""02_preprocess.py
Load every recording by column name, band-pass 1-45 Hz, trim, and save to data/interim/*.npz.
Run from the project root:  .venv\\Scripts\\python scripts\\02_preprocess.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from imu_eeg.config import FS, INTERIM, TABLES
from imu_eeg.io_emotiv import list_recordings, load_recording
from imu_eeg.preprocess import preprocess


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, path in list_recordings().items():
        rec = load_recording(path)
        eeg, imu = preprocess(rec["eeg"], rec["imu"])
        np.savez_compressed(INTERIM / f"{stem}.npz", eeg=eeg, imu=imu, fs=FS,
                            eeg_names=np.array(rec["eeg_names"]), imu_names=np.array(rec["imu_names"]),
                            subject=rec["subject"], label=rec["label"])
        row = {"file": stem, "label": rec["label"], "samples": len(eeg), "seconds": round(len(eeg) / FS, 1),
               "eeg_rms": round(float(np.sqrt((eeg ** 2).mean())), 2),
               "imu_rms": round(float(np.sqrt((imu ** 2).mean())), 2),
               "dropped_columns": ";".join(rec["dropped"])}
        rows.append(row)
        print(f"{stem:<16} {row['samples']:>5} samples  {row['seconds']:>5} s  "
              f"EEG rms {row['eeg_rms']:>8}  IMU rms {row['imu_rms']:>8}  dropped: {row['dropped_columns'] or 'none'}")
    pd.DataFrame(rows).to_csv(INTERIM / "manifest.csv", index=False)
    print(f"\nSaved {len(rows)} files to {INTERIM} and manifest.csv")


if __name__ == "__main__":
    main()