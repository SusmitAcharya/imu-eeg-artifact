"""Shared constants. Every script imports from here so a value changes in one place only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_EMOTIV = ROOT / "data" / "raw" / "emotiv"
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "results" / "tables"

FS = 128.0                    # confirmed by the 10.25 Hz alpha peak in Relax
BAND = (1.0, 45.0)            # zero-phase band-pass applied to EEG and IMU alike
TRIM_S = 3.0                  # seconds cut from each end after filtering (start-up transients)

# The 14 EPOC channels, in file order. Two files carry an extra EEG.RawCq column that is NOT EEG.
EEG_NAMES = ["EEG." + n for n in "AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4".split()]
# Gyroscope and accelerometer only. Magnetometer columns are excluded on purpose.
IMU_NAMES = ["MOT.GyroX", "MOT.GyroY", "MOT.GyroZ", "MOT.AccX", "MOT.AccY", "MOT.AccZ"]

CHANNEL_SUBSETS = {
    14: EEG_NAMES,
    8:  ["EEG.AF3", "EEG.AF4", "EEG.T7", "EEG.T8", "EEG.F7", "EEG.F8", "EEG.P7", "EEG.P8"],
    4:  ["EEG.AF3", "EEG.AF4", "EEG.T7", "EEG.T8"],
}