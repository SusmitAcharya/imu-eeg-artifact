"""Plain-assert checks. Run:  .venv\\Scripts\\python tests\\test_preprocess.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from imu_eeg.config import FS
from imu_eeg.preprocess import bandpass, preprocess


def test_dc_removed_and_alpha_kept():
    t = np.arange(int(60 * FS)) / FS
    x = (5000 + 30 * np.sin(2 * np.pi * 10 * t) + 500 * np.sin(2 * np.pi * 0.1 * t))[:, None]
    y = bandpass(x)[int(5 * FS):-int(5 * FS)]
    assert abs(y.mean()) < 1.0, "DC and 0.1 Hz drift must be removed"
    amp = (y.max() - y.min()) / 2
    assert 28 < amp < 32, f"10 Hz amplitude should survive, got {amp:.2f}"


def test_no_phase_shift():
    t = np.arange(int(60 * FS)) / FS
    x = np.sin(2 * np.pi * 10 * t)[:, None]
    y = bandpass(x)
    seg = slice(int(5 * FS), -int(5 * FS))
    corr = np.corrcoef(x[seg, 0], y[seg, 0])[0, 1]
    assert corr > 0.999, f"zero-phase filter must not shift the signal, corr {corr:.4f}"


def test_shapes_and_trim():
    n = int(45 * FS)
    e, m = preprocess(np.random.randn(n, 14), np.random.randn(n, 6))
    assert e.shape == m.shape[:1] + (14,) and m.shape[1] == 6
    assert len(e) == n - 2 * int(round(3.0 * FS))


if __name__ == "__main__":
    test_dc_removed_and_alpha_kept()
    test_no_phase_shift()
    test_shapes_and_trim()
    print("All 3 tests passed")