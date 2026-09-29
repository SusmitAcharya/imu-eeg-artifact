"""ICA artifact removal via MNE. Components are flagged by correlation with the
IMU channels (an IMU-aware selector), since ICLabel is unvalidated on
non-standard low channel counts and 14/8/4-channel EPOC montages are outside
its training distribution. See methods/README or project notes.
"""
import numpy as np
import mne


def clean(eeg, imu, ch_names, fs=128.0, corr_thresh=0.3, random_state=0):
    """eeg: (n, k). ch_names: list[str] of length k, WITHOUT the 'EEG.' prefix
    (e.g. 'AF3'), matching an MNE standard_1020 montage label."""
    n, k = eeg.shape
    info = mne.create_info(ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(eeg.T * 1e-6, info, verbose=False)  # assume EPOC units -> volts
    raw.set_montage("standard_1020", on_missing="warn", verbose=False)

    n_components = min(k, max(2, k - 1))
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state,
                                 max_iter="auto", verbose=False)
    ica.fit(raw, verbose=False)
    sources = ica.get_sources(raw).get_data()  # (n_components, n)

    bad = []
    for c in range(n_components):
        r = max(abs(np.corrcoef(sources[c], imu[:, j])[0, 1]) for j in range(imu.shape[1]))
        if r > corr_thresh:
            bad.append(c)
    ica.exclude = bad
    cleaned = ica.apply(raw.copy(), verbose=False)
    return cleaned.get_data().T * 1e6, bad